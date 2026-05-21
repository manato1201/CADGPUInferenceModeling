"""
core/parser.py
==============
DXF/DWG ファイルを読み込み、三面図画像とメタデータを返す純粋関数群。

依存: ezdxf, matplotlib, numpy, Pillow
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ezdxf
import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境向け
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# データ型定義
# ─────────────────────────────────────────────

@dataclass
class CADMeta:
    """CAD図面から抽出したメタデータ。"""
    source_path: str
    layers: list[str] = field(default_factory=list)
    # 全エンティティのワールド座標 bounding box (min/max)
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    unit: str = "mm"  # DXF $INSUNITS から推定
    entity_count: int = 0

    @property
    def dimensions(self) -> np.ndarray:
        """X/Y/Z の寸法 [mm] を返す。"""
        return self.bbox_max - self.bbox_min

    @property
    def aspect_ratio(self) -> tuple[float, float, float]:
        """X:Y:Z の比率（最大辺を1に正規化）。"""
        d = self.dimensions
        m = d.max() if d.max() > 0 else 1.0
        return tuple((d / m).tolist())


@dataclass
class CADParseResult:
    """パース結果をまとめて保持する。"""
    meta: CADMeta
    # 三面図: {"front": PIL.Image, "side": PIL.Image, "top": PIL.Image}
    views: dict[str, Image.Image] = field(default_factory=dict)
    # 生のエンティティポイント群（点群プレビュー用）
    point_cloud: Optional[np.ndarray] = None


# ─────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────

_INSUNITS_TO_MM = {
    0: 1.0,    # 未定義→そのまま
    1: 25.4,   # inch
    2: 304.8,  # foot
    4: 1.0,    # mm
    5: 10.0,   # cm
    6: 1000.0, # m
}


def _get_unit_scale(doc: ezdxf.document.Drawing) -> tuple[float, str]:
    """$INSUNITS からスケール係数と単位名を返す。"""
    code = doc.header.get("$INSUNITS", 4)
    scale = _INSUNITS_TO_MM.get(code, 1.0)
    name = {4: "mm", 5: "cm", 6: "m", 1: "inch", 2: "foot"}.get(code, "unit")
    return scale, name


def _collect_points(msp, scale: float) -> np.ndarray:
    """
    modelspace の全エンティティから代表点を収集して (N,3) ndarray を返す。
    LINE / LWPOLYLINE / CIRCLE / ARC / SPLINE を対象とする。
    """
    pts: list[np.ndarray] = []

    for e in msp:
        dxftype = e.dxftype()
        try:
            if dxftype == "LINE":
                pts.append(np.array([e.dxf.start.x, e.dxf.start.y, e.dxf.start.z]))
                pts.append(np.array([e.dxf.end.x,   e.dxf.end.y,   e.dxf.end.z]))

            elif dxftype == "LWPOLYLINE":
                for x, y, *_ in e.get_points():
                    pts.append(np.array([x, y, 0.0]))

            elif dxftype in ("CIRCLE", "ARC"):
                cx, cy, cz = e.dxf.center.x, e.dxf.center.y, e.dxf.center.z
                r = e.dxf.radius
                # 8点で近似
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    pts.append(np.array([cx + r * np.cos(a), cy + r * np.sin(a), cz]))

            elif dxftype == "SPLINE":
                for p in e.control_points:
                    pts.append(np.array([p.x, p.y, p.z]))

        except Exception:
            # 読み取れないエンティティはスキップ
            continue

    if not pts:
        return np.zeros((1, 3))

    arr = np.array(pts, dtype=np.float64) * scale
    return arr


def _collect_lines(msp, scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """LINE エンティティのリストを [(start, end), ...] で返す。"""
    lines = []
    for e in msp:
        if e.dxftype() == "LINE":
            try:
                s = np.array([e.dxf.start.x, e.dxf.start.y, e.dxf.start.z]) * scale
                t = np.array([e.dxf.end.x,   e.dxf.end.y,   e.dxf.end.z])   * scale
                lines.append((s, t))
            except Exception:
                continue
        elif e.dxftype() == "LWPOLYLINE":
            try:
                verts = [(x * scale, y * scale) for x, y, *_ in e.get_points()]
                for i in range(len(verts) - 1):
                    s = np.array([verts[i][0],   verts[i][1],   0.0])
                    t = np.array([verts[i+1][0], verts[i+1][1], 0.0])
                    lines.append((s, t))
            except Exception:
                continue
    return lines


def _render_view(
    lines: list[tuple[np.ndarray, np.ndarray]],
    pts: np.ndarray,
    axis_h: int,
    axis_v: int,
    title: str,
    image_size: int,
    margin: float = 0.05,
) -> Image.Image:
    """
    lines を指定軸に投影して matplotlib でレンダリングし PIL.Image を返す。

    axis_h: 水平方向として使う座標軸インデックス (0=X, 1=Y, 2=Z)
    axis_v: 垂直方向として使う座標軸インデックス
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=image_size // 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    segs = []
    for s, t in lines:
        segs.append([(s[axis_h], s[axis_v]), (t[axis_h], t[axis_v])])

    if segs:
        lc = LineCollection(segs, linewidths=0.8, colors="black")
        ax.add_collection(lc)

    # パディング付きで表示範囲を決定
    if pts.shape[0] > 1:
        xmin, xmax = pts[:, axis_h].min(), pts[:, axis_h].max()
        ymin, ymax = pts[:, axis_v].min(), pts[:, axis_v].max()
        xpad = max((xmax - xmin) * margin, 1e-3)
        ypad = max((ymax - ymin) * margin, 1e-3)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.set_title(title, fontsize=8, pad=2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=image_size // 4)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return img.resize((image_size, image_size), Image.LANCZOS)


# ─────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────

def parse_dxf(
    path: str | Path,
    image_size: int = 512,
) -> CADParseResult:
    """
    DXF ファイルを読み込み CADParseResult を返す。

    Parameters
    ----------
    path       : DXF ファイルパス
    image_size : 出力画像の一辺ピクセル数（Zero123++ 推奨: 512）

    Returns
    -------
    CADParseResult
        .meta       - メタデータ
        .views      - {"front", "side", "top"} の三面図 PIL.Image
        .point_cloud - (N,3) ndarray [mm]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DXF file not found: {path}")

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    scale, unit = _get_unit_scale(doc)
    pts = _collect_points(msp, scale)
    lines = _collect_lines(msp, scale)

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)

    layers = [layer.dxf.name for layer in doc.layers]

    meta = CADMeta(
        source_path=str(path),
        layers=layers,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        unit=unit,
        entity_count=len(list(msp)),
    )

    views = {
        # 正面図: X(水平) × Z(垂直)
        "front": _render_view(lines, pts, axis_h=0, axis_v=2, title="Front (XZ)", image_size=image_size),
        # 側面図: Y(水平) × Z(垂直)
        "side":  _render_view(lines, pts, axis_h=1, axis_v=2, title="Side (YZ)",  image_size=image_size),
        # 上面図: X(水平) × Y(垂直)
        "top":   _render_view(lines, pts, axis_h=0, axis_v=1, title="Top (XY)",   image_size=image_size),
    }

    return CADParseResult(meta=meta, views=views, point_cloud=pts)


def save_views(result: CADParseResult, output_dir: str | Path) -> dict[str, Path]:
    """三面図を PNG として保存し、保存パスの dict を返す。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, img in result.views.items():
        p = output_dir / f"{name}.png"
        img.save(p)
        saved[name] = p
    return saved
