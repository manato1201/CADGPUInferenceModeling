"""構成図の SVG を作る小さな道具(依存なし)。docs/diagrams/ の図は tools/make_diagrams.py が、これで作る。

見た目は、格子の背景・角丸のグループ枠・アイコンのカード・ラベルつきの矢印(主役の流れは青、補助は灰の破線、強調はオレンジ)。
色は CSS の変数にしてあり、ダークモード(prefers-color-scheme: dark)に追従する。文字は埋め込みフォントを使わず、システムのフォント。

使い方::

    d = Diagram(1200, 700, "題名", "図の説明(読み上げ用)")
    g = d.group(40, 40, 500, 300, "BROWSER", color="blue")
    a = d.card(120, 160, "browser", "ページ", "youtube.com")      # 中心の座標
    b = d.card(400, 160, "server", "エンジン", "127.0.0.1")
    d.arrow([a.right, b.left], "HTTP")
    open("x.svg", "w", encoding="utf-8").write(d.svg())

card の戻り値(Card)には、top / bottom / left / right(矢印をつなぐ点)がある。
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

CARD = 64  # カードの一辺

CSS = """
:root{--bg:#ffffff;--grid:#edf0f4;--ink:#1d2433;--muted:#667085;--card:#ffffff;--cardline:#d0d5dd;--lab:#ffffff;
--blue:#1a73e8;--orange:#e8710a;--teal:#0f9d8a;--red:#d92d20;--gray:#667085;--purple:#7a5af8;
--f-blue:#f3f7fe;--f-orange:#fdf6ee;--f-teal:#f0f9f7;--f-red:#fef5f4;--f-gray:#f6f7f9;--f-purple:#f6f3ff}
@media (prefers-color-scheme: dark){:root{--bg:#14171d;--grid:#1f242d;--ink:#e6e9ef;--muted:#98a2b3;--card:#1b2029;--cardline:#3a4150;--lab:#14171d;
--blue:#6ea8fe;--orange:#f5a25d;--teal:#52c7b4;--red:#f97066;--gray:#98a2b3;--purple:#a99bff;
--f-blue:#17202e;--f-orange:#271f17;--f-teal:#152422;--f-red:#2a1b1a;--f-gray:#1a1e26;--f-purple:#1f1b2e}}
text{font-family:"Noto Sans JP","Hiragino Sans","Yu Gothic UI","Meiryo",system-ui,sans-serif;fill:var(--ink)}
.mono{font-family:"JetBrains Mono","Cascadia Mono",Consolas,"Courier New",monospace}
.title{font-size:16px;font-weight:600;text-anchor:middle}
.sub{font-size:11.5px;text-anchor:middle;fill:var(--muted)}
.glabel{font-size:12.5px;font-weight:700;letter-spacing:.08em}
.alabel{font-size:12px;text-anchor:middle;fill:var(--muted)}
.alabel.blue{fill:var(--blue)} .alabel.orange{fill:var(--orange)} .alabel.red{fill:var(--red)}
.note{font-size:12px;fill:var(--muted)}
.hd{font-size:20px;font-weight:700}
.card{fill:var(--card);stroke:var(--cardline);stroke-width:1.2}
.ic{fill:none;stroke:var(--ink);stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}
.ic .fill{fill:var(--ink);stroke:none}
.lab{fill:var(--lab)}
.a{fill:none;stroke-width:2}
.a.primary{stroke:var(--blue)} .a.secondary{stroke:var(--gray);stroke-dasharray:6 5} .a.accent{stroke:var(--orange)} .a.danger{stroke:var(--red);stroke-dasharray:6 5}
.h.primary{fill:var(--blue)} .h.secondary{fill:var(--gray)} .h.accent{fill:var(--orange)} .h.danger{fill:var(--red)}
.badge{font-size:11px;font-weight:700;text-anchor:middle;fill:#fff}
"""

# 40x40 の枠に描く、線のアイコン。class="ic" の中に入る。
GLYPHS = {
    "browser": '<rect x="4" y="7" width="32" height="26" rx="4"/><path d="M4 15h32"/><circle class="fill" cx="9" cy="11" r="1.2"/><circle class="fill" cx="13.5" cy="11" r="1.2"/>',
    "extension": '<path d="M9 13h8a4 4 0 1 1 8 0h6v8a4 4 0 1 0 0 8v5H9z"/>',
    "gear": '<circle cx="20" cy="20" r="6"/><path d="M20 4v6M20 30v6M4 20h6M30 20h6M8.7 8.7l4.2 4.2M27.1 27.1l4.2 4.2M31.3 8.7l-4.2 4.2M12.9 27.1l-4.2 4.2"/>',
    "wave": '<path d="M7 17v6M12.5 11v18M18 6v28M23.5 13v14M29 9v22M34 16v8"/>',
    "server": '<rect x="5" y="6" width="30" height="12" rx="3"/><rect x="5" y="22" width="30" height="12" rx="3"/><circle class="fill" cx="11" cy="12" r="1.5"/><circle class="fill" cx="11" cy="28" r="1.5"/>',
    "db": '<ellipse cx="20" cy="9" rx="13" ry="4.5"/><path d="M7 9v22c0 2.5 5.8 4.5 13 4.5S33 33.500 33 31V9M7 20c0 2.5 5.8 4.500 13 4.500S33 22.500 33 20"/>',
    "shield": '<path d="M20 4l13 5v10c0 8-5.500 13.500-13 17C12.500 32.500 7 27 7 19V9z"/><path d="M14 20l4 4 8-9"/>',
    "lock": '<rect x="8" y="17" width="24" height="17" rx="3"/><path d="M13 17v-4a7 7 0 0 1 14 0v4"/><circle class="fill" cx="20" cy="25.500" r="2"/>',
    "doc": '<path d="M10 4h14l8 8v24H10z"/><path d="M24 4v8h8M15 20h12M15 26h12M15 32h8"/>',
    "list": '<path d="M14 11h22M14 20h22M14 29h22"/><circle class="fill" cx="7" cy="11" r="2"/><circle class="fill" cx="7" cy="20" r="2"/><circle class="fill" cx="7" cy="29" r="2"/>',
    "chart": '<path d="M5 5v30h30"/><path d="M10 27l7-9 6 5 9-14"/>',
    "clock": '<circle cx="20" cy="20" r="15"/><path d="M20 10v10l7 4"/>',
    "user": '<circle cx="20" cy="13" r="7"/><path d="M6 35c0-8 6-13 14-13s14 5 14 13"/>',
    "users": '<circle cx="15" cy="14" r="6"/><path d="M3 34c0-7 5-11 12-11s12 4 12 11"/><circle cx="29" cy="15" r="4.5"/><path d="M29 24c5 0 8 3 8 9"/>',
    "video": '<rect x="4" y="9" width="32" height="22" rx="4"/><path d="M17 15l9 5-9 5z"/>',
    "terminal": '<rect x="4" y="7" width="32" height="26" rx="4"/><path d="M11 16l6 5-6 5M21 27h8"/>',
    "mic": '<rect x="14" y="4" width="12" height="20" rx="6"/><path d="M8 19c0 7 5 12 12 12s12-5 12-12M20 31v6"/>',
    "pulse": '<path d="M3 21h8l4-12 6 24 5-16 3 4h8"/>',
    "film": '<rect x="6" y="5" width="28" height="30" rx="3"/><path d="M6 13h28M6 27h28M13 5v30M27 5v30"/>',
    "scissors": '<circle cx="11" cy="11" r="5"/><circle cx="11" cy="29" r="5"/><path d="M15 14l20 16M15 26l20-16"/>',
    "branch": '<circle cx="10" cy="8" r="4"/><circle cx="10" cy="32" r="4"/><circle cx="30" cy="16" r="4"/><path d="M10 12v16M30 20c0 6-8 6-18 9"/>',
    "check": '<circle cx="20" cy="20" r="15"/><path d="M12 20l6 6 11-12"/>',
    "folder": '<path d="M4 10a3 3 0 0 1 3-3h8l4 5h14a3 3 0 0 1 3 3v15a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3z"/>',
    "cloud": '<path d="M11 31a7 7 0 0 1 0-14 9 9 0 0 1 17-2 7.500 7.500 0 0 1 1 16z"/>',
    "globe": '<circle cx="20" cy="20" r="15"/><path d="M5 20h30M20 5c-6 5-6 25 0 30M20 5c6 5 6 25 0 30"/>',
    "tag": '<path d="M5 5h14l16 16-14 14L5 19z"/><circle class="fill" cx="12" cy="12" r="2"/>',
}


@dataclass(frozen=True)
class Card:
    cx: float
    cy: float
    size: float = CARD

    @property
    def top(self) -> tuple[float, float]:
        return (self.cx, self.cy - self.size / 2)

    @property
    def bottom(self) -> tuple[float, float]:
        return (self.cx, self.cy + self.size / 2)

    @property
    def left(self) -> tuple[float, float]:
        return (self.cx - self.size / 2, self.cy)

    @property
    def right(self) -> tuple[float, float]:
        return (self.cx + self.size / 2, self.cy)

    def below(self, dy: float) -> tuple[float, float]:
        """題名・小見出しの下の点(矢印を、文字に重ねずに下へ出すとき)。"""
        return (self.cx, self.cy + self.size / 2 + dy)


class Diagram:
    def __init__(self, width: int, height: int, title: str, desc: str, *, grid: int = 48) -> None:
        self.w, self.h, self.grid = width, height, grid
        self.title, self.desc = title, desc
        self._groups: list[str] = []
        self._cards: list[str] = []
        self._arrows: list[str] = []
        self._labels: list[str] = []
        self._texts: list[str] = []

    # ---- 部品 ----
    def group(self, x: float, y: float, w: float, h: float, label: str, *, color: str = "blue", dashed: bool = False, mono: bool = True) -> None:
        """角丸の枠。左上に、ラベルの札(背景色で、枠線を切る)。"""
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self._groups.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="var(--f-{color})" stroke="var(--{color})" stroke-width="1.5"{dash}/>'
        )
        lw = 12 + len(label) * (8.2 if mono else 13.5)
        cls = "glabel mono" if mono else "glabel"
        self._groups.append(f'<rect x="{x + 22}" y="{y - 10}" width="{lw}" height="20" fill="var(--bg)"/>')
        self._groups.append(f'<text class="{cls}" x="{x + 28}" y="{y + 4.5}" style="fill:var(--{color})">{escape(label)}</text>')

    def card(self, cx: float, cy: float, glyph: str, title: str, sub: str = "", *, size: float = CARD) -> Card:
        x, y = cx - size / 2, cy - size / 2
        scale = size / 64 * 1.0
        self._cards.append(f'<rect class="card" x="{x}" y="{y}" width="{size}" height="{size}" rx="12"/>')
        gx, gy = cx - 20 * scale * 1.0, cy - 20 * scale * 1.0
        self._cards.append(f'<g class="ic" transform="translate({gx:.1f} {gy:.1f}) scale({scale:.3f})">{GLYPHS[glyph]}</g>')
        ty = cy + size / 2 + 20
        for i, line in enumerate(title.split("\n")):
            self._labels.append(f'<text class="title" x="{cx}" y="{ty + i * 19}">{escape(line)}</text>')
        sy = ty + 17 * (title.count("\n") + 1) + 2
        for i, line in enumerate(sub.split("\n") if sub else []):
            self._labels.append(f'<text class="sub mono" x="{cx}" y="{sy + i * 15}">{escape(line)}</text>')
        return Card(cx, cy, size)

    def arrow(self, pts: list[tuple[float, float]], label: str = "", *, kind: str = "primary", both: bool = False, at: int = 0,
              dx: float = 0, dy: float = -8, lab_color: str | None = None) -> None:
        """折れ線の矢印。label は、at 番目の線分の中央(+dx, dy)に、背景つきで置く。"""
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        self._arrows.append(f'<path class="a {kind}" d="{d}"/>')
        self._arrows.append(self._head(pts[-2], pts[-1], kind))
        if both:
            self._arrows.append(self._head(pts[1], pts[0], kind))
        if label:
            (x1, y1), (x2, y2) = pts[at], pts[at + 1]
            mx, my = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
            width = sum(1.9 if ord(c) < 128 else 3.6 for c in label) * 3.4 + 10
            self._labels.append(f'<rect class="lab" x="{mx - width / 2:.1f}" y="{my - 11:.1f}" width="{width:.1f}" height="16" rx="3"/>')
            color = lab_color or ("blue" if kind == "primary" else "orange" if kind == "accent" else "red" if kind == "danger" else "")
            self._labels.append(f'<text class="alabel mono {color}" x="{mx:.1f}" y="{my + 1:.1f}">{escape(label)}</text>')

    @staticmethod
    def _head(frm: tuple[float, float], to: tuple[float, float], kind: str) -> str:
        import math

        ang = math.atan2(to[1] - frm[1], to[0] - frm[0])
        tip = to
        l, w = 11, 5.5
        p1 = (tip[0] - l * math.cos(ang) + w * math.sin(ang), tip[1] - l * math.sin(ang) - w * math.cos(ang))
        p2 = (tip[0] - l * math.cos(ang) - w * math.sin(ang), tip[1] - l * math.sin(ang) + w * math.cos(ang))
        return f'<path class="h {kind}" d="M{tip[0]:.1f} {tip[1]:.1f} L{p1[0]:.1f} {p1[1]:.1f} L{p2[0]:.1f} {p2[1]:.1f} Z"/>'

    def text(self, x: float, y: float, s: str, *, cls: str = "note", anchor: str = "start") -> None:
        self._texts.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{escape(s)}</text>')

    def badge(self, x: float, y: float, s: str, *, color: str = "teal", w: float | None = None) -> None:
        """カードの角に付ける小さな札(「実装済」など)。"""
        width = w or 12 + len(s) * 11
        self._texts.append(f'<rect x="{x - width / 2:.1f}" y="{y - 9}" width="{width:.1f}" height="18" rx="9" fill="var(--{color})"/>')
        self._texts.append(f'<text class="badge" x="{x}" y="{y + 4}">{escape(s)}</text>')

    # ---- 出力 ----
    def svg(self) -> str:
        g = self.grid
        grid = (
            f'<defs><pattern id="g" width="{g}" height="{g}" patternUnits="userSpaceOnUse"><path d="M{g} 0H0V{g}" fill="none" stroke="var(--grid)" stroke-width="1"/></pattern></defs>'
        )
        body = "".join(self._groups + self._arrows + self._cards + self._labels + self._texts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" role="img" aria-labelledby="t d">'
            f"<title id=\"t\">{escape(self.title)}</title><desc id=\"d\">{escape(self.desc)}</desc>"
            f"<style>{CSS}</style>{grid}"
            f'<rect width="{self.w}" height="{self.h}" fill="var(--bg)"/><rect width="{self.w}" height="{self.h}" fill="url(#g)"/>'
            f"{body}</svg>\n"
        )
