"""
fbx2obj.py
==========
FBXファイルをOBJに変換するユーティリティスクリプト。
Blenderをバックグラウンドで呼び出して変換する。

使い方:
    python fbx2obj.py model.fbx              # model.obj を生成
    python fbx2obj.py model.fbx -o out.obj   # 出力先を指定
    python fbx2obj.py *.fbx                  # 複数ファイルを一括変換
"""

import argparse
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


# Blenderの一般的なインストールパス（Windows）
BLENDER_CANDIDATES = [
    "blender",  # PATH が通っている場合
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
    r"C:\Program Files (x86)\Blender Foundation\Blender 4.5\blender.exe",
]


def find_blender() -> str | None:
    """Blenderの実行ファイルを探す。"""
    for candidate in BLENDER_CANDIDATES:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                ver = result.stdout.decode(errors="replace").split("\n")[0]
                print(f"Blender 発見: {candidate} ({ver.strip()})")
                return candidate
        except Exception:
            continue
    return None


def convert_fbx_to_obj(
    fbx_path: Path,
    obj_path: Path,
    blender_exe: str,
) -> bool:
    """
    Blender CLI でFBXをOBJに変換する。

    Returns True if successful.
    """
    # Blender Python スクリプト
    script = textwrap.dedent(f"""
        import bpy, sys

        # シーンをクリア
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # FBX をインポート
        try:
            bpy.ops.import_scene.fbx(filepath=r"{fbx_path}")
        except Exception as e:
            print(f"FBX import error: {{e}}", file=sys.stderr)
            sys.exit(1)

        # 全オブジェクトを選択
        bpy.ops.object.select_all(action='SELECT')

        # OBJ としてエクスポート
        try:
            # Blender 4.x
            bpy.ops.wm.obj_export(
                filepath=r"{obj_path}",
                export_selected_objects=False,
                export_materials=False,
            )
        except AttributeError:
            try:
                # Blender 3.x
                bpy.ops.export_scene.obj(
                    filepath=r"{obj_path}",
                    use_selection=False,
                    use_materials=False,
                )
            except Exception as e:
                print(f"OBJ export error: {{e}}", file=sys.stderr)
                sys.exit(1)

        print(f"変換完了: {obj_path}")
    """)

    # 一時スクリプトファイルに書き出し
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = Path(f.name)

    try:
        result = subprocess.run(
            [blender_exe, "--background", "--python", str(script_path)],
            capture_output=True,
            timeout=120,
        )
        script_path.unlink(missing_ok=True)

        if result.returncode != 0 or not obj_path.exists():
            stderr = result.stderr.decode(errors="replace")
            stdout = result.stdout.decode(errors="replace")
            print(f"  [ERROR] Blender 変換失敗")
            # エラーの関連行だけ表示
            for line in (stderr + stdout).splitlines():
                if any(k in line.lower() for k in ["error", "exception", "failed", "import"]):
                    print(f"    {line}")
            return False

        return True

    except subprocess.TimeoutExpired:
        script_path.unlink(missing_ok=True)
        print("  [ERROR] タイムアウト（120秒）")
        return False
    except Exception as e:
        script_path.unlink(missing_ok=True)
        print(f"  [ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="FBXファイルをOBJに変換する（Blender経由）"
    )
    parser.add_argument("inputs", nargs="+", help="入力FBXファイル（複数可）")
    parser.add_argument("-o", "--output", help="出力OBJパス（1ファイル時のみ有効）")
    parser.add_argument(
        "--blender", help="Blender実行ファイルのパス（省略時は自動検索）"
    )
    args = parser.parse_args()

    # Blenderを探す
    blender_exe = args.blender or find_blender()
    if blender_exe is None:
        print("[ERROR] Blenderが見つかりません。")
        print("以下のいずれかで対応してください:")
        print("  1. Blenderをインストールしてパスを通す")
        print("  2. --blender オプションでパスを直接指定")
        print('     例: python fbx2obj.py model.fbx --blender "C:\\path\\to\\blender.exe"')
        sys.exit(1)

    # 入力ファイルを解決
    fbx_files = []
    for pattern in args.inputs:
        from glob import glob
        matched = glob(pattern)
        if matched:
            fbx_files.extend([Path(p) for p in matched])
        else:
            p = Path(pattern)
            if p.exists():
                fbx_files.append(p)
            else:
                print(f"[WARN] ファイルが見つかりません: {pattern}")

    if not fbx_files:
        print("[ERROR] 変換対象のFBXファイルがありません。")
        sys.exit(1)

    # 変換実行
    success_count = 0
    for fbx_path in fbx_files:
        if len(fbx_files) == 1 and args.output:
            obj_path = Path(args.output)
        else:
            obj_path = fbx_path.with_suffix(".obj")

        print(f"\n変換中: {fbx_path} → {obj_path}")
        if convert_fbx_to_obj(fbx_path, obj_path, blender_exe):
            size_kb = obj_path.stat().st_size / 1024
            print(f"  ✓ 完了 ({size_kb:.0f} KB)")
            success_count += 1
        else:
            print(f"  ✗ 失敗")

    print(f"\n結果: {success_count}/{len(fbx_files)} ファイル変換成功")

    if success_count > 0:
        print("\n変換後にDXFに変換するには:")
        for fbx_path in fbx_files:
            obj_path = fbx_path.with_suffix(".obj")
            if obj_path.exists():
                print(f"  python cli.py export {obj_path.name}")


if __name__ == "__main__":
    main()
