"""docs/ と README の Mermaid の図が、実際に描画できるか(構文の誤りが無いか)を、mmdc で確かめる。

使い方::

    python tools/check_mermaid.py            # 全部
    python tools/check_mermaid.py docs/lecture/09-server-and-security.md

mmdc(`npm install -g @mermaid-js/mermaid-cli`)が要る。無ければ、飛ばして終了コード 2。
標準の init の行の検査は tools/check_docs.py。ここは、**描画できるか**だけ(時間がかかるので、テストには入れない)。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import quote_code  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    mmdc = shutil.which("mmdc")
    if not mmdc:
        print("mmdc が見つかりません(npm install -g @mermaid-js/mermaid-cli)", file=sys.stderr)
        return 2
    files = [Path(a) for a in argv] or sorted((ROOT / "docs").rglob("*.md")) + sorted(ROOT.glob("README*.md"))
    failed = total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path in files:
            for block in quote_code.fenced_blocks(path.read_text(encoding="utf-8")):
                if block.info != "mermaid":
                    continue
                total += 1
                src = Path(tmp) / "d.mmd"
                src.write_text("\n".join(block.lines), encoding="utf-8")
                res = subprocess.run([mmdc, "-q", "-i", str(src), "-o", str(Path(tmp) / "d.svg")], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
                if res.returncode != 0:
                    failed += 1
                    print(f"{path.relative_to(ROOT)}:{block.start + 1}: 描画できない\n{res.stderr.strip()[-400:]}", file=sys.stderr)
    print(f"Mermaid {total} 個のうち、描画できない {failed} 個")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
