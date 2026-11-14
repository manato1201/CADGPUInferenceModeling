"""docs/ と README の検査(pytest 不要・標準ライブラリだけ)。説明が、コードの変更で嘘にならないようにする。

使い方::

    python tools/check_docs.py              # 全部を検査
    python tools/check_docs.py --update     # 引用の更新・実行例の出力の作り直しをしてから検査

検査する項目
  ・``# 引用:`` のブロックは、今のソースと完全に一致する(Python のみ。``@@引用 パス::名前@@`` で作る)
  ・``# 抜粋:`` / ``# 抜粋ファイル:`` のブロックの各行は、そのソースに、この順で現れる(どの言語でもよい)
  ・``# 実行できる`` のブロックは、実際に 2 回実行でき、直後の ``# 出力`` のブロックと標準出力が一致する(Python)
  ・講義(docs/lecture/)は、標準の節(ゴール・確認問題・解答・まとめ)を持つ
  ・文書の中の相対リンクと、画像の参照先が、実在する
  ・Mermaid は、標準の ``%%{init}%%`` の行で始まる(``docspipeline check-mermaid --fix`` で入れる)
終了コードは、問題があれば 1。tools/quote_code.py と同じフォルダに置く。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import quote_code  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REQUIRED_SECTIONS = ("## この回のゴール", "## 確認問題", "## 解答", "## まとめ")
MERMAID_INIT = '%%{init: {"theme":"base","fontFamily":"Noto Sans JP'
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\((?P<img>[^)\s]+)\)")


def all_markdown() -> list[Path]:
    readmes = sorted(ROOT.glob("README*.md"))
    return sorted(DOCS.rglob("*.md")) + readmes if DOCS.exists() else readmes


def check_quotes(path: Path) -> list[str]:
    return [f"{path.relative_to(ROOT)}:{p}" for p in quote_code.problems(path.read_text(encoding="utf-8"), ROOT)]


def check_sections(path: Path) -> list[str]:
    if path.parent.name != "lecture" or path.name == "index.md":
        return []
    text = path.read_text(encoding="utf-8")
    return [f"{path.relative_to(ROOT)}: 「{h}」が無い" for h in REQUIRED_SECTIONS if h not in text]


def check_links(path: Path) -> list[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    out = []
    for m in LINK.finditer(text):
        url = m.group("url") or m.group("img")
        if re.match(r"^(https?:|mailto:|#)", url):
            continue
        if not (path.parent / url.split("#")[0]).resolve().exists():
            out.append(f"{path.relative_to(ROOT)}: リンク先がない: {url}")
    return out


def check_mermaid(path: Path) -> list[str]:
    out = []
    for b in quote_code.fenced_blocks(path.read_text(encoding="utf-8")):
        if b.info == "mermaid" and not (b.lines and b.lines[0].startswith(MERMAID_INIT)):
            out.append(f"{path.relative_to(ROOT)}:{b.start + 1}: Mermaid が標準の init の行で始まらない(docspipeline check-mermaid --fix)")
    return out


def runnable(path: Path):
    blocks = quote_code.fenced_blocks(path.read_text(encoding="utf-8"))
    for i, b in enumerate(blocks):
        if b.lines and b.lines[0].strip() == quote_code.RUN_MARK:
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            expected = "\n".join(nxt.lines[1:]).strip() if nxt and nxt.lines and nxt.lines[0].strip() == quote_code.OUTPUT_MARK else None
            yield b.start + 1, "\n".join(b.lines[1:]), expected


def check_examples(path: Path) -> tuple[list[str], int]:
    out, n = [], 0
    for line, code, expected in runnable(path):
        n += 1
        for _ in range(2):  # 2 回走らせる(乱数・順序に頼る出力を見つける)
            res = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=300, check=False, env=quote_code.example_env())
            if res.returncode != 0:
                out.append(f"{path.relative_to(ROOT)}:{line}: 実行に失敗\n{res.stderr.strip()[-300:]}")
                break
            if expected is not None and res.stdout.strip() != expected:
                out.append(f"{path.relative_to(ROOT)}:{line}: 出力が記述と違う(実行ごとに変わる出力かもしれない)")
                break
    return out, n


def main(argv: list[str]) -> int:
    files = all_markdown()
    if "--update" in argv:
        quote_code.main([str(DOCS), "--outputs"])
    problems: list[str] = []
    examples = 0
    for p in files:
        problems += check_quotes(p) + check_sections(p) + check_links(p) + check_mermaid(p)
        errs, n = check_examples(p)
        problems += errs
        examples += n
    for line in problems:
        print(line, file=sys.stderr)
    lectures = len([p for p in (DOCS / "lecture").glob("*.md")]) if (DOCS / "lecture").exists() else 0
    print(f"文書 {len(files)} 本(講義 {lectures}・実行できる例 {examples} 個)を検査: 問題 {len(problems)} 件")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
