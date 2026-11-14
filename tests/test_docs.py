"""文書(docs/・README)の検査。説明が、コードの変更で嘘にならないようにする。

- ``# 引用:`` のブロックは、今のソースと完全に一致する(``python tools/quote_code.py docs`` で更新)
- ``# 抜粋:`` / ``# 抜粋ファイル:`` のブロックの各行は、そのソースに、この順で現れる
- ``# 実行できる`` のブロックは、実際に実行でき、直後の ``# 出力`` のブロックと標準出力が一致する
- 講義(docs/lecture/)は、標準の節(ゴール・確認問題・解答・まとめ)を持つ
- 文書の中の相対リンクと、画像(SVG)の参照先が、実在する
- Mermaid の図は、標準の ``%%{init}%%`` の行で始まる(GitHub と HTML で、色とフォントが揃う)
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import quote_code  # noqa: E402

DOCS = ROOT / "docs"
LECTURES = sorted((DOCS / "lecture").glob("*.md"))
LEARN = sorted((DOCS / "learn").glob("*.md"))
READMES = [ROOT / "README.md"]
ALL_MD = sorted(DOCS.rglob("*.md")) + [p for p in READMES if p.exists()]
REQUIRED_SECTIONS = ("## この回のゴール", "## 確認問題", "## 解答", "## まとめ")


def test_there_are_lectures_and_learning_pages():
    assert len(LECTURES) >= 5, f"講義が足りない ({len(LECTURES)})"
    assert len(LEARN) >= 1, f"学習資料が足りない ({len(LEARN)})"


@pytest.mark.parametrize("path", ALL_MD, ids=lambda p: str(p.relative_to(ROOT)))
def test_quotes_and_excerpts_match_the_current_source(path):
    assert quote_code.problems(path.read_text(encoding="utf-8"), ROOT) == []


@pytest.mark.parametrize("path", [p for p in LECTURES if p.name != "index.md"], ids=lambda p: p.name)
def test_every_lecture_has_the_standard_sections(path):
    text = path.read_text(encoding="utf-8")
    for heading in REQUIRED_SECTIONS:
        assert heading in text, f"{path.name}: 「{heading}」が無い"


def _runnable(path: Path):
    blocks = quote_code.fenced_blocks(path.read_text(encoding="utf-8"))
    for i, block in enumerate(blocks):
        if block.lines and block.lines[0].strip() == quote_code.RUN_MARK:
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            expected = None
            if nxt and nxt.lines and nxt.lines[0].strip() == quote_code.OUTPUT_MARK:
                expected = "\n".join(nxt.lines[1:]).strip()
            yield block.start + 1, "\n".join(block.lines[1:]), expected


@pytest.mark.parametrize("path", LECTURES + LEARN, ids=lambda p: p.name if p.parent.name == "lecture" else f"learn/{p.name}")
def test_runnable_examples_run_and_print_what_the_page_says(path, tmp_path):
    # 2 回走らせる(集合の順序など、実行ごとに変わる出力に頼る例を見つける)
    for line, code, expected in _runnable(path):
        for _ in range(2):
            result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, encoding="utf-8",
                                    cwd=ROOT, timeout=300, check=False, env=quote_code.example_env())
            assert result.returncode == 0, f"{path.name}:{line}: 実行に失敗\n{result.stderr}"
            if expected is not None:
                assert result.stdout.strip() == expected, f"{path.name}:{line}: 出力が記述と違う\n{result.stdout}"


def test_runnable_examples_exist_in_total():
    total = sum(len(list(_runnable(p))) for p in LECTURES + LEARN)
    assert total >= 5, f"実行できる例が少ない ({total})"


_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\((?P<img>[^)\s]+)\)")


@pytest.mark.parametrize("path", ALL_MD, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_and_images_exist(path):
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)  # コードブロックの中は見ない
    missing = []
    for m in _LINK.finditer(text):
        url = m.group("url") or m.group("img")
        if re.match(r"^(https?:|mailto:|#)", url):
            continue
        target = (path.parent / url.split("#")[0]).resolve()
        if not target.exists():
            missing.append(url)
    assert not missing, f"{path.relative_to(ROOT)}: リンク先がない: {missing}"


MERMAID_INIT = '%%{init: {"theme":"base","fontFamily":"Noto Sans JP'


@pytest.mark.parametrize("path", ALL_MD, ids=lambda p: str(p.relative_to(ROOT)))
def test_mermaid_blocks_start_with_the_standard_init_line(path):
    for block in quote_code.fenced_blocks(path.read_text(encoding="utf-8")):
        if block.info == "mermaid":
            assert block.lines and block.lines[0].startswith(MERMAID_INIT), (
                f"{path.relative_to(ROOT)}:{block.start + 1}: Mermaid が標準の init の行で始まらない"
                "(docspipeline check-mermaid --fix で入れる)"
            )
