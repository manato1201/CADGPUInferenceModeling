"""講義資料の中のコード引用を、実際のソースから作る・更新する。

(DocsPipeline の tools/quote_code.py をもとにした複製。このリポジトリの docs/ の検査に使う。ルートはこのリポジトリ。
実行できる例は、リポジトリのルートで、`-I` なしで動かす(プロジェクトのモジュールを import できるように))

講義にコードを貼ると、コードが変わったときに、説明が古くなる。引用を手で写さず、このツールが
ソースから作る。``tests/test_lectures.py`` が、引用が今のソースと一致しているかを確かめる。

使い方::

    python tools/quote_code.py docs/lecture            # 引用を作る・更新する
    python tools/quote_code.py docs/lecture --check    # ずれていないか確かめる (書き換えない)

Markdown の中の書き方は 3 種類。

``@@引用 パス::名前@@`` (1 行)
    その場所に、関数・クラス・メソッドの全体を、コードブロックとして展開する。
    ``パス`` はリポジトリのルートからの相対。``名前`` は ``関数`` か ``クラス.メソッド``。

``# 引用: パス::名前`` で始まるコードブロック
    展開済みの引用。残りの中身は、ソースと **完全に一致** する。更新のとき、ソースから書き直す。

``# 抜粋: パス::名前`` で始まるコードブロック
    一部の抜粋 (手で書く)。各行が、そのソースの中に、**この順で** 現れること。省略は ``...`` の行で書く。

``# 抜粋ファイル: パス`` で始まるコードブロック (JavaScript は ``//``、CSS は ``/* ... */`` の形でもよい)
    Python 以外のファイルの抜粋。各行が、そのファイルの中に、**この順で** 現れること。

``# 実行できる`` で始まるコードブロックは、テストが実際に実行する。直後のコードブロックが ``# 出力`` で始まっていれば、
標準出力と一致することも確かめる。``--outputs`` を付けると、実行して、``# 出力`` のブロックを実際の出力で書き直す
(講義を書くとき、出力を手で写さないため)。
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def example_env() -> dict[str, str]:
    """実行できる例を動かすときの環境変数。``tools/docs_pythonpath.txt`` に 1 行 1 パス(ルートからの相対)で書くと、PYTHONPATH に足す
    (ソースが ``src/`` や ``backend/`` の下にあるプロジェクト向け)。無ければ、そのまま。"""
    import os

    env = dict(os.environ)
    cfg = ROOT / "tools" / "docs_pythonpath.txt"
    if cfg.exists():
        extra = [str(ROOT / line.strip()) for line in cfg.read_text(encoding="utf-8").splitlines() if line.strip()]
        env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


QUOTE_MARK = "# 引用:"
EXCERPT_MARK = "# 抜粋:"
RUN_MARK = "# 実行できる"
OUTPUT_MARK = "# 出力"

FILE_EXCERPT = re.compile(r"^(?:#|//|/\*) 抜粋ファイル: (?P<path>\S+?)(?:\s*\*/)?\s*$")
_PLACEHOLDER = re.compile(r"^@@引用 (?P<ref>\S+::\S+)@@\s*$")
_FENCE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")


@dataclass(frozen=True)
class Block:
    """Markdown のコードブロック。"""

    start: int  # 開きの囲みの行 (0 始まり)
    end: int  # 閉じの囲みの行
    info: str  # 言語名など
    lines: list[str]  # 中身の行


def fenced_blocks(text: str) -> list[Block]:
    """Markdown から、コードブロックを探す。長い囲み (````) の中の短い囲みは、中身として扱う。"""
    lines = text.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        match = _FENCE.match(lines[i])
        if not match:
            i += 1
            continue
        fence = match.group("fence")
        end = i + 1
        while end < len(lines):
            stripped = lines[end].strip()
            if stripped.startswith(fence[0] * len(fence)) and not stripped.strip(fence[0]):
                break
            end += 1
        blocks.append(Block(i, end, match.group("info").strip(), lines[i + 1 : end]))
        i = end + 1
    return blocks


def source_of(ref: str, root: Path = ROOT) -> str:
    """``パス::名前`` が指す、関数・クラス・メソッドのソースを、インデントを外して返す。

    Raises:
        KeyError: 名前が見つからないとき。
        FileNotFoundError: ファイルが無いとき。
    """
    path_text, qualname = ref.split("::", 1)
    source = (root / path_text).read_text(encoding="utf-8")
    nodes: list[ast.AST] = list(ast.parse(source).body)
    found: ast.AST | None = None
    for part in qualname.split("."):
        found = next(
            (
                n
                for n in nodes
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and n.name == part
            ),
            None,
        )
        if found is None:
            raise KeyError(f"{ref}: {part} が見つからない")
        nodes = list(getattr(found, "body", []))
    assert found is not None
    # 行単位で切り出す。ast.get_source_segment は、デコレータと、最後の行の末尾のコメントを落とす
    first = min([found.lineno, *[d.lineno for d in getattr(found, "decorator_list", [])]])
    segment = "\n".join(source.split("\n")[first - 1 : found.end_lineno])
    return textwrap.dedent(segment).rstrip("\n")


def render_quote(ref: str, root: Path = ROOT) -> list[str]:
    """引用のコードブロックの行を作る。"""
    return ["```python", f"{QUOTE_MARK} {ref}", *source_of(ref, root).split("\n"), "```"]


def refresh(text: str, root: Path = ROOT) -> str:
    """プレースホルダーを展開し、既存の引用をソースから書き直す。"""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        match = _PLACEHOLDER.match(line)
        out += render_quote(match.group("ref"), root) if match else [line]
    expanded = "\n".join(out)
    for block in reversed(fenced_blocks(expanded)):
        if not block.lines or not block.lines[0].startswith(QUOTE_MARK):
            continue
        ref = block.lines[0][len(QUOTE_MARK) :].strip()
        new = render_quote(ref, root)
        all_lines = expanded.split("\n")
        all_lines[block.start : block.end + 1] = new
        expanded = "\n".join(all_lines)
    return expanded


def fill_outputs(text: str) -> str:
    """``# 実行できる`` のブロックを実行し、直後の ``# 出力`` のブロックを、実際の標準出力に書き直す。

    Raises:
        RuntimeError: コードが失敗したとき (どの行のブロックかを含む)。
    """
    lines = text.split("\n")
    blocks = fenced_blocks(text)
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if not block.lines or block.lines[0].strip() != RUN_MARK:
            continue
        nxt = blocks[index + 1] if index + 1 < len(blocks) else None
        if not (nxt and nxt.lines and nxt.lines[0].strip() == OUTPUT_MARK):
            continue
        code = "\n".join(block.lines[1:])
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=300, check=False,
                env=example_env(),
            )
        if result.returncode != 0:
            raise RuntimeError(f"{block.start + 1} 行目のコードが失敗しました:\n{result.stderr}")
        output = result.stdout.rstrip("\n").split("\n") if result.stdout.strip() else ["(出力なし)"]
        lines[nxt.start + 1 : nxt.end] = [OUTPUT_MARK, *output]
    return "\n".join(lines)


def problems(text: str, root: Path = ROOT) -> list[str]:
    """引用・抜粋・プレースホルダーの、ずれを返す (空なら問題なし)。"""
    found = []
    for number, line in enumerate(text.split("\n"), 1):
        if _PLACEHOLDER.match(line):
            found.append(f"{number}: 展開されていない引用のプレースホルダー")
    for block in fenced_blocks(text):
        if not block.lines:
            continue
        head = block.lines[0]
        where = block.start + 1
        file_match = FILE_EXCERPT.match(head)
        if file_match:
            try:
                file_lines = [
                    ln.strip() for ln in (root / file_match.group("path")).read_text(encoding="utf-8").split("\n")
                ]
            except FileNotFoundError as exc:
                found.append(f"{where}: {exc}")
                continue
            position = 0
            for ln in block.lines[1:]:
                text_line = ln.strip()
                if not text_line or text_line in {"...", "// ..."} or text_line.startswith("# ..."):
                    continue
                try:
                    position = file_lines.index(text_line, position) + 1
                except ValueError:
                    name = file_match.group("path")
                    found.append(f"{where}: {name} の抜粋の行が、ファイルに無い (または順序が違う): {text_line!r}")
                    break
            continue
        if head.startswith(QUOTE_MARK):
            ref = head[len(QUOTE_MARK) :].strip()
            try:
                expected = source_of(ref, root).split("\n")
            except (KeyError, FileNotFoundError) as exc:
                found.append(f"{where}: {exc}")
                continue
            if block.lines[1:] != expected:
                found.append(f"{where}: {ref} の引用が、今のソースと違う (python tools/quote_code.py で更新する)")
        elif head.startswith(EXCERPT_MARK):
            ref = head[len(EXCERPT_MARK) :].strip()
            try:
                source_lines = [ln.strip() for ln in source_of(ref, root).split("\n")]
            except (KeyError, FileNotFoundError) as exc:
                found.append(f"{where}: {exc}")
                continue
            position = 0
            for ln in block.lines[1:]:
                text_line = ln.strip()
                if not text_line or text_line == "..." or text_line.startswith("# ..."):
                    continue
                try:
                    position = source_lines.index(text_line, position) + 1
                except ValueError:
                    found.append(f"{where}: {ref} の抜粋の行が、ソースに無い (または順序が違う): {text_line!r}")
                    break
    return found


def main(argv: list[str] | None = None) -> int:
    """引用を更新する。``--check`` では、書き換えずに、ずれを報告する。"""
    args = sys.argv[1:] if argv is None else argv
    check = "--check" in args
    outputs = "--outputs" in args
    paths = [Path(a) for a in args if not a.startswith("--")]
    status = 0
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.md"))
        for path in files:
            text = path.read_text(encoding="utf-8")
            if check:
                for item in problems(text):
                    print(f"{path}:{item}", file=sys.stderr)
                    status = 1
                continue
            new = refresh(text)
            if outputs:
                new = fill_outputs(new)
            if new != text:
                path.write_text(new, encoding="utf-8")
                print(f"更新: {path}")
            for item in problems(new):
                print(f"{path}:{item}", file=sys.stderr)
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
