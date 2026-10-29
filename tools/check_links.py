"""Check the relative links of README.md and docs/*.md.

Every relative link target (markdown links, ``src``/``srcset`` of inline HTML) must exist in the
checkout (tracked by git), and every ``#anchor`` into a markdown file must match one of its headings (GitHub slug
rules). External links are not fetched. Exits 1 and lists the broken links if any are found.

    uv run --no-project python tools/check_links.py [files ...]
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import cache
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = ["README.md", "docs/*.md"]

FENCE = re.compile(r"^(```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")
MD_LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_ATTR = re.compile(r"\b(?:src|href|srcset)=\"([^\"]+)\"")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
EXTERNAL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _slug(heading: str) -> str:
    """GitHub's heading anchor: drop markup and punctuation, lowercase, spaces to hyphens."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links keep their text
    text = re.sub(r"<[^>]+>", "", text).replace("`", "").lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _lines(path: Path, strip_code: bool = True) -> list[tuple[int, str]]:
    """Lines outside fenced code blocks, inline code removed unless ``strip_code`` is False."""
    out, fenced = [], False
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line.lstrip()):
            fenced = not fenced
            continue
        if not fenced:
            out.append((no, INLINE_CODE.sub("", line) if strip_code else line))
    return out


@cache
def tracked() -> frozenset[str] | None:
    """Paths known to git (relative to the root), or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return frozenset(p for p in out.stdout.decode().split("\0") if p)


def exists(dest: Path) -> bool:
    """True if ``dest`` is in the checkout (tracked by git, when git is available)."""
    files = tracked()
    try:
        rel = dest.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    if files is None or rel == ".":
        return dest.exists()
    return rel in files or any(f.startswith(rel + "/") for f in files)


@cache
def anchors(path: Path) -> frozenset[str]:
    seen: dict[str, int] = {}
    result = set()
    for _, line in _lines(path, strip_code=False):
        m = HEADING.match(line)
        if not m:
            continue
        base = _slug(m.group(1))
        n = seen.get(base, 0)
        seen[base] = n + 1
        result.add(base if n == 0 else f"{base}-{n}")
    for m in re.finditer(r"<a\s+(?:name|id)=\"([^\"]+)\"", path.read_text(encoding="utf-8")):
        result.add(m.group(1))
    return frozenset(result)


def targets(line: str) -> list[str]:
    found = [m.group(1) for m in MD_LINK.finditer(line)]
    for m in HTML_ATTR.finditer(line):
        # srcset may hold "a.png 1x, b.png 2x"
        found += [part.strip().split()[0] for part in m.group(1).split(",") if part.strip()]
    return found


def check(path: Path) -> list[str]:
    errors = []
    rel = path.relative_to(ROOT)
    for no, line in _lines(path):
        for target in targets(line):
            if EXTERNAL.match(target) or target.startswith("//"):
                continue
            file_part, _, anchor = target.partition("#")
            dest = (path.parent / unquote(file_part)).resolve() if file_part else path
            if not exists(dest):
                errors.append(f"{rel}:{no}: missing file {target}")
                continue
            if anchor and dest.suffix == ".md" and unquote(anchor) not in anchors(dest):
                errors.append(f"{rel}:{no}: missing anchor {target}")
    return errors


def main(argv: list[str]) -> int:
    patterns = argv or DEFAULT_FILES
    files = sorted({p.resolve() for pat in patterns for p in ROOT.glob(pat) if exists(p.resolve())})
    errors = [e for f in files for e in check(f)]
    for e in errors:
        print(e)
    print(f"checked {len(files)} files, {len(errors)} broken links")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
