#!/usr/bin/env python3
"""Generate docs/index.md from README.md and ensure local images resolve in MkDocs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
INDEX = DOCS / "index.md"


def _rewrite_paths(text: str) -> str:
    # Convert GitHub raw/blob links that target docs/assets into local docs paths.
    text = re.sub(
        r'https://github\.com/[^"\)\s]+/blob/[^"\)\s]+/docs/assets/([^"\)\s\?]+)\?raw=true',
        r"assets/\1",
        text,
    )
    text = re.sub(
        r'https://raw\.githubusercontent\.com/[^"\)\s]+/docs/assets/([^"\)\s]+)',
        r"assets/\1",
        text,
    )

    # README paths often start with docs/assets; in docs/index they should be assets.
    text = text.replace("(docs/assets/", "(assets/")
    text = text.replace('src="docs/assets/', 'src="assets/')
    return text


def _image_refs(text: str) -> list[str]:
    refs: list[str] = []

    # Markdown images: ![alt](path)
    refs.extend(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text))

    # HTML images: <img src="path" ...>
    refs.extend(re.findall(r'<img[^>]+src="([^"]+)"', text))

    cleaned: list[str] = []
    for ref in refs:
        ref = ref.strip().strip("<>")
        if not ref:
            continue
        if ref.startswith(("http://", "https://", "data:", "#", "/")):
            continue
        cleaned.append(ref)
    return cleaned


def _resolve_source(path_str: str) -> Path | None:
    path_str = path_str.split("?", 1)[0].split("#", 1)[0]
    rel = Path(path_str)
    candidates = [
        ROOT / rel,
        ROOT / "docs" / rel,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _copy_images(refs: list[str]) -> None:
    for ref in refs:
        normalized = ref.split("?", 1)[0].split("#", 1)[0]
        source = _resolve_source(normalized)
        if source is None:
            print(f"warning: image not found for reference: {ref}")
            continue

        destination = DOCS / normalized
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Skip if this is already the same file in docs.
        if source.resolve() == destination.resolve():
            continue

        shutil.copy2(source, destination)


def main() -> None:
    if not README.exists():
        raise FileNotFoundError(f"Missing README: {README}")

    DOCS.mkdir(parents=True, exist_ok=True)

    text = README.read_text(encoding="utf-8")
    rewritten = _rewrite_paths(text)
    INDEX.write_text(rewritten, encoding="utf-8")

    refs = _image_refs(rewritten)
    _copy_images(refs)

    print(f"Generated {INDEX.relative_to(ROOT)} from README.md")


if __name__ == "__main__":
    main()
