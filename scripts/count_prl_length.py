#!/usr/bin/env python3
"""Estimate the APS word-equivalent length of a RevTeX Letter.

Physical Review Letters caps a Letter at 3750 word equivalents. The estimate
follows the APS length guide:

* body text and figure captions are counted as words; the title, byline,
  abstract, acknowledgments, and references are excluded;
* each line of a displayed equation counts as 16 word equivalents
  (32 for a full-width ``align``/``equation`` inside a starred float);
* a figure counts as ``150 / aspect + 20`` words in one column and
  ``300 / (0.5 * aspect) + 40`` words across both, with
  ``aspect = width / height`` taken from the included graphics file;
* the reference list is excluded.

The result is an estimate. It is close enough to decide whether a manuscript
needs trimming before submission, and it reports its own breakdown so that a
disputed term can be checked by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PRL_LIMIT = 3750
def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def count_words(text: str) -> int:
    text = re.sub(r"\\begin\{(equation|align|eqnarray|gather)\*?\}.*?\\end\{\1\*?\}", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\(label|ref|eqref|cite|includegraphics)\s*(\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[\\{}$&_^~\[\]]", " ", text)
    return len([word for word in text.split() if any(character.isalnum() for character in word)])


def graphic_aspect(root: Path, name: str) -> float | None:
    for suffix in ("", ".pdf", ".png", ".eps", ".jpg"):
        path = root / (name + suffix)
        if not path.is_file():
            continue
        try:
            info = subprocess.run(
                ["pdfinfo", str(path)], capture_output=True, text=True, check=True
            ).stdout
            match = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info)
            if match:
                return float(match.group(1)) / float(match.group(2))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as image:
                return image.width / image.height
        except Exception:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tex", type=Path)
    parser.add_argument("--limit", type=int, default=PRL_LIMIT)
    args = parser.parse_args()

    root = args.tex.resolve().parent
    source = strip_comments(args.tex.read_text(encoding="utf-8"))
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S)
    abstract = abstract_match.group(1) if abstract_match else ""
    body = source.split(r"\maketitle", 1)[-1].split(r"\begin{thebibliography}", 1)[0]

    figures = []
    figure_words = 0
    for match in re.finditer(r"\\begin\{figure(\*?)\}(.*?)\\end\{figure\1\}", body, re.S):
        wide = match.group(1) == "*"
        block = match.group(2)
        caption = re.search(r"\\caption\{(.*)\}?", block, re.S)
        caption_words = count_words(caption.group(1)) if caption else 0
        graphic = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block)
        aspect = graphic_aspect(root, graphic.group(1)) if graphic else None
        if aspect is None:
            aspect = 3.0
            resolved = False
        else:
            resolved = True
        area = (300.0 / (0.5 * aspect) + 40.0) if wide else (150.0 / aspect + 20.0)
        figure_words += area
        figures.append(
            {
                "graphic": graphic.group(1) if graphic else None,
                "full_width": wide,
                "aspect_ratio": round(aspect, 3),
                "aspect_from_file": resolved,
                "figure_word_equivalents": round(area),
                "caption_words": caption_words,
            }
        )

    text_without_figures = re.sub(r"\\begin\{figure(\*?)\}.*?\\end\{figure\1\}", " ", body, flags=re.S)
    text_without_figures = re.sub(
        r"\\begin\{acknowledgments\}.*?\\end\{acknowledgments\}",
        " ",
        text_without_figures,
        flags=re.S,
    )
    equation_lines = 0
    for match in re.finditer(r"\\begin\{(equation|align|gather|eqnarray)(\*?)\}(.*?)\\end\{\1\2\}", body, re.S):
        content = match.group(3)
        equation_lines += 1 + content.count(r"\\")
    equation_words = 16 * equation_lines
    caption_words = sum(item["caption_words"] for item in figures)
    body_words = count_words(text_without_figures)
    abstract_words = count_words(abstract)

    total = round(body_words + caption_words + equation_words + figure_words)
    report = {
        "source": str(args.tex),
        "limit": args.limit,
        "estimated_core_word_equivalents": total,
        "estimated_word_equivalents": total,
        "over_limit_by": max(0, total - args.limit),
        "breakdown": {
            "excluded_abstract_words": abstract_words,
            "body_words": body_words,
            "caption_words": caption_words,
            "equation_lines": equation_lines,
            "equation_word_equivalents": equation_words,
            "figure_word_equivalents": round(figure_words),
        },
        "figures": figures,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
