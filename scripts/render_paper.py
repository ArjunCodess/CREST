"""Render every compiled manuscript page for visual quality review."""
import argparse
import json
from pathlib import Path

import pymupdf as fitz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', type=Path, default=Path('paper/working_notes.pdf'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    text = '\n'.join(page.get_text() for page in doc)
    for stale in ['At this draft stage', 'Experiments still required', 'Participant manuscript draft', 'tab:placeholder']:
        if stale in text:
            raise ValueError(f'Stale draft text remains: {stale}')
    if len(doc) < 2:
        raise ValueError('Expected a complete two-column manuscript')
    pages = []
    for index, page in enumerate(doc):
        destination = args.output / f'page_{index + 1}.png'
        page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False).save(destination)
        pages.append(str(destination.resolve()))
    (args.output / 'extracted_text.txt').write_text(text, encoding='utf-8')
    result = {'pages': len(doc), 'rendered_pages': pages, 'stale_draft_checks': 'passed',
              'visual_review': 'pending'}
    (args.output / 'paper_qa.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
