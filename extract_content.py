#!/usr/bin/env python3
"""
Extract content from PDF reports (text + images) and detect file theft (hash).
Usage: python extract_content.py <dirpath>
"""

import argparse
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from content_extractor import ContentExtractor, FileAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Extract text and images from PDF reports")
    parser.add_argument("dirpath", type=Path, help="Directory with PDF files (e.g. ni_writer)")
    args = parser.parse_args()
    dirpath = args.dirpath

    if not dirpath.is_dir():
        print(f"Error: '{dirpath}' is not a directory", file=sys.stderr)
        sys.exit(1)

    text_dir = dirpath / "text"
    if text_dir.exists():
        shutil.rmtree(text_dir)

    extractor = ContentExtractor()
    groups = extractor.detect_groups(dirpath)
    print(f"Groups {'detected' if groups else 'not detected'}")

    report_list = extractor.get_report_list(dirpath, groups)
    reports_dirpath = extractor.extract_text(report_list)
    extractor.extract_images(reports_dirpath, dirpath)
    extractor.refuse_non_reports(reports_dirpath)

    file_analyzer = FileAnalyzer()
    file_analyzer.detect_duplicates(dirpath, reports_dirpath)


if __name__ == "__main__":
    main()
