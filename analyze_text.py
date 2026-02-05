#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


DEFAULT_TEXT_SIMILARITY_THRESHOLD = 0.7


class TextAnalyzer:
    """Analyzes text similarity between reports using TF-IDF + cosine similarity."""

    def _load_report_text(self, report_dirpath: Path) -> str:
        combined = ""
        for f in sorted(report_dirpath.iterdir(), key=lambda p: (not p.stem.isdigit(), p.name)):
            if f.suffix == ".txt" and f.name != "metadata.txt" and f.stem.isdigit():
                combined += (f.read_text(encoding="utf-8") or "") + "\n"
        return combined

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # TF-IDF vectorization + cosine similarity. Returns n x n matrix.
    def _compute_similarity_matrix(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",  # Unicode-aware word tokens (Polish etc.)
            ngram_range=(1, 2),  # unigrams + bigrams for better plagiarism detection
        )
        tfidf = vectorizer.fit_transform(texts)
        matrix = cosine_similarity(tfidf, tfidf)
        return np.clip(matrix, 0, 1)

    # Parse label, submid from report_dirpath/metadata.txt
    def _parse_report_label(self, report_dirpath: Path, groups: bool) -> tuple[str, str | None]:
        meta_file = report_dirpath / "metadata.txt"
        authors = ""
        group = None
        submid = None
        if meta_file.exists():
            for line in meta_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("authors:"):
                    authors = line.split(":", 1)[1].strip()
                elif line.startswith("group:"):
                    val = line.split(":", 1)[1].strip()
                    group = val if val else None
                elif line.startswith("submid:"):
                    val = line.split(":", 1)[1].strip()
                    submid = val if val else None
        label = group if (groups and group) else authors
        return (label or report_dirpath.name, submid)

    def compare_texts(self, reports_dirpath: Path, threshold: float | None = None) -> None:
        # Read groups flag from reports_dirpath/metadata.txt
        groups_meta = reports_dirpath / "metadata.txt"
        groups = False
        if groups_meta.exists():
            for line in groups_meta.read_text(encoding="utf-8").splitlines():
                if line.startswith("groups:"):
                    groups = "true" in line.lower().split(":", 1)[1].strip()
                    break

        report_dirs = [d for d in reports_dirpath.iterdir() if d.is_dir()]
        if not report_dirs:
            return

        # Load report dirs sorted by submid
        def sort_key(rd: Path) -> tuple[int, str]:
            _, submid = self._parse_report_label(rd, groups)
            return (int(submid) if submid else 0, rd.name)

        report_dirs = sorted(report_dirs, key=sort_key)

        labels = []
        submids = []
        texts = []
        for rd in report_dirs:
            label, submid = self._parse_report_label(rd, groups)
            labels.append(label)
            submids.append(submid or "0")
            texts.append(self._normalize_text(self._load_report_text(rd)))

        n = len(report_dirs)
        sys.stdout.write("\rComparing (TF-IDF + cosine similarity)...")
        sys.stdout.flush()
        matrix = self._compute_similarity_matrix(texts)
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

        th = threshold if threshold is not None else DEFAULT_TEXT_SIMILARITY_THRESHOLD

        # Find plagiarism pairs (submid order: earlier = source)
        circle_cells = set()
        plagiarism_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] >= th:
                    submid_i = int(submids[i]) if submids[i] else 0
                    submid_j = int(submids[j]) if submids[j] else 0
                    if submid_i < submid_j:
                        plagiarism_pairs.append((labels[j], labels[i], matrix[i, j]))
                        circle_cells.add((j, i))
                    elif submid_j < submid_i:
                        plagiarism_pairs.append((labels[i], labels[j], matrix[i, j]))
                        circle_cells.add((i, j))
                    else:
                        plagiarism_pairs.append((labels[i], labels[j], matrix[i, j]))

        for a, b, sim in sorted(plagiarism_pairs, key=lambda x: -x[2]):
            print(f"{a} report is similar to {b} report")

        if plagiarism_pairs:
            print()

        # Similarity matrix visualization
        cmap = LinearSegmentedColormap.from_list("white_red", [(1, 1, 1), (1, 0.5, 0.5)])
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        short_labels = [l[:20] + "..." if len(l) > 20 else l for l in labels]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels, rotation=45, ha="right")
        ax.set_yticklabels(short_labels)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", color="black", fontsize=9)
                if (i, j) in circle_cells:
                    circle = Circle((j, i), 0.4, fill=False, edgecolor="green", linewidth=2)
                    ax.add_patch(circle)

        plt.colorbar(im, ax=ax, label="Similarity")
        ax.set_title(f"Text similarity matrix (threshold {th:.0%})")
        plt.tight_layout()
        plt.show()


def _parse_args() -> tuple[Path, float | None]:
    """
    Parse CLI: [threshold] dirpath.
    Returns (reports_dirpath, threshold_arg). Threshold accepts 0.6 or 60 (percent).
    """
    parser = argparse.ArgumentParser(description="Analyze text similarity of reports")
    parser.add_argument("args", nargs="*", metavar="[threshold] dirpath",
                        help="Optional threshold (60 or 0.6), then dirpath")
    parsed = parser.parse_args()

    if len(parsed.args) == 0:
        parser.error("dirpath required")

    if len(parsed.args) == 1:
        dirpath = Path(parsed.args[0])
        threshold_arg = None
    else:
        # Parse threshold: "60", "60%", "0.6" -> float in (0, 1]
        raw = parsed.args[0].strip().rstrip("%")
        try:
            v = float(raw)
            threshold_arg = v / 100.0 if v > 1 else v
        except ValueError:
            print(f"Error: '{parsed.args[0]}' is not a valid threshold", file=sys.stderr)
            sys.exit(1)
        if not 0 < threshold_arg <= 1:
            print(f"Error: threshold must be in (0, 1] or (0, 100]", file=sys.stderr)
            sys.exit(1)
        dirpath = Path(parsed.args[1])

    reports_dirpath = dirpath / "text"
    if not reports_dirpath.is_dir():
        print(f"Error: '{reports_dirpath}' not found. Run extract_content.py first.", file=sys.stderr)
        sys.exit(1)

    return reports_dirpath, threshold_arg


def main():
    reports_dirpath, threshold_arg = _parse_args()
    th = threshold_arg if threshold_arg is not None else DEFAULT_TEXT_SIMILARITY_THRESHOLD
    print(f"Comparing reports with threshold {th:.0%}...")
    TextAnalyzer().compare_texts(reports_dirpath, threshold=threshold_arg)


if __name__ == "__main__":
    main()
