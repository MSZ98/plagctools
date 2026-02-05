#!/usr/bin/env python3
"""
Analyze image similarity between reports (requires prior extraction via extract_content.py).
Phase 1: identical images (file hash).
Phase 2: similar images (phash).
Usage: python analyze_images.py [threshold] <dirpath>
"""

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import imagehash
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from content_extractor import ContentExtractor

DEFAULT_IMAGE_SIMILARITY_THRESHOLD = 0.8

# Phash algorithm parameters (internal)
PHASH_HASH_SIZE = 32  # range 2–32, typically 8|16; larger = more precise, slower
PHASH_HIGH_FREQ_FACTOR = 8  # range 2–8, typically 2|4|8; img_size = hash_size * this (DCT input)


class ImageAnalyzer:
    """Analyzes image similarity between reports using file hash and phash."""

    def _hash_similarity(self, h1: imagehash.ImageHash | None, h2: imagehash.ImageHash | None) -> float:
        if h1 is None or h2 is None:
            return 0.0
        dist = h1 - h2
        size = h1.hash.size
        return max(0.0, 1.0 - dist / size)

    def _max_image_similarity(self, hashes_a: list, hashes_b: list) -> float:
        valid_a = [h for h in hashes_a if h is not None]
        valid_b = [h for h in hashes_b if h is not None]
        if not valid_a or not valid_b:
            return 0.0
        best = 0.0
        for ha in valid_a:
            for hb in valid_b:
                best = max(best, self._hash_similarity(ha, hb))
        return best

    # Parse authors, group, submid from report_dirpath/metadata.txt
    def _parse_report_metadata(self, report_dirpath: Path) -> tuple[str, str | None, str | None] | None:
        meta_file = report_dirpath / "metadata.txt"
        if not meta_file.exists():
            return None
        authors = ""
        group = None
        submid = None
        for line in meta_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("authors:"):
                authors = line.split(":", 1)[1].strip()
            elif line.startswith("group:"):
                val = line.split(":", 1)[1].strip()
                group = val if val else None
            elif line.startswith("submid:"):
                val = line.split(":", 1)[1].strip()
                submid = val if val else None
        return (authors, group, submid) if submid else None

    # Read groups: true/false from reports_dirpath/metadata.txt
    def _read_groups_flag(self, reports_dirpath: Path) -> bool:
        meta = reports_dirpath / "metadata.txt"
        if not meta.exists():
            return False
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.startswith("groups:"):
                return "true" in line.lower().split(":", 1)[1].strip()
        return False

    # Label for display: group if groups mode, else authors
    def _get_report_label(self, authors: str, group: str | None, groups: bool) -> str:
        return group if (groups and group) else authors

    def compare_images(self, reports_dirpath: Path, threshold: float | None = None) -> None:
        # Load report dirs sorted by submid
        groups = self._read_groups_flag(reports_dirpath)
        report_dirs = [d for d in reports_dirpath.iterdir() if d.is_dir()]

        def sort_key(rd: Path) -> tuple[int, str]:
            p = self._parse_report_metadata(rd)
            return (int(p[2]) if p and p[2] else 0, rd.name)

        report_dirs = sorted(report_dirs, key=sort_key)

        # Build entries: (report_dir, label, submid, authors, img_paths, hashes_phash)
        entries: list[tuple[Path, str, str | None, str, list[Path], list]] = []
        for report_dir in report_dirs:
            parsed = self._parse_report_metadata(report_dir)
            if not parsed:
                continue
            authors, group, submid = parsed
            images_dir = report_dir / "images"
            if not images_dir.is_dir():
                continue
            img_paths = sorted(images_dir.iterdir(), key=lambda p: p.name)
            img_paths = [p for p in img_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp")]
            if not img_paths:
                continue
            label = self._get_report_label(authors, group, groups)
            hashes_phash = []
            for p in img_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    hashes_phash.append(imagehash.phash(img, hash_size=PHASH_HASH_SIZE, highfreq_factor=PHASH_HIGH_FREQ_FACTOR))
                except Exception:
                    hashes_phash.append(None)
            entries.append((report_dir, label, submid, authors, img_paths, hashes_phash))

        if not entries:
            print("No reports with images found.")
            return

        th = threshold if threshold is not None else DEFAULT_IMAGE_SIMILARITY_THRESHOLD

        # Phase 1: identical images (MD5 file hash)
        print("Identical images (file hash):")
        stolen_by_hash: dict[str, list[tuple[str, str]]] = {}
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                _, label_i, submid_i, _, paths_i, _ = entries[i]
                _, label_j, submid_j, _, paths_j, _ = entries[j]
                hash_to_path_i = {ContentExtractor.get_file_hash(p): (p, label_i) for p in paths_i}
                hash_to_path_j = {ContentExtractor.get_file_hash(p): (p, label_j) for p in paths_j}
                common = set(hash_to_path_i) & set(hash_to_path_j)
                if not common:
                    continue
                submid_i_int = int(submid_i) if submid_i else 0
                submid_j_int = int(submid_j) if submid_j else 0
                if submid_i_int < submid_j_int:
                    copier, source = label_j, label_i
                    for h in common:
                        p_j, _ = hash_to_path_j[h]
                        img_name = p_j.stem
                        stolen_by_hash.setdefault(copier, []).append((img_name, source))
                else:
                    copier, source = label_i, label_j
                    for h in common:
                        p_i, _ = hash_to_path_i[h]
                        img_name = p_i.stem
                        stolen_by_hash.setdefault(copier, []).append((img_name, source))

        for copier in sorted(stolen_by_hash):
            pairs = stolen_by_hash[copier]
            by_source: dict[str, list[str]] = {}
            for img_name, source in pairs:
                by_source.setdefault(source, []).append(img_name)
            for source in sorted(by_source):
                imgs = sorted(by_source[source])
                print(f"{copier} stole:")
                for img in imgs:
                    print(f" - {img} from {source} ({img})")

        if not stolen_by_hash:
            print("None.")
        print()

        # Phase 2: similar images (perceptual hash)
        n = len(entries)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = self._max_image_similarity(entries[i][5], entries[j][5])

        stolen_by_phash: dict[str, list[tuple[str, str, str, float]]] = {}
        for i in range(n):
            for j in range(i + 1, n):
                _, label_i, submid_i, _, paths_i, hashes_i = entries[i]
                _, label_j, submid_j, _, paths_j, hashes_j = entries[j]
                submid_i_int = int(submid_i) if submid_i else 0
                submid_j_int = int(submid_j) if submid_j else 0
                if submid_i_int < submid_j_int:
                    copier_label, source_label = label_j, label_i
                    copier_paths, copier_hashes = paths_j, hashes_j
                    source_paths, source_hashes = paths_i, hashes_i
                elif submid_j_int < submid_i_int:
                    copier_label, source_label = label_i, label_j
                    copier_paths, copier_hashes = paths_i, hashes_i
                    source_paths, source_hashes = paths_j, hashes_j
                else:
                    continue
                for idx_a, (ha, pa) in enumerate(zip(copier_hashes, copier_paths)):
                    if ha is None:
                        continue
                    best_sim = 0.0
                    best_pb = None
                    for idx_b, (hb, pb) in enumerate(zip(source_hashes, source_paths)):
                        if hb is None:
                            continue
                        sim = self._hash_similarity(ha, hb)
                        if sim >= th and sim > best_sim:
                            best_sim = sim
                            best_pb = pb
                    if best_pb is not None:
                        stolen_by_phash.setdefault(copier_label, []).append((pa.stem, best_pb.stem, source_label, best_sim))

        circle_cells = set()
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] >= th or matrix[j, i] >= th:
                    _, _, submid_i, _, _, _ = entries[i]
                    _, _, submid_j, _, _, _ = entries[j]
                    submid_i_int = int(submid_i) if submid_i else 0
                    submid_j_int = int(submid_j) if submid_j else 0
                    if submid_i_int < submid_j_int:
                        circle_cells.add((j, i))
                    elif submid_j_int < submid_i_int:
                        circle_cells.add((i, j))

        print(f"Similar images (phash, threshold {th:.0%}):")
        for copier in sorted(stolen_by_phash):
            pairs = stolen_by_phash[copier]
            by_source: dict[str, list[tuple[str, str, float]]] = {}
            for img_c, img_s, source, sim in pairs:
                by_source.setdefault(source, []).append((img_c, img_s, sim))
            for source in sorted(by_source):
                print(f"{copier} likely copied:")
                for img_c, img_s, sim in sorted(by_source[source], key=lambda x: (x[0], x[1])):
                    print(f" - {img_c} from {source} ({img_s}) - {sim:.0%}")

        if not stolen_by_phash:
            print("None.")
        print()

        # Similarity matrix visualization
        labels = [e[1] for e in entries]
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
        plt.colorbar(im, ax=ax, label="Image similarity")
        ax.set_title(f"Image similarity matrix (threshold {th:.0%})")
        plt.tight_layout()
        plt.show()


def _parse_args() -> tuple[Path, float | None]:
    """
    Parse CLI: [threshold] dirpath.
    Returns (reports_dirpath, threshold_arg). Threshold accepts 0.6 or 60 (percent).
    """
    parser = argparse.ArgumentParser(description="Analyze image similarity of reports")
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
    th = threshold_arg if threshold_arg is not None else DEFAULT_IMAGE_SIMILARITY_THRESHOLD
    print(f"Comparing images with threshold {th:.0%}...")
    ImageAnalyzer().compare_images(reports_dirpath, threshold=threshold_arg)


if __name__ == "__main__":
    main()
