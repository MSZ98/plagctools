"""
ContentExtractor - extracts text and images from PDF reports.
FileAnalyzer - detects duplicate files by hash.
"""

import hashlib
import re
import shutil
import sys
from pathlib import Path

import fitz

DETECT_REPORT_KEYWORDS = [
    "wnioski",
    "sprawozdani",
    "laboratorium",
    "grupa",
    "data",
    "temat",
    "zadani",
    "ćwicz"
]

DETECT_REPORT_THRESHOLD = 0.6

FILENAME_PATTERN = re.compile(r"^(?:(\d+[A-Z])-)?(.+?)_(\d+)_assignsubmission", re.IGNORECASE)
GROUP_PREFIX_PATTERN = re.compile(r"^\d+[A-Z]-", re.IGNORECASE)
ASSIGNSUBMISSION_PREFIX = "assignsubmission_file_"


class ContentExtractor:
    @staticmethod
    def get_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def detect_groups(dirpath: Path) -> bool:
        pdfs = list(dirpath.glob("*.pdf")) + list(dirpath.glob("*.PDF"))
        groups = False
        for p in pdfs:
            if p.is_file() and GROUP_PREFIX_PATTERN.match(p.name):
                groups = True
                break

        reports_dirpath = dirpath / "text"
        meta_file = reports_dirpath / "metadata.txt"
        if not meta_file.exists():
            reports_dirpath.mkdir(parents=True, exist_ok=True)
            meta_file.write_text(f"groups: {groups}\n", encoding="utf-8")
        return groups

    @staticmethod
    def _parse_filename(name):
        m = FILENAME_PATTERN.match(name)
        if m:
            author = m.group(2).strip().replace(" - ", " ")
            return (m.group(1), author, m.group(3))
        stem = name.rsplit(".", 1)[0] if "." in name else name
        return (None, stem.replace(" - ", " "), None)

    def get_report_list(self, dirpath: Path, groups: bool) -> list[tuple[str | None, list[str], str | None, Path]]:
        if not dirpath.is_dir():
            print(f"Error: '{dirpath}' is not a directory", file=sys.stderr)
            sys.exit(1)

        pdfs = [p for p in (list(dirpath.glob("*.pdf")) + list(dirpath.glob("*.PDF"))) if p.is_file()]

        if not groups:
            result = []
            for p in pdfs:
                g, author, submid = self._parse_filename(p.name)
                authors = [author] if author else []
                result.append((g, authors, submid, p))
            return result

        hash_to_paths: dict[str, list[Path]] = {}
        for p in pdfs:
            try:
                h = self.get_file_hash(p)
                hash_to_paths.setdefault(h, []).append(p)
            except OSError:
                continue

        result = []
        for paths in hash_to_paths.values():
            group = None
            authors_set: set[str] = set()
            for p in paths:
                g, author, _ = self._parse_filename(p.name)
                if g is not None:
                    group = g
                if author:
                    authors_set.add(author)
            _, _, submid = self._parse_filename(paths[0].name)
            result.append((group, sorted(authors_set), submid, paths[0]))
        return result

    def extract_text(self, report_list: list) -> Path:
        """Extracts text and metadata from reports. Does not extract images."""
        if not report_list:
            raise ValueError("report_list is empty")
        base_dir = report_list[0][3].parent
        reports_dirpath = base_dir / "text"
        reports_dirpath.mkdir(exist_ok=True)

        for group, authors, submid, pdf_path in report_list:
            out_dir = reports_dirpath / pdf_path.stem
            out_dir.mkdir(exist_ok=True)
            doc = fitz.open(pdf_path)
            meta_lines = [
                f"filename: {pdf_path.name}",
                f"group: {group}" if group else "group: ",
                f"authors: {', '.join(authors)}",
                f"submid: {submid}" if submid else "submid: ",
            ]
            key_rename = {"author": "file_author", "creator": "file_creator"}
            for k, v in doc.metadata.items():
                if v:
                    key = key_rename.get(k, k)
                    meta_lines.append(f"{key}: {v}")
            (out_dir / "metadata.txt").write_text("\n".join(meta_lines), encoding="utf-8")
            for i in range(len(doc)):
                text = doc[i].get_text() or ""
                (out_dir / f"{i + 1}.txt").write_text(text, encoding="utf-8")
            doc.close()
        return reports_dirpath

    def extract_images(self, reports_dirpath: Path, dirpath: Path) -> None:
        """Extracts images from PDF to report_dir/images/. Reads PDF path from metadata.txt.
        Uses get_image_info(xrefs=True) - only images actually displayed on the page."""
        for report_dir in reports_dirpath.iterdir():
            if not report_dir.is_dir():
                continue
            meta_file = report_dir / "metadata.txt"
            if not meta_file.exists():
                continue
            filename = None
            for line in meta_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("filename:"):
                    filename = line.split(":", 1)[1].strip()
                    break
            if not filename:
                continue
            pdf_path = dirpath / filename
            if not pdf_path.exists():
                continue
            images_dir = report_dir / "images"
            images_dir.mkdir(exist_ok=True)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                img_infos = page.get_image_info(xrefs=True)
                img_idx = 0
                for info in img_infos:
                    xref = info.get("xref", 0)
                    if xref <= 0:
                        continue
                    try:
                        base_img = doc.extract_image(xref)
                    except Exception:
                        continue
                    img_idx += 1
                    ext = base_img["ext"]
                    out_path = images_dir / f"page{page_num + 1}_img{img_idx}.{ext}"
                    out_path.write_bytes(base_img["image"])
            doc.close()

    def detect_report(self, report_dirpath: Path) -> float:
        combined = ""
        for f in report_dirpath.iterdir():
            if f.suffix == ".txt" and f.name != "metadata.txt" and f.stem.isdigit():
                combined += (f.read_text(encoding="utf-8") or "").lower()
        if not combined:
            return 0.0
        found = sum(1 for kw in DETECT_REPORT_KEYWORDS if kw in combined)
        return found / len(DETECT_REPORT_KEYWORDS)

    def print_report_certainty(self, reports_dirpath: Path) -> None:
        for report in reports_dirpath.iterdir():
            if report.is_dir():
                prob = self.detect_report(report)
                print(f"{prob * 100:>4.0f}% {report.name}.pdf")

    def refuse_non_reports(self, reports_dirpath: Path) -> None:
        for report in reports_dirpath.iterdir():
            if not report.is_dir():
                continue
            prob = self.detect_report(report)
            if prob < DETECT_REPORT_THRESHOLD:
                shutil.rmtree(report)


class FileAnalyzer:
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

    def _extract_filename(self, full_name: str) -> str:
        if ASSIGNSUBMISSION_PREFIX in full_name:
            return full_name.split(ASSIGNSUBMISSION_PREFIX, 1)[1]
        return full_name

    def _read_groups_flag(self, reports_dirpath: Path) -> bool:
        meta = reports_dirpath / "metadata.txt"
        if not meta.exists():
            return False
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.startswith("groups:"):
                return "true" in line.lower().split(":", 1)[1].strip()
        return False

    def detect_duplicates(self, dirpath: Path, reports_dirpath: Path) -> None:
        groups = self._read_groups_flag(reports_dirpath)
        entries: list[tuple[str, str | None, str, list[Path]]] = []

        for report_dir in reports_dirpath.iterdir():
            if not report_dir.is_dir():
                continue
            parsed = self._parse_report_metadata(report_dir)
            if not parsed:
                continue
            authors, group, submid = parsed

            files_with_submid = [
                p for p in dirpath.iterdir()
                if p.is_file() and f"_{submid}_" in p.name
            ]
            if files_with_submid:
                entries.append((authors, group, submid, files_with_submid))

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                authors_i, group_i, submid_i, files_i = entries[i]
                authors_j, group_j, submid_j, files_j = entries[j]

                hashes_i = {ContentExtractor.get_file_hash(f): f for f in files_i}
                hashes_j = {ContentExtractor.get_file_hash(f): f for f in files_j}

                common_hashes = set(hashes_i) & set(hashes_j)
                if not common_hashes:
                    continue

                if int(submid_i) < int(submid_j):
                    earlier_label = group_i if groups and group_i else authors_i
                    later_label = group_j if groups and group_j else authors_j
                    for h in common_hashes:
                        f_earlier = hashes_i[h]
                        f_later = hashes_j[h]
                        print(f"{later_label} stole {self._extract_filename(f_later.name)} from {earlier_label} {self._extract_filename(f_earlier.name)}")
                else:
                    earlier_label = group_j if groups and group_j else authors_j
                    later_label = group_i if groups and group_i else authors_i
                    for h in common_hashes:
                        f_earlier = hashes_j[h]
                        f_later = hashes_i[h]
                        print(f"{later_label} stole {self._extract_filename(f_later.name)} from {earlier_label} {self._extract_filename(f_earlier.name)}")
