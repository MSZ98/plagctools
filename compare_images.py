"""
Porównanie obrazów z plików PDF - wykrywanie podobieństwa (plagiatu).
Ekstrahuje obrazy, porównuje między dokumentami, wyświetla macierz.
"""

import sys
from collections import defaultdict
from pathlib import Path

import fitz
import imagehash
import numpy as np
from PIL import Image

from pdf_utils import extract_author, get_pdf_date, normalize_author

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Gdy próg nie podany: próg = mediana * ten współczynnik
AUTOTHRESHOLD_MEDIAN_FACTOR = 1.5


def _parse_threshold(s: str) -> float:
    """Parsuje próg: 0.60, 60, 60% -> 0.60"""
    s = s.strip().rstrip("%")
    v = float(s)
    return v / 100.0 if v > 1 else v


def extract_images_from_pdf(pdf_path: Path, out_base: Path) -> list[Path]:
    """Ekstrahuje obrazy z PDF do out_base/<pdf_stem>/."""
    out_dir = out_base / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_idx, img in enumerate(page.get_images()):
            xref = img[0]
            base_img = doc.extract_image(xref)
            ext = base_img["ext"]
            out_path = out_dir / f"page{page_num + 1}_img{img_idx + 1}.{ext}"
            with open(out_path, "wb") as f:
                f.write(base_img["image"])
            paths.append(out_path)
    doc.close()
    return paths


def get_image_hashes(img_paths: list[Path]) -> list[imagehash.ImageHash]:
    """Oblicza phash dla każdego obrazu."""
    hashes = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            hashes.append(imagehash.phash(img))
        except Exception:
            hashes.append(None)
    return hashes


def hash_similarity(h1: imagehash.ImageHash | None, h2: imagehash.ImageHash | None) -> float:
    """Podobieństwo 0-1 na podstawie odległości Hamminga."""
    if h1 is None or h2 is None:
        return 0.0
    dist = h1 - h2
    size = h1.hash.size
    return max(0.0, 1.0 - dist / size)


def doc_image_similarity(hashes_a: list, hashes_b: list) -> float:
    """Podobieństwo dokumentów na podstawie obrazów (średnia najlepszych dopasowań)."""
    valid_a = [h for h in hashes_a if h is not None]
    valid_b = [h for h in hashes_b if h is not None]
    if not valid_a or not valid_b:
        return 0.0
    sim_a_to_b = []
    for ha in valid_a:
        best = max((hash_similarity(ha, hb) for hb in valid_b), default=0.0)
        sim_a_to_b.append(best)
    sim_b_to_a = []
    for hb in valid_b:
        best = max((hash_similarity(ha, hb) for ha in valid_a), default=0.0)
        sim_b_to_a.append(best)
    return (sum(sim_a_to_b) / len(valid_a) + sum(sim_b_to_a) / len(valid_b)) / 2


def main():
    if len(sys.argv) < 2:
        print("Użycie: python compare_images.py [próg] <katalog>")
        print("  próg   – opcjonalnie; podobieństwo 0–1 lub 0–100 (np. 0.60 lub 60)")
        print("          brak = obliczany z mediany (AUTOTHRESHOLD_MEDIAN_FACTOR)")
        print("  katalog – ścieżka do folderu z plikami PDF")
        sys.exit(1)

    if len(sys.argv) >= 3:
        try:
            threshold = _parse_threshold(sys.argv[1])
        except ValueError:
            print(f"Błąd: '{sys.argv[1]}' nie jest poprawnym progiem (liczba 0–1 lub 0–100)")
            sys.exit(1)
        if not 0 < threshold <= 1:
            print("Błąd: próg musi być w zakresie (0, 1] lub (0, 100]")
            sys.exit(1)
        folder = Path(sys.argv[2])
    else:
        threshold = None
        folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Błąd: '{folder}' nie jest katalogiem")
        sys.exit(1)

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"Brak plików PDF w katalogu {folder}/")
        return

    names = [p.stem for p in pdfs]
    authors = [extract_author(str(p)) for p in pdfs]
    display_names = [a or n for a, n in zip(authors, names)]
    dates = [get_pdf_date(str(p)) for p in pdfs]

    # Ekstrakcja obrazów
    print("Ekstrakcja obrazów z PDF:")
    extracted_base = folder / "extracted_images"
    all_img_paths = []
    for pdf in pdfs:
        paths = extract_images_from_pdf(pdf, extracted_base)
        all_img_paths.append(paths)
        print(f"  {pdf.stem}: {len(paths)} obrazów")

    print()

    # Autor, data oddania, nazwa pliku
    rows = []
    for pdf, author, dt in zip(pdfs, authors, dates):
        auth_str = author or "(autor nieznany)"
        date_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "nieznana"
        rows.append((auth_str, date_str, pdf.name))
    w_auth = max(len(r[0]) for r in rows) if rows else 0
    w_date = max(len(r[1]) for r in rows) if rows else 0
    w_auth = max(w_auth, len("Autor"))
    w_date = max(w_date, len("Data oddania"))
    fmt = f"  {{0:<{w_auth}}} | {{1:<{w_date}}} | {{2}}"
    print(fmt.format("Autor", "Data oddania", "Nazwa pliku"))
    for auth, date, name in rows:
        print(fmt.format(auth, date, name))
    print()

    # Analiza powtarzających się autorów
    author_to_docs = defaultdict(list)
    for i, a in enumerate(authors):
        if a:
            author_to_docs[normalize_author(a)].append(display_names[i])
    duplicates = [docs for docs in author_to_docs.values() if len(docs) > 1]
    if duplicates:
        print("Uwaga: powtarzające się imię i nazwisko:")
        for docs in duplicates:
            print(f"  {', '.join(docs)}")
        print()

    # Hashe obrazów
    all_hashes = [get_image_hashes(paths) for paths in all_img_paths]

    # Macierz podobieństwa
    n = len(pdfs)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = doc_image_similarity(all_hashes[i], all_hashes[j])

    if threshold is None:
        off_diag = [matrix[i, j] for i in range(n) for j in range(n) if i != j]
        med = float(np.median(off_diag)) if off_diag else 0.0
        threshold = min(1.0, med * AUTOTHRESHOLD_MEDIAN_FACTOR)
        print(f"Próg = mediana * współczynnik")
        print(f"{threshold:.0%} = {med:.0%} * {AUTOTHRESHOLD_MEDIAN_FACTOR:.0%}\n")

    # Analiza plagiatu obrazów
    print(f"Analiza plagiatu obrazów (próg {threshold:.0%}):")
    suspicions = []
    circle_cells = set()  # (row, col) = (copier_idx, source_idx)
    for i in range(n):
        for j in range(i + 1, n):
            sim = matrix[i, j]
            if sim >= threshold:
                di, dj = dates[i], dates[j]
                copier = display_names[i]
                source = display_names[j]
                if di and dj:
                    if di > dj:
                        suspicions.append((copier, source, sim, False))
                        circle_cells.add((i, j))
                    elif dj > di:
                        suspicions.append((source, copier, sim, False))
                        circle_cells.add((j, i))
                    else:
                        suspicions.append((copier, source, sim, True))  # daty równe
                else:
                    suspicions.append((copier, source, sim, True))
    for a, b, sim, unknown in sorted(suspicions, key=lambda x: -x[2]):
        if unknown:
            print(f"  Sprawozdania {a} i {b} mają podobne obrazy ({sim:.0%})")
        else:
            print(f"  {a} prawdopodobnie ściągał obrazy od {b} ({sim:.0%})")
    if not suspicions:
        print("  Brak par powyżej progu.")
    print()

    # Heatmap
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle

    colors_white_red = [(1, 1, 1), (1, 0.5, 0.5)]
    cmap = LinearSegmentedColormap.from_list("white_red", colors_white_red)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    short_names = [d[:20] + "..." if len(d) > 20 else d for d in display_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_yticklabels(short_names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", color="black", fontsize=9)
            if (i, j) in circle_cells:
                circle = Circle((j, i), 0.4, fill=False, edgecolor="green", linewidth=2)
                ax.add_patch(circle)

    plt.colorbar(im, ax=ax, label="Podobieństwo obrazów")
    ax.set_title(f"Macierz podobieństwa obrazów (próg {threshold:.0%})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
