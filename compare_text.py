"""
Porównanie tekstu z plików PDF - wykrywanie podobieństwa (plagiatu).
Wyświetla macierz podobieństwa między dokumentami.
"""

import re
import sys
from pathlib import Path

import numpy as np
from PyPDF2 import PdfReader

from pdf_utils import extract_author, get_pdf_date

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Gdy próg nie podany: próg = mediana * ten współczynnik
AUTOTHRESHOLD_MEDIAN_FACTOR = 2.5


def _parse_threshold(s: str) -> float:
    """Parsuje próg: 0.30, 30, 30% -> 0.30"""
    s = s.strip().rstrip("%")
    v = float(s)
    return v / 100.0 if v > 1 else v


def extract_text_from_pdf(path: str) -> str:
    """Wyciąga tekst z pliku PDF."""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"[BŁĄD: {e}]"


def normalize_text(text: str) -> str:
    """Normalizacja: małe litery, usunięcie nadmiarowych spacji."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity_ratio(text1: str, text2: str) -> float:
    """Podobieństwo tekstów 0-1 (SequenceMatcher)."""
    from difflib import SequenceMatcher
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()


def main():
    if len(sys.argv) < 2:
        print("Użycie: python compare_text.py [próg] <katalog>")
        print("  próg   – opcjonalnie; podobieństwo 0–1 lub 0–100 (np. 0.30 lub 30)")
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
    texts = [normalize_text(extract_text_from_pdf(str(p))) for p in pdfs]

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
    from pdf_utils import normalize_author
    from collections import defaultdict
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

    n = len(pdfs)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                # SequenceMatcher nie jest symetryczny - używamy max
                sim = max(similarity_ratio(texts[i], texts[j]), similarity_ratio(texts[j], texts[i]))
                matrix[i, j] = sim

    if threshold is None:
        off_diag = [matrix[i, j] for i in range(n) for j in range(n) if i != j]
        med = float(np.median(off_diag)) if off_diag else 0.0
        threshold = min(1.0, med * AUTOTHRESHOLD_MEDIAN_FACTOR)
        print(f"Próg = mediana * współczynnik")
        print(f"{threshold:.0%} = {med:.0%} * {AUTOTHRESHOLD_MEDIAN_FACTOR:.0%}\n")

    # Analiza: kto od kogo mógł ściągnąć (późniejszy od wcześniejszego)
    print(f"Analiza plagiatu tekstu (próg {threshold:.0%}):")
    suspicions = []
    circle_cells = set()  # (row, col) = (copier_idx, source_idx) - tylko gdy znamy kierunek
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
                        circle_cells.add((i, j))  # i późniejszy, j wcześniejszy
                    elif dj > di:
                        suspicions.append((source, copier, sim, False))
                        circle_cells.add((j, i))  # j późniejszy, i wcześniejszy
                    else:
                        suspicions.append((copier, source, sim, True))  # daty równe
                else:
                    suspicions.append((copier, source, sim, True))
    for a, b, sim, unknown in sorted(suspicions, key=lambda x: -x[2]):
        if unknown:
            print(f"  Sprawozdania {a} i {b} są podobne ({sim:.0%})")
        else:
            print(f"  {a} prawdopodobnie ściągał od {b} ({sim:.0%})")
    if not suspicions:
        print("  Brak par powyżej progu.")
    print()

    # Heatmap: biały (0) -> czerwony półintensywny (1)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle

    colors_white_red = [(1, 1, 1), (1, 0.5, 0.5)]  # biały -> czerwony 50%
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

    plt.colorbar(im, ax=ax, label="Podobieństwo")
    ax.set_title(f"Macierz podobieństwa tekstów (próg {threshold:.0%})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
