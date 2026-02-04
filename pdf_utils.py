"""
Wspólne funkcje do ekstrakcji informacji z PDF.
"""

import re
from datetime import datetime
from pathlib import Path

from PyPDF2 import PdfReader


def get_pdf_date(path: str) -> datetime | None:
    """Pobiera datę oddania (creation/modification). Fallback: data modyfikacji pliku."""
    try:
        reader = PdfReader(path)
        meta = reader.metadata
        for attr in ("creation_date", "modification_date"):
            val = getattr(meta, attr, None)
            if isinstance(val, datetime):
                return val
            if val:
                m = re.search(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", str(val))
                if m:
                    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    h, mi, s = int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0)
                    return datetime(y, mo, d, h, mi, s)
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime)
    except Exception:
        return None


AUTHOR_KEYWORDS = r"(?:Autor|Autorzy|Wykonawca|Wykonawcy|Wykonał|Wykonali|Student|Studenci)\s*[:\s]+"
NAME_PATTERN = r"[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻa-ząćęłńóśźż\-]+"


def _extract_author_from_text(text: str) -> str | None:
    """Szuka autorów w tekście - po słowach Autor, Wykonali, Wykonawca itp. Obsługuje wielu autorów."""
    if not text or len(text) < 5:
        return None
    # Szukaj sekcji po słowie kluczowym (np. "Wykonali:" lub "Autor:")
    m = re.search(AUTHOR_KEYWORDS + r"(.+?)(?=\n\n|\n[A-ZŁ][a-ząćęłńóśźż]+\s*[:\s]|$)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        # Fallback: proste wzorce w pierwszym 1500 znakach
        chunk = text[:1500]
        m = re.search(AUTHOR_KEYWORDS + r"(.+?)(?=\n|$)", chunk, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    block = m.group(1).strip()
    # Wyciągnij wszystkie imiona i nazwiska (np. "Krzysztof Kawecki", "•Jan Kowalski")
    names = re.findall(r"[•\-\*·]?\s*(" + NAME_PATTERN + r")", block)
    # Odrzuć typowe nie-imiona (nagłówki, etykiety)
    not_names = {"data oddania", "imię nazwisko", "wstęp", "ćwiczenie", "cel ćwiczenia", "narzędzia informatyczne",
                 "wstęp ćwiczenie", "cel ćwiczenia", "data wykonania", "politechnika świętokrzyska"}
    names = [n.strip() for n in names if len(n.strip()) > 4 and " " in n.strip()
             and n.strip().lower() not in not_names and not any(nn in n.strip().lower() for nn in not_names)]
    # Usuń duplikaty zachowując kolejność
    seen = set()
    unique = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            unique.append(n)
    if unique:
        return ", ".join(unique)
    return None


GENERIC_AUTHORS = {"student", "author", "unknown", "nieznany", "politechnika", "narzędzia", "informatyczne"}


INVALID_FROM_FILENAME = {"sprawozdanie", "libre", "office", "writter", "ni", "kk"}


def _author_from_filename(stem: str) -> str | None:
    """Próba wyciągnięcia imienia i nazwiska z nazwy pliku."""
    # "Sprawozdanie 1 Piotr Siembida" -> "Piotr Siembida"
    m = re.search(r"(?:sprawozdanie|ni)\s*\d*\.?\d*\s+([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+\s+[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-]+)", stem, re.I)
    if m:
        name = m.group(1).strip()
        if not any(inv in name.lower() for inv in INVALID_FROM_FILENAME):
            return name
    # "Pryhara_Kyrylo_sprawozdanie" -> "Kyrylo Pryhara"
    m = re.search(r"([A-Za-z]+)_([A-Za-z]+)_", stem)
    if m:
        return f"{m.group(2)} {m.group(1)}"
    # "Sprawozdanie_Derevenko-1" -> "Derevenko"
    m = re.search(r"_?([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)(?:-\d+)?$", stem)
    if m:
        name = m.group(1)
        if len(name) > 2 and name.lower() not in INVALID_FROM_FILENAME:
            return name
    return None


def extract_author(path: str) -> str | None:
    """Ekstrahuje imię i nazwisko autora z PDF (metadata, tekst, lub nazwa pliku)."""
    stem = Path(path).stem
    try:
        reader = PdfReader(path)
        meta = reader.metadata
        author = getattr(meta, "author", None)
        if author:
            a = str(author).strip()
            al = a.lower()
            if len(a) > 2 and not any(g in al for g in GENERIC_AUTHORS):
                return a
        # Fallback: przeszukaj tekst PDF (pierwsze 3 strony - autor zwykle na początku)
        skip_text = {"narzędzia informatyczne", "politechnika", "sprawozdanie"}
        if reader.pages:
            text = ""
            for i in range(min(3, len(reader.pages))):
                text += reader.pages[i].extract_text() or ""
            from_text = _extract_author_from_text(text)
            if from_text and from_text.lower() not in skip_text and not any(g in from_text.lower() for g in GENERIC_AUTHORS):
                return from_text
    except Exception:
        pass
    return _author_from_filename(stem)


def normalize_author(name: str | None) -> str:
    """Normalizacja do porównania (małe litery, bez nadmiarowych spacji)."""
    if not name:
        return ""
    return " ".join(str(name).lower().split())
