# plagctools

**Autor:** MSZ98 & C

---

## Instalacja

```bash
pip install -r requirements.txt
```

### Wymagania

- Python 3.10+
- PyPDF2, numpy, matplotlib, pymupdf, imagehash, Pillow

---

## Użycie

### Porównanie tekstu

```bash
python compare_text.py [próg] <katalog>
```

| Argument | Opis |
|----------|------|
| `próg` | Opcjonalnie. Próg podobieństwa (0–1 lub 0–100). Pary powyżej progu są uznawane za podejrzane. **Pominięcie = obliczany z mediany** (współczynnik w `AUTOTHRESHOLD_MEDIAN_FACTOR`). |
| `katalog` | Ścieżka do folderu z plikami PDF. |

Przykłady:
```bash
python compare_text.py katalog/             # próg obliczany z mediany
python compare_text.py 0.30 katalog/        # próg 30%
python compare_text.py 30 katalog/          # 30 = 30%
python compare_text.py 0.25 ./sprawozdania
```

### Porównanie obrazów

```bash
python compare_images.py [próg] <katalog>
```

| Argument | Opis |
|----------|------|
| `próg` | Opcjonalnie. Próg podobieństwa obrazów (0–1 lub 0–100). **Pominięcie = obliczany z mediany.** |
| `katalog` | Ścieżka do folderu z plikami PDF. |

Przykłady:
```bash
python compare_images.py katalog/           # próg obliczany z mediany
python compare_images.py 0.60 katalog/
python compare_images.py 60 ./pdf
```

Katalog musi zawierać pliki PDF. Skrypty automatycznie przetworzą wszystkie pliki `*.pdf`.

---

## Co robią skrypty

### compare_text.py

1. **Ekstrakcja** – wyciąga tekst z każdego PDF
2. **Autor i data** – odczytuje autora (z metadanych lub tekstu) oraz datę oddania
3. **Porównanie** – liczy podobieństwo tekstów (SequenceMatcher)
4. **Analiza** – na podstawie dat ustala, kto mógł ściągnąć od kogo (późniejszy od wcześniejszego)
5. **Wynik** – tabela autorów, dat i plików; lista podejrzeń; heatmapa z macierzą podobieństwa

### compare_images.py

1. **Ekstrakcja** – wyciąga obrazy z PDF do `extracted_images/<nazwa_pliku>/`
2. **Autor i data** – jak w compare_text
3. **Porównanie** – perceptual hash (phash) między obrazami z różnych dokumentów
4. **Analiza** – jak w compare_text, ale dla podobieństwa obrazów
5. **Wynik** – jak w compare_text + heatmapa podobieństwa obrazów

### pdf_utils.py

Moduł wspólny z funkcjami:

- `get_pdf_date(path)` – data oddania (metadata lub data modyfikacji pliku)
- `extract_author(path)` – imię i nazwisko (metadata, tekst po „Autor:”, „Wykonali:” itd., lub nazwa pliku)
- `normalize_author(name)` – normalizacja do porównań

---

## Próg podobieństwa

Próg jest **opcjonalnym pierwszym argumentem**. Gdy go pominiemy, jest obliczany z mediany podobieństw (poza przekątną). Współczynnik: `AUTOTHRESHOLD_MEDIAN_FACTOR` na początku pliku.

Gdy podany:

- **Ułamek** (0–1): `0.30`, `0.60`
- **Procent** (0–100): `30`, `60` – wartość > 1 jest traktowana jako procent
- Z sufiksem `%`: `30%`

Im niższy próg, tym więcej par zostanie zgłoszonych (większa czułość, więcej fałszywych alarmów). Im wyższy – tym mniej (tylko wyraźne podobieństwa). Obliczanie z mediany dostosowuje próg do rozkładu podobieństw w danym zestawie sprawozdań.

---

## Interpretacja wyników

### Tabela

| Autor | Data oddania | Nazwa pliku |
|-------|--------------|-------------|
| Jan Kowalski | 2026-02-01 18:00 | sprawozdanie.pdf |

### Analiza plagiatu

- **„X prawdopodobnie ściągał od Y”** – X oddał później niż Y, podobieństwo powyżej progu
- **„Sprawozdania X i Y są podobne”** – podobieństwo powyżej progu, ale daty równe lub nieznane

### Heatmapa

- **Biały** – małe podobieństwo
- **Czerwony** – duże podobieństwo
- **Zielone kółko** – para z ustalonym kierunkiem (kto od kogo), zgodnie z datami
