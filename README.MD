# plagctools

**Authors:** MSZ98 & C

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

---

## Input directory

The directory passed as argument must contain submission files **directly from Moodle**, downloaded via:

**Zadanie → Zobacz wszystkie zadania → Akcja oceniania → Pobierz wszystkie przesłane rozwiązania**

Filenames must follow the format:

```
<group (optional)>-<name>_<SubmissionID>_assignsubmission_file_<filename>
```

Examples:
- `1A-Jan Kowalski_234189_assignsubmission_file_sprawozdanie.pdf` (with group)
- `Jan Kowalski_234189_assignsubmission_file_sprawozdanie.pdf` (no group)

---

## Workflow

1. **extract_content.py** – extract content from PDFs (required first step)
2. **analyze_text.py** – text similarity analysis
3. **analyze_images.py** – image similarity analysis

---

## extract_content.py

**Usage:** `python extract_content.py <dirpath>`

**What it does:**
- Scans directory for PDF files (e.g. `ni_writer`)
- Detects group mode (prefix `1A-` in filenames)
- Extracts page text to `dirpath/text/<report>/1.txt`, `2.txt`, ...
- Extracts images to `dirpath/text/<report>/images/`
- Creates `metadata.txt` with authors, group, submid
- Filters out files that do not look like reports
- Detects duplicate files (MD5)

**Example:** `python extract_content.py ni_writer`

---

## analyze_text.py

**Usage:** `python analyze_text.py [threshold] <dirpath>`

**What it does:**
- Requires prior extraction (`extract_content.py`)
- Loads text from `dirpath/text/`
- TF-IDF + cosine similarity (unigrams + bigrams)
- Prints similar report pairs
- Displays similarity matrix (matplotlib)

**Threshold:** 0–1 or 0–100 (e.g. `0.6`, `60`). Default: 0.6.

**Examples:**
```bash
python analyze_text.py ni_writer
python analyze_text.py 0.7 ni_writer
```

---

## analyze_images.py

**Usage:** `python analyze_images.py [threshold] <dirpath>`

**What it does:**
- Requires prior extraction (`extract_content.py`)
- **Phase 1:** identical images (file MD5)
- **Phase 2:** similar images (perceptual hash – phash)
- Prints who copied from whom
- Displays similarity matrix (matplotlib)

**Threshold:** 0–1 or 0–100. Default: 0.8.

**Examples:**
```bash
python analyze_images.py ni_writer
python analyze_images.py 0.9 ni_writer
```

---

## Output structure

```
ni_writer/                    # input directory with PDFs
├── *.pdf
└── text/                     # created by extract_content
    ├── metadata.txt          # groups: true/false
    └── Autor_123_.../        # per-report directory
        ├── metadata.txt      # authors, group, submid
        ├── 1.txt, 2.txt...   # page text
        └── images/           # extracted images
```
