# Music Genre Classification on FMA-small

Classical machine-learning pipeline for top-level music genre classification on the
**FMA-small** dataset (8,000 clips of 30 s, 8 balanced genres). Features are extracted
once from the raw audio with `librosa` and exported to a CSV; a Random Forest is then
trained and evaluated with `scikit-learn`. The tuned model reaches **0.420 accuracy /
0.415 macro-F1** on the official held-out test set (random baseline = 12.5%).

**Authors:** Ana Belén Cabello Martínez · Francisco Javier Muñoz Mohedano
**Course:** Advanced Technologies Integration — Universidad de Jaén

Full write-up and results are in the project manuscript (PDF).

---

## Repository structure

```
.
├── 01_extraccion.ipynb       # Phase 1 — feature extraction from raw audio  -> fma_small_features.csv
├── 02_modelado.ipynb     # Phase 2 — training, tuning and evaluation
├── requirements.txt            # Python dependencies
└── README.md
```

The pipeline runs in two phases connected by a single CSV. Phase 1 reads the 7.2 GB
audio archive and is slow (~1–4 h on CPU); it only needs to be run **once** to produce
`fma_small_features.csv` (a few MB). Phase 2 works entirely from that CSV.

> **Quick reproduction:** `fma_small_features.csv` is already included in this repository, so the reviewer can **skip Phase 1 entirely** and reproduce all reported results by running only `02_modelado.ipynb`. Phase 1 is only needed to regenerate the CSV from the raw audio.

---

## 1. Software requirements and dependencies

- **Python 3.14**
- **ffmpeg** (system package) — required by `librosa` to decode MP3 files
  - Windows: https://www.gyan.dev/ffmpeg/builds/ (add `bin` to PATH)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- Python packages (see `requirements.txt`):

```
librosa
pandas
numpy
tqdm
scikit-learn
matplotlib
jupyter
```

Install everything (ideally inside a virtual environment):

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Dataset preparation

This project extracts features from the **raw audio** (it does not use FMA's
precomputed feature tables). Download the two archives from the official FMA repository
(https://github.com/mdeff/fma):

- `fma_small.zip` (~7.2 GB) — the 8,000 audio clips
- `fma_metadata.zip` (~342 MB) — metadata, including `tracks.csv`

Unzip both. You should end up with a folder of audio (with 3-digit subfolders such as
`000/`, `001/`, …) and a metadata folder containing `tracks.csv`.

Then open `01_extraccion.ipynb` and edit the two path variables at the top to
point to where you unzipped them:

```python
AUDIO_DIR    = ".../fma_small"        # folder with the 000/, 001/, ... subfolders
METADATA_DIR = ".../fma_metadata"     # folder containing tracks.csv
```

Only `tracks.csv` is read from the metadata folder: the `track_id` (index), the
`('track', 'genre_top')` label, the `('set', 'subset')` column (to select `small`),
and the `('set', 'split')` column (the official train/validation/test partition).

---

## 3. Training and evaluation procedures

**Phase 1 — feature extraction** (`01_extraccion.ipynb`) — *optional, the CSV is already provided*

Run all cells. For each track the notebook loads 30 s of audio at 22,050 Hz mono and
computes an **89-dimensional feature vector** (MFCCs, chroma, spectral contrast,
spectral centroid/bandwidth/rolloff, ZCR, RMS, and tempo — each summarized by its mean
and standard deviation over time). Results are written incrementally to
`fma_small_features.csv`; the run is resumable (re-running skips tracks already saved).

**Phase 2 — modeling and evaluation** (`02_modelado.ipynb`)

Run all cells. The notebook:

1. Loads `fma_small_features.csv` and rebuilds the official train/validation/test split.
2. Standardizes features (`StandardScaler` fit on training data only).
3. Establishes baselines (`DummyClassifier`, `k`-NN) on validation.
4. Compares Logistic Regression, Random Forest, SVM (RBF) and Histogram Gradient
   Boosting on validation.
5. Tunes the Random Forest with `GridSearchCV` using a `PredefinedSplit` (so the search
   validates on the official validation set, not a random fold).
6. Refits the tuned model on train + validation and evaluates **once** on the test set:
   accuracy, macro-F1, per-class `classification_report`, and a row-normalized
   confusion matrix.
7. Reports feature importances and the most-confused genre pairs.

All random components use `RANDOM_STATE = 42`.

---

## 4. Commands to reproduce the reported results

**Quick path (recommended)** — the feature CSV is already in the repo, so just run Phase 2:

```bash
# 1. Install dependencies (ffmpeg is NOT needed for this path)
pip install -r requirements.txt

# 2. Run the modeling notebook end to end to reproduce the tables and figures
jupyter notebook 02_modelado.ipynb
```

**Full path (regenerate the CSV from the raw audio)** — only if you want to rebuild the features:

```bash
# Install dependencies + ffmpeg (see section 1)
pip install -r requirements.txt

# Edit AUDIO_DIR and METADATA_DIR in 01_extraccion.ipynb, then run it to produce fma_small_features.csv
jupyter notebook 01_extraccion.ipynb
jupyter notebook 02_modelado.ipynb
```

To run a notebook non-interactively from the command line, use:

```bash
jupyter nbconvert --to notebook --execute 02_modelado.ipynb
```

Expected output (tuned Random Forest, official test split):

| Metric | Value |
|---|---|
| Accuracy | 0.420 |
| Macro-F1 | 0.415 |
| Random baseline | 0.125 |

> Note: `librosa.beat.beat_track` and small differences across `librosa` / `numba` /
> BLAS versions can cause tiny numerical variations, but the seeded classifiers are
> otherwise deterministic.

---

## Notes

- The 6 audio files documented as corrupt in FMA-small
  (`98565, 98567, 98569, 99134, 108925, 133297`) are skipped; clips shorter than 1 s are
  discarded, leaving 7,994 usable tracks (6,394 / 800 / 800).
- The helpers `load_tracks()` and `get_audio_path()` are adapted from the official
  `utils.py` in the `mdeff/fma` repository.
- FMA dataset: M. Defferrard, K. Benzi, P. Vandergheynst, X. Bresson, *FMA: A Dataset for Music Analysis*, ISMIR 2017 — https://github.com/mdeff/fma. Tracks are distributed under per-track Creative Commons licences.
