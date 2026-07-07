# Dataset

This directory holds the [Doctor's Handwritten Prescription BD Dataset](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset)
from Kaggle: cropped images of handwritten medicine names, each labeled with the medicine's
brand name and its generic name.

## Download instructions

The images are not committed to this repo: download them from Kaggle into this `data/` directory.

### Option 1: Kaggle CLI (recommended)

```bash
# One-time setup: install the CLI and add your Kaggle API token.
# Get kaggle.json from https://www.kaggle.com/settings -> "Create New Token"
pip install kaggle
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# From the repo root, download and unzip into data/
kaggle datasets download -d mamun1113/doctors-handwritten-prescription-bd-dataset -p data/ --unzip
```

### Option 2: Manual download

1. Open the [dataset page](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset)
   and click **Download** (requires a free Kaggle account).
2. Unzip the archive and move the `Training/`, `Testing/`, and `Validation/` folders into this `data/` directory.

## What's in the data

The dataset is split into three folders. Each split has a `*_labels.csv` file and a `*_words/` folder
of PNG images:

```
data/
├── Training/
│   ├── training_labels.csv
│   └── training_words/      # 3120 PNG images (0.png ... 3119.png)
├── Testing/
│   ├── testing_labels.csv
│   └── testing_words/       # 780 PNG images
└── Validation/
    ├── validation_labels.csv
    └── validation_words/    # 780 PNG images
```

| Split | Images | Label rows |
| --- | ---: | ---: |
| Training | 3120 | 3120 |
| Testing | 780 | 780 |
| Validation | 780 | 780 |

Each image is a small RGBA PNG (roughly 200×100 px) showing a single handwritten medicine name.

The label CSVs have three columns:

| Column | Description | Example |
| --- | --- | --- |
| `IMAGE` | Filename of the image in the split's `*_words/` folder | `0.png` |
| `MEDICINE_NAME` | Handwritten brand name shown in the image (the OCR ground truth) | `Aceta` |
| `GENERIC_NAME` | The medicine's generic/active ingredient | `Paracetamol` |

The Testing split covers 78 distinct medicines. In this repo, the Testing split is used as the demo
evaluation set, while the Training split can serve as a labeled pool for prompt optimization (see the
main [README](../README.md)).
