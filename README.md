# QuitOS
ML backed habit quitting, utilize machine learning to predict cravings and how to avoid relapse

Repo Structure:
QuitOS/
│
├── README.md                  # Project overview, setup, pipeline, results, and roadmap.
├── requirements.txt           # Python dependencies.
├── .gitignore                 # Files and generated artifacts excluded from Git.
│
├── data/
│   ├── raw/                   # Original, unchanged source dataset.
│   └── processed/             # Cleaned and model-ready datasets.
│
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory analysis of the dataset.
│
├── src/
│   ├── __init__.py            # Makes src a Python package.
│   ├── config.py              # Shared constants, feature lists, targets, paths, and thresholds.
│   │
│   ├── data/
│   │   ├── __init__.py        # Makes data a Python package.
│   │   ├── load_data.py       # Loads and validates raw dataset files.
│   │   └── preprocess.py      # Cleans and standardizes raw observations.
│   │
│   ├── features/
│   │   ├── __init__.py        # Makes features a Python package.
│   │   └── build_features.py  # Builds ML features and future-craving targets.
│   │
│   ├── models/
│   │   ├── __init__.py        # Makes models a Python package.
│   │   ├── train_linear.py    # Trains the craving-intensity regression baseline.
│   │   ├── train_logistic.py  # Trains the high-craving classification baseline.
│   │   └── evaluate.py        # Shared regression and classification metrics.
│   │
│   └── utils/
│       ├── __init__.py        # Makes utils a Python package.
│       └── splits.py           # Creates participant-level train/test splits.
│
├── models/                    # Saved trained models, scalers, and encoders.
├── results/                   # Metrics, plots, and experiment outputs.
│
└── tests/
    ├── test_preprocess.py      # Tests data-cleaning behavior.
    ├── test_features.py        # Tests feature and target alignment.
    └── test_splits.py          # Tests against participant leakage.

This project uses the publicly available EMASENS dataset from Leppin et al. (2026). Dataset files are not redistributed in this repository. See the original EMASENS repository for access.