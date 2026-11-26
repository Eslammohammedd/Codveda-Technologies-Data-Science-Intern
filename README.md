# 📊 Codveda Data Science Internship Portfolio
[![CI](https://github.com/Eslammohammedd/Codveda-Technologies-Data-Science-Intern/actions/workflows/ci.yml/badge.svg)](https://github.com/Eslammohammedd/Codveda-Technologies-Data-Science-Intern/actions/workflows/ci.yml)

> **Complete End-to-End Data Science Pipeline**

This repository contains all technical deliverables for the **Codveda Technology Data Science Internship**. The portfolio is structured into three levels of increasing complexity, covering the entire machine learning lifecycle—from web scraping and data cleaning to regression, classification, clustering, time-series forecasting, NLP, and neural networks.

---

## 📁 Project Structure

The repository has been organized cleanly to separate deliverables, documentation, test suites, and utility scripts:

```text
Codveda_Internship/
├── level_1_basic/
│   ├── 00_web_scraping.ipynb         # Task 1: Scrape book data from Books to Scrape
│   ├── 01_data_cleaning.ipynb        # Task 2: Clean and preprocess book data
│   ├── 02_eda.ipynb                  # Task 3: EDA & PDF Report Generation
│   └── data/, outputs/               # Dataset & EDA reports
│
├── level_2_intermediate/
│   ├── 01_regression_modeling.ipynb  # Task 1: House Price Regression Model
│   ├── 02_classification.ipynb       # Task 2: Iris Species Classification (Binary classification)
│   ├── 03_clustering.ipynb           # Task 3: Customer Churn Segmentation (K-Means)
│   └── data/, models/, outputs/      # Datasets, trained .pkl models & predictions
│
├── level_3_advanced/
│   ├── 01_time_series.ipynb          # Task 1: Airline Passenger Forecasting (ARIMA)
│   ├── 02_nlp_classification.ipynb   # Task 2: Sentiment Analysis (Logistic Reg & Naive Bayes)
│   ├── 03_neural_networks.ipynb      # Task 3: MNIST Digit Recognition (TensorFlow/Keras)
│   └── data/, models/, outputs/      # Datasets, advanced models & predictions
│
├── docs/                             # Project instructions and reference PDFs
│   ├── Data Science Task List (1).pdf
│   └── INTERNSHIP_INSTRUCTIONS_AR.md
│
├── scripts/                          # Utility & Notebook generation scripts
│   ├── generate_levels_2_3.py
│   ├── generate_unified_level1.py
│   └── ... (Other utility/notebook fixing scripts)
│
├── tests/                            # Automated test suite
│   └── test_project.py               # Unit tests verifying datasets and model artifacts
│
├── .gitignore                        # Git ignore file (excludes virtualenvs, checkpoints, cache)
├── requirements.txt                  # Dependency configurations
├── inference.py                      # Production-ready model deployment & prediction tool
└── README.md                         # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Eslammohammedd/Codveda-Technologies-Data-Science-Intern.git
   cd Codveda-Data-Science-Intern
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📈 Executive Summary of Tasks

| Level | Task / Notebook | Goal | Primary Tools | Status |
| :--- | :--- | :--- | :--- | :---: |
| **L1: Basic** | `00_web_scraping.ipynb` | Extract books data with multi-page handling | `BeautifulSoup`, `requests`, `pandas` |  Completed |
| | `01_data_cleaning.ipynb` | Standardize price, encode ratings & impute | `pandas`, `scikit-learn` |  Completed |
| | `02_eda.ipynb` | Analyze variables & export PDF report | `seaborn`, `matplotlib`, `PdfPages` |  Completed |
| **L2: Intermediate** | `01_regression_modeling.ipynb` | Predict house prices & compare ML regressors | `scikit-learn`, `RandomForestRegressor` |  Completed |
| | `02_classification.ipynb` | Classify Iris species & evaluate ROC curves | `scikit-learn`, `SVC`, `LogisticRegression` |  Completed |
| | `03_clustering.ipynb` | Customer segmentation via K-Means & PCA | `scikit-learn`, `KMeans`, `PCA` |  Completed |
| **L3: Advanced** | `01_time_series.ipynb` | Forecast airline passenger volume (ARIMA) | `statsmodels` (ARIMA), `seasonal_decompose` |  Completed |
| | `02_nlp_classification.ipynb` | Text sentiment classification & confusion matrix | `nltk`, `TfidfVectorizer`, `MultinomialNB` |  Completed |
| | `03_neural_networks.ipynb` | Deep learning classifier on MNIST digits | `TensorFlow`, `Keras` |  Completed |

---

## 🧪 Testing

Automated unit tests are written with `pytest` to verify dataset integrity and trained model exports. Run them using:
```bash
python -m pytest
```

---

## 🚀 Model Deployment & Inference

A production-ready inference script is provided to showcase real-time prediction using the saved SVM classifier:
```bash
python inference.py
```
**Output Example:**
```text
--- Iris Species Inference Tool ---
Input Features: [5.9, 2.7, 4.2, 1.3]
Predicted Species: versicolor
```

---

## 📤 Submission & Social Pointers

- **GitHub Repository**: https://github.com/Eslammohammedd/Codveda-Technologies-Data-Science-Intern

## 📄 License

MIT — see [LICENSE](LICENSE).
