import nbformat as nbf

def create_notebook(filename, cells_content):
    nb = nbf.v4.new_notebook()
    cells = []
    for c in cells_content:
        if c['type'] == 'markdown':
            cells.append(nbf.v4.new_markdown_cell(c['source']))
        else:
            cells.append(nbf.v4.new_code_cell(c['source']))
    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

nb_nlp = [
    {"type": "markdown", "source": "# 📝 Natural Language Processing (NLP) - Text Classification\n**Task:** Classify text data into categories (e.g., sentiment analysis).  \n**Requirements:** Preprocess text, use TF-IDF, train Logistic Regression / Naive Bayes, evaluate with precision/recall/f1.  \n**Tools:** `Python`, `nltk`, `scikit-learn`, `pandas`"},
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay\nimport matplotlib.pyplot as plt\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Load Data\ntry:\n    df = pd.read_csv('data/sentiment_reviews.csv')\n    print('✅ Data loaded successfully')\n    text_col = 'Text'\n    label_col = 'Sentiment'\nexcept:\n    # Fallback dummy data\n    df = pd.DataFrame({'Text': ['Great!', 'Terrible', 'Okay', 'Loved it', 'Hated it'], 'Sentiment': ['Pos', 'Neg', 'Neu', 'Pos', 'Neg']})\n    text_col, label_col = 'Text', 'Sentiment'\n\n# Clean data\ndf = df.dropna(subset=[text_col, label_col])\ndf[label_col] = df[label_col].str.strip()\ndisplay(df[[text_col, label_col]].head())"},
    {"type": "code", "source": "# Preprocessing: TF-IDF Vectorization\nvectorizer = TfidfVectorizer(stop_words='english', max_features=1000)\nX = vectorizer.fit_transform(df[text_col])\ny = df[label_col]\n\n# Split Data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nprint(f'✅ Data split: Train {X_train.shape}, Test {X_test.shape}')"},
    {"type": "code", "source": "# Train Models\nmodels = {\n    'Naive Bayes': MultinomialNB(),\n    'Logistic Regression': LogisticRegression(max_iter=1000)\n}\n\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    preds = model.predict(X_test)\n    print(f\"\\n--- {name} ---\")\n    print(f\"Accuracy: {accuracy_score(y_test, preds):.2f}\")\n    print(classification_report(y_test, preds))"},
    {"type": "code", "source": "# Plot Confusion Matrix for Best Model (Logistic Regression)\nbest_model = models['Logistic Regression']\npreds = best_model.predict(X_test)\n\ncm = confusion_matrix(y_test, preds, labels=best_model.classes_)\ndisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)\nfig, ax = plt.subplots(figsize=(8, 8))\ndisp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')\nplt.title('Confusion Matrix - Logistic Regression')\nplt.tight_layout()\nplt.show()"}
]

create_notebook('level_3_advanced/02_nlp_classification.ipynb', nb_nlp)
print("NLP notebook fixed.")
