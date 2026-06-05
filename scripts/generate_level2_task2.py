import nbformat as nbf
import os

# Rename the old clustering notebook so classification can be #2
old_cluster = 'level_2_intermediate/02_clustering.ipynb'
new_cluster = 'level_2_intermediate/03_clustering.ipynb'
if os.path.exists(old_cluster):
    os.rename(old_cluster, new_cluster)

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

# --- Notebook: Classification ---
nb_class = [
    {"type": "markdown", "source": "# 🌸 Classification with Logistic Regression\n**Task:** Build a decision tree classifier and logistic regression to predict a categorical outcome (e.g., species of flowers).  \n**Requirements:** Preprocess data, evaluate metrics (accuracy, precision, recall, ROC curve), and compare classifiers.  \n**Tools:** `pandas`, `scikit-learn`, `matplotlib`"},
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.metrics import classification_report, accuracy_score, RocCurveDisplay\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# We will use the Iris dataset from Level 1. \n# To plot a standard ROC curve easily, we will filter it to a Binary Classification problem (2 species).\ndata_path = '../level_1_basic/data/iris.csv'\ntry:\n    df = pd.read_csv(data_path)\n    df = df[df['species'] != 'setosa'] # Keep only versicolor and virginica for binary classification\n    print('✅ Data loaded successfully')\nexcept:\n    from sklearn.datasets import load_iris\n    data = load_iris()\n    df = pd.DataFrame(data.data, columns=data.feature_names)\n    df['species'] = data.target\n    df = df[df['species'] != 0]\n    print('⚠️ Used sklearn built-in Iris data')\n\ndisplay(df.head())"},
    {"type": "code", "source": "# Preprocessing: Scaling and Encoding\nX = df.drop(columns=['species'] if 'species' in df.columns else [df.columns[-1]])\ny = df['species'] if 'species' in df.columns else df.iloc[:, -1]\n\n# Encode target labels to 0 and 1\nle = LabelEncoder()\ny_encoded = le.fit_transform(y)\n\n# Scale features\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# Train-Test Split\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)\nprint(f'✅ Data split: Train {X_train.shape}, Test {X_test.shape}')"},
    {"type": "code", "source": "# Train Logistic Regression\nlog_reg = LogisticRegression()\nlog_reg.fit(X_train, y_train)\n\ny_pred_log = log_reg.predict(X_test)\nprint(\"--- Logistic Regression Performance ---\")\nprint(f\"Accuracy: {accuracy_score(y_test, y_pred_log):.2f}\")\nprint(classification_report(y_test, y_pred_log, target_names=le.classes_.astype(str)))"},
    {"type": "code", "source": "# Compare with Random Forest and SVM\nmodels = {\n    'Random Forest': RandomForestClassifier(random_state=42),\n    'SVM': SVC(probability=True, random_state=42)\n}\n\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    preds = model.predict(X_test)\n    print(f\"\\n--- {name} Performance ---\")\n    print(f\"Accuracy: {accuracy_score(y_test, preds):.2f}\")\n    print(classification_report(y_test, preds, target_names=le.classes_.astype(str)))"},
    {"type": "code", "source": "# ROC Curve Comparison\nfig, ax = plt.subplots(figsize=(8, 6))\n\nRocCurveDisplay.from_estimator(log_reg, X_test, y_test, ax=ax, name='Logistic Regression')\nfor name, model in models.items():\n    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)\n\nplt.plot([0, 1], [0, 1], color='navy', linestyle='--')\nplt.title('Receiver Operating Characteristic (ROC) Curve Comparison')\nplt.show()"}
]

create_notebook('level_2_intermediate/02_classification.ipynb', nb_class)

print("Task 2 Classification notebook added and old clustering notebook renamed.")
