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

nb2_1 = [
    {"type": "markdown", "source": "# 📈 Predictive Modeling (Regression)\n**Task:** Build and evaluate regression models to predict house prices.  \n**Requirements:** Train Linear Regression, Decision Trees, Random Forest. Evaluate using MSE and R-squared.  \n**Tools:** `pandas`, `scikit-learn`, `matplotlib`"},
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_squared_error, r2_score\nfrom sklearn.preprocessing import StandardScaler\n\n# Load Data (Handling whitespace delimiter and no header)\ntry:\n    df = pd.read_csv('data/house_prices.csv', sep='\\\\s+', header=None)\n    print('✅ Data loaded successfully')\nexcept:\n    # Fallback to standard csv if formatting changes\n    df = pd.read_csv('data/house_prices.csv')\n\ndf.head()"},
    {"type": "code", "source": "# Preprocessing (Handling missing values)\ndf = df.select_dtypes(include='number').fillna(df.median(numeric_only=True))\n\n# Assuming the last column is the target (House Price/MEDV in standard datasets).\ntarget_col = df.columns[-1]\nprint(f'🎯 Target Column Identified: {target_col}')\n\nX = df.drop(columns=[target_col])\ny = df[target_col]\n\n# Standardize features\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# Split Data\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\nprint(f'✅ Data split into Train {X_train.shape} and Test {X_test.shape}')"},
    {"type": "code", "source": "# Model Training & Evaluation\nmodels = {\n    'Linear Regression': LinearRegression(),\n    'Decision Tree': DecisionTreeRegressor(random_state=42),\n    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)\n}\n\nresults = []\n\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    predictions = model.predict(X_test)\n    mse = mean_squared_error(y_test, predictions)\n    r2 = r2_score(y_test, predictions)\n    results.append({'Model': name, 'MSE': mse, 'R2 Score': r2})\n\nresults_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)\ndisplay(results_df)"},
    {"type": "code", "source": "# Visualize Best Model Predictions\nbest_model_name = results_df.iloc[0]['Model']\nbest_model = models[best_model_name]\npreds = best_model.predict(X_test)\n\nplt.figure(figsize=(8, 6))\nplt.scatter(y_test, preds, alpha=0.5, color='blue')\nplt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')\nplt.xlabel('Actual Values')\nplt.ylabel('Predicted Values')\nplt.title(f'{best_model_name}: Actual vs Predicted')\nplt.show()"}
]

create_notebook('level_2_intermediate/01_regression_modeling.ipynb', nb2_1)
print("Regression notebook fixed.")
