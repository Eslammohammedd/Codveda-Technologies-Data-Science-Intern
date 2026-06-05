import nbformat as nbf
import os

# --- Notebook 1: Data Cleaning ---
nb1 = nbf.v4.new_notebook()

cells_nb1 = [
    {"type": "markdown", "source": "# 🧹 Data Cleaning & Preprocessing Pipeline\n**Task:** Clean the Iris dataset and prepare it for modeling.  \n**Requirements:** Handle missing values, detect outliers, encode categorical features, and standardize numerical features.  \n**Tools:** `pandas`, `scikit-learn`, `numpy`"},
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom scipy import stats\nimport os\n\n# Load Data\ndata_path = 'data/iris.csv'\ndf = pd.read_csv(data_path)\n\nprint('✅ Data loaded successfully')\nprint(f'📊 Dimensions: {df.shape}')\ndf.head()"},
    {"type": "code", "source": "# Basic Data Information\nprint('📋 Column Information and Data Types:')\ndf.info()\nprint('\\n🔍 Descriptive Statistics:')\ndisplay(df.describe())\nprint('\\n⚠️ Number of missing values per column:')\nprint(df.isnull().sum())"},
    {"type": "code", "source": "# Create a copy to work on\ndf_clean = df.copy()\n\n# Use Imputer for handling missing values (median for numerical, most frequent for categorical)\nnum_cols = df_clean.select_dtypes(include='number').columns\ncat_cols = df_clean.select_dtypes(include='object').columns\n\nnum_imputer = SimpleImputer(strategy='median')\ncat_imputer = SimpleImputer(strategy='most_frequent')\n\nif len(num_cols) > 0:\n    df_clean[num_cols] = num_imputer.fit_transform(df_clean[num_cols])\nif len(cat_cols) > 0:\n    df_clean[cat_cols] = cat_imputer.fit_transform(df_clean[cat_cols])\n\nprint('✅ Missing values handled successfully')"},
    {"type": "code", "source": "# Outlier detection and handling using the IQR Method\ndef cap_outliers(df, cols):\n    for col in cols:\n        Q1 = df[col].quantile(0.25)\n        Q3 = df[col].quantile(0.75)\n        IQR = Q3 - Q1\n        lower_bound = Q1 - 1.5 * IQR\n        upper_bound = Q3 + 1.5 * IQR\n        # Cap the outliers\n        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)\n    return df\n\nif len(num_cols) > 0:\n    df_clean = cap_outliers(df_clean, num_cols)\nprint('✅ Outliers capped using the IQR Method')"},
    {"type": "code", "source": "# Encode categorical variables\nif len(cat_cols) > 0:\n    col = cat_cols[0]\n    le = LabelEncoder()\n    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])\n    mapping = dict(zip(le.classes_, le.transform(le.classes_)))\n    print(f'🔤 Encoding used: {mapping}')\n    print(f'✅ Encoding completed successfully for column: {col}')\nelse:\n    print('No categorical columns to encode')"},
    {"type": "code", "source": "# Standardization for numerical variables\nscaler = StandardScaler()\nif len(num_cols) > 0:\n    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])\n\nprint('✅ StandardScaler applied to numerical data')\ndf_clean.head()"},
    {"type": "code", "source": "# Save the cleaned dataset\noutput_path = 'outputs/cleaned_iris.csv'\nos.makedirs('outputs', exist_ok=True)\ndf_clean.to_csv(output_path, index=False)\n\nprint(f'💾 Cleaned data saved at: {output_path}')\nprint('\\n✅ Final Verification:')\nprint(df_clean.isnull().sum().sum(), 'missing values remaining')\nprint('📦 Final Columns:', df_clean.columns.tolist())"}
]

nb1['cells'] = [nbf.v4.new_markdown_cell(c['source']) if c['type'] == 'markdown' else nbf.v4.new_code_cell(c['source']) for c in cells_nb1]

with open('level_1_basic/01_data_cleaning.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb1, f)


# --- Notebook 2: EDA ---
nb2 = nbf.v4.new_notebook()

cells_nb2 = [
    {"type": "markdown", "source": "# 📊 Exploratory Data Analysis (EDA)\n**Task:** Explore the cleaned Iris dataset and understand relationships between variables.  \n**Tools:** `pandas`, `matplotlib`, `seaborn`"},
    {"type": "code", "source": "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Set plotting style\nsns.set_theme(style='whitegrid', palette='muted')\n\n# Load the cleaned dataset\ndata_path = 'outputs/cleaned_iris.csv'\n\ntry:\n    df = pd.read_csv(data_path)\n    print('✅ Cleaned data loaded successfully')\nexcept FileNotFoundError:\n    print('⚠️ Cleaned data not found. Loading raw data as a fallback.')\n    df = pd.read_csv('data/iris.csv')\n\ndf.head()"},
    {"type": "markdown", "source": "## 1. Descriptive Statistics"},
    {"type": "code", "source": "print('📋 General Information:')\ndf.info()\nprint('\\n🔍 Descriptive Statistics for Numerical Variables:')\ndisplay(df.describe())"},
    {"type": "markdown", "source": "## 2. Univariate Analysis"},
    {"type": "code", "source": "# Plot distributions for all numerical variables\nnum_cols = df.select_dtypes(include='number').columns\n\nplt.figure(figsize=(12, 8))\nfor i, col in enumerate(num_cols, 1):\n    plt.subplot(2, 3, i)\n    sns.histplot(df[col], kde=True, bins=20)\n    plt.title(f'Distribution of {col}')\n    plt.xlabel('')\n    plt.ylabel('')\n\nplt.tight_layout()\nplt.show()"},
    {"type": "markdown", "source": "## 3. Bivariate Analysis\nComparing numerical variables against categorical variables (Species)."},
    {"type": "code", "source": "cat_col = 'species_encoded' if 'species_encoded' in df.columns else ('species' if 'species' in df.columns else None)\n\nif cat_col:\n    plt.figure(figsize=(12, 8))\n    for i, col in enumerate([c for c in num_cols if c != cat_col], 1):\n        plt.subplot(2, 2, i)\n        sns.boxplot(x=cat_col, y=col, data=df)\n        plt.title(f'{col} by {cat_col}')\n    plt.tight_layout()\n    plt.show()\nelse:\n    print('⚠️ No categorical column found for plotting')"},
    {"type": "markdown", "source": "## 4. Multivariate Analysis\nUsing Pairplot and Correlation Heatmap to discover relationships."},
    {"type": "code", "source": "# Correlation Matrix Heatmap\ncorr_matrix = df.select_dtypes(include='number').corr()\n\nplt.figure(figsize=(8, 6))\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)\nplt.title('Correlation Heatmap')\nplt.show()"},
    {"type": "code", "source": "# Pairplot for pairwise relationships\nif cat_col:\n    sns.pairplot(df, hue=cat_col, diag_kind='kde')\nelse:\n    sns.pairplot(df, diag_kind='kde')\nplt.suptitle('Pairwise Relationships (Pairplot)', y=1.02)\nplt.show()"},
    {"type": "markdown", "source": "## 💡 Key EDA Insights:\n1. **Correlation:** There are strong positive correlations between certain features (e.g., petal length and petal width).\n2. **Distributions:** Some variables exhibit normal distributions, while others have bimodal distributions, indicating the presence of distinct clusters.\n3. **Separability:** We can clearly observe that specific features (like Petal Length) are highly capable of separating the target classes."}
]

nb2['cells'] = [nbf.v4.new_markdown_cell(c['source']) if c['type'] == 'markdown' else nbf.v4.new_code_cell(c['source']) for c in cells_nb2]

with open('level_1_basic/02_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb2, f)
