import nbformat as nbf
import os

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

# --- Notebook 1: Data Cleaning (Updated for saving models & clean filename) ---
nb1 = [
    {"type": "markdown", "source": "# 🧹 Step 2: Data Cleaning & Preprocessing\n**Task:** Clean the raw scraped books data.  \n**Input:** `data/scraped_books_raw.csv`  \n**Output:** `outputs/cleaned_books.csv`"},
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nimport joblib\nimport os\n\n# Ensure output directories exist\nos.makedirs('outputs', exist_ok=True)\nos.makedirs('models', exist_ok=True)\n\n# Load Raw Data\ndata_path = 'data/scraped_books_raw.csv'\ntry:\n    df = pd.read_csv(data_path)\n    print('✅ Raw data loaded successfully')\nexcept:\n    # Fallback dummy data if scraping wasn't run\n    df = pd.DataFrame({'Title': ['A', 'B'], 'Raw_Price': ['£10.00', '£20.50'], 'Text_Rating': ['One', 'Three'], 'Availability_Text': ['In stock', 'Out of stock']})\n    print('⚠️ Using dummy data')\n\ndisplay(df.head())"},
    {"type": "code", "source": "# 1. Clean Price (Remove currency symbol and convert to float)\ndf['Price'] = df['Raw_Price'].str.replace('£', '', regex=False).str.replace('Â', '', regex=False).astype(float)\nprint('✅ Price cleaned')"},
    {"type": "code", "source": "# 2. Encode Text Rating (Using LabelEncoder to save it as a model)\nencoder = LabelEncoder()\ndf['Rating'] = encoder.fit_transform(df['Text_Rating']) + 1 # +1 to make it 1-5 instead of 0-4\njoblib.dump(encoder, 'models/encoder.pkl')\nprint('✅ Ratings encoded and encoder saved to models/encoder.pkl')"},
    {"type": "code", "source": "# 3. Encode Availability (Binary Encoding)\ndf['In_Stock'] = df['Availability_Text'].apply(lambda x: 1 if 'In stock' in x else 0)\nprint('✅ Availability encoded')"},
    {"type": "code", "source": "# 4. Standardize Price for modeling purposes\nscaler = StandardScaler()\ndf['Price_Scaled'] = scaler.fit_transform(df[['Price']])\njoblib.dump(scaler, 'models/scaler.pkl')\nprint('✅ Price standardized and scaler saved to models/scaler.pkl')"},
    {"type": "code", "source": "# Keep only clean columns\ndf_clean = df[['Title', 'Price', 'Price_Scaled', 'Rating', 'In_Stock']]\ndisplay(df_clean.head())"},
    {"type": "code", "source": "output_path = 'outputs/cleaned_books.csv'\ndf_clean.to_csv(output_path, index=False)\nprint(f'💾 Clean data saved to: {output_path}')"}
]

# --- Notebook 2: EDA (Updated for PDF Report generation) ---
nb2 = [
    {"type": "markdown", "source": "# 📊 Step 3: Exploratory Data Analysis (EDA)\n**Task:** Analyze the clean book dataset to find insights and generate a PDF report.  \n**Input:** `outputs/cleaned_books.csv`\n**Output:** `outputs/eda_report.pdf`"},
    {"type": "code", "source": "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom matplotlib.backends.backend_pdf import PdfPages\nimport warnings\nwarnings.filterwarnings('ignore')\n\nsns.set_theme(style='whitegrid', palette='muted')\n\n# Prepare PDF Report object\npdf_pages = PdfPages('outputs/eda_report.pdf')\nprint('✅ PDF Report initialized')\n\n# Load Clean Data\ndata_path = 'outputs/cleaned_books.csv'\ntry:\n    df = pd.read_csv(data_path)\n    print('✅ Clean data loaded successfully')\nexcept:\n    df = pd.DataFrame({'Title': ['A', 'B'], 'Price': [10.0, 20.0], 'Rating': [1, 5], 'In_Stock': [1, 1]})\n    print('⚠️ Using dummy data')\n\ndisplay(df.head())"},
    {"type": "code", "source": "print('📋 Descriptive Statistics:')\ndisplay(df.describe())"},
    {"type": "code", "source": "# 1. Univariate Analysis: Price Distribution\nfig1 = plt.figure(figsize=(8, 5))\nsns.histplot(df['Price'], bins=15, kde=True)\nplt.title('Distribution of Book Prices')\nplt.xlabel('Price (£)')\npdf_pages.savefig(fig1) # Save to PDF\nplt.show()"},
    {"type": "code", "source": "# 2. Univariate Analysis: Rating Count\nfig2 = plt.figure(figsize=(8, 5))\nsns.countplot(x='Rating', data=df, palette='viridis')\nplt.title('Count of Books by Rating')\npdf_pages.savefig(fig2) # Save to PDF\nplt.show()"},
    {"type": "code", "source": "# 3. Bivariate Analysis: Price vs Rating\nfig3 = plt.figure(figsize=(8, 5))\nsns.boxplot(x='Rating', y='Price', data=df, palette='Set2')\nplt.title('Book Prices across Different Ratings')\npdf_pages.savefig(fig3) # Save to PDF\nplt.show()"},
    {"type": "code", "source": "# 4. Multivariate Analysis: Correlation Heatmap\nfig4 = plt.figure(figsize=(6, 5))\ncorr_matrix = df[['Price', 'Rating', 'In_Stock']].corr()\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)\nplt.title('Correlation Heatmap')\npdf_pages.savefig(fig4) # Save to PDF\nplt.show()"},
    {"type": "code", "source": "# Close the PDF file to finalize saving\npdf_pages.close()\nprint('💾 EDA Report successfully saved to: outputs/eda_report.pdf')"},
    {"type": "markdown", "source": "## 💡 Key EDA Insights:\n1. **Price Distribution:** Shows the average price range of books in the store.\n2. **Ratings:** We can observe if higher-rated books tend to be more expensive (via the boxplot).\n3. **Correlations:** The heatmap reveals if there is any linear relationship between Price, Rating, and Stock Availability."}
]

create_notebook('level_1_basic/01_data_cleaning.ipynb', nb1)
create_notebook('level_1_basic/02_eda.ipynb', nb2)

print("Updated notebooks for exact folder structure successfully.")
