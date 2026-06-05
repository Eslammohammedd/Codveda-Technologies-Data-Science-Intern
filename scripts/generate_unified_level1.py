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

# --- Notebook 0: Web Scraping ---
nb0 = [
    {"type": "markdown", "source": "# 🕸️ Step 1: Data Collection and Web Scraping\n**Task:** Scrape book data from books.toscrape.com.  \n**Output:** Raw CSV file `data/scraped_books_raw.csv`"},
    {"type": "code", "source": "import requests\nfrom bs4 import BeautifulSoup\nimport pandas as pd\nimport time\nimport os\n\nprint('✅ Libraries imported successfully')"},
    {"type": "code", "source": "BASE_URL = 'http://books.toscrape.com/catalogue/page-{}.html'\nMAX_PAGES = 5  # Scrape 5 pages for a good sample size\n\nall_books = []\nprint(f'🔍 Starting web scraping across {MAX_PAGES} pages...')\n\nfor page in range(1, MAX_PAGES + 1):\n    url = BASE_URL.format(page)\n    response = requests.get(url)\n    if response.status_code == 200:\n        soup = BeautifulSoup(response.content, 'html.parser')\n        books = soup.find_all('article', class_='product_pod')\n        for book in books:\n            title = book.h3.a['title']\n            price = book.find('p', class_='price_color').text  # Keeping currency symbol for cleaning task later\n            rating = book.p['class'][1]  # E.g., 'Three', 'One'\n            stock = book.find('p', class_='instock availability').text.strip()\n            \n            all_books.append({\n                'Title': title,\n                'Raw_Price': price,\n                'Text_Rating': rating,\n                'Availability_Text': stock\n            })\n        print(f'✅ Page {page} scraped successfully.')\n    else:\n        print(f'⚠️ Failed to retrieve page {page}.')\n    time.sleep(1)\n\nprint(f'\\n🎉 Total books scraped: {len(all_books)}')"},
    {"type": "code", "source": "df_raw = pd.DataFrame(all_books)\ndisplay(df_raw.head())"},
    {"type": "code", "source": "os.makedirs('data', exist_ok=True)\noutput_file = 'data/scraped_books_raw.csv'\ndf_raw.to_csv(output_file, index=False)\nprint(f'💾 Raw data saved to: {output_file}')"}
]

# --- Notebook 1: Data Cleaning ---
nb1 = [
    {"type": "markdown", "source": "# 🧹 Step 2: Data Cleaning & Preprocessing\n**Task:** Clean the raw scraped books data.  \n**Input:** `data/scraped_books_raw.csv`  \n**Output:** `outputs/scraped_books_clean.csv`"},
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\nimport os\n\n# Load Raw Data\ndata_path = 'data/scraped_books_raw.csv'\ntry:\n    df = pd.read_csv(data_path)\n    print('✅ Raw data loaded successfully')\nexcept:\n    # Fallback dummy data if scraping wasn't run\n    df = pd.DataFrame({'Title': ['A', 'B'], 'Raw_Price': ['£10.00', '£20.50'], 'Text_Rating': ['One', 'Three'], 'Availability_Text': ['In stock', 'Out of stock']})\n    print('⚠️ Using dummy data')\n\ndisplay(df.head())"},
    {"type": "code", "source": "# 1. Clean Price (Remove currency symbol and convert to float)\ndf['Price'] = df['Raw_Price'].str.replace('£', '', regex=False).str.replace('Â', '', regex=False).astype(float)\nprint('✅ Price cleaned')"},
    {"type": "code", "source": "# 2. Convert Text Rating to Numeric\nrating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}\ndf['Rating'] = df['Text_Rating'].map(rating_map)\nprint('✅ Ratings converted to numeric')"},
    {"type": "code", "source": "# 3. Encode Availability (Binary Encoding)\ndf['In_Stock'] = df['Availability_Text'].apply(lambda x: 1 if 'In stock' in x else 0)\nprint('✅ Availability encoded')"},
    {"type": "code", "source": "# 4. Standardize Price for modeling purposes\nscaler = StandardScaler()\ndf['Price_Scaled'] = scaler.fit_transform(df[['Price']])\nprint('✅ Price standardized')"},
    {"type": "code", "source": "# Keep only clean columns\ndf_clean = df[['Title', 'Price', 'Price_Scaled', 'Rating', 'In_Stock']]\ndisplay(df_clean.head())"},
    {"type": "code", "source": "os.makedirs('outputs', exist_ok=True)\noutput_path = 'outputs/scraped_books_clean.csv'\ndf_clean.to_csv(output_path, index=False)\nprint(f'💾 Clean data saved to: {output_path}')"}
]

# --- Notebook 2: EDA ---
nb2 = [
    {"type": "markdown", "source": "# 📊 Step 3: Exploratory Data Analysis (EDA)\n**Task:** Analyze the clean book dataset to find insights.  \n**Input:** `outputs/scraped_books_clean.csv`"},
    {"type": "code", "source": "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\nsns.set_theme(style='whitegrid', palette='muted')\n\n# Load Clean Data\ndata_path = 'outputs/scraped_books_clean.csv'\ntry:\n    df = pd.read_csv(data_path)\n    print('✅ Clean data loaded successfully')\nexcept:\n    df = pd.DataFrame({'Title': ['A', 'B'], 'Price': [10.0, 20.0], 'Rating': [1, 5], 'In_Stock': [1, 1]})\n    print('⚠️ Using dummy data')\n\ndisplay(df.head())"},
    {"type": "code", "source": "print('📋 Descriptive Statistics:')\ndisplay(df.describe())"},
    {"type": "code", "source": "# 1. Univariate Analysis: Price Distribution\nplt.figure(figsize=(8, 5))\nsns.histplot(df['Price'], bins=15, kde=True)\nplt.title('Distribution of Book Prices')\nplt.xlabel('Price (£)')\nplt.show()"},
    {"type": "code", "source": "# 2. Univariate Analysis: Rating Count\nplt.figure(figsize=(8, 5))\nsns.countplot(x='Rating', data=df, palette='viridis')\nplt.title('Count of Books by Rating')\nplt.show()"},
    {"type": "code", "source": "# 3. Bivariate Analysis: Price vs Rating\nplt.figure(figsize=(8, 5))\nsns.boxplot(x='Rating', y='Price', data=df, palette='Set2')\nplt.title('Book Prices across Different Ratings')\nplt.show()"},
    {"type": "code", "source": "# 4. Multivariate Analysis: Correlation Heatmap\nplt.figure(figsize=(6, 5))\ncorr_matrix = df[['Price', 'Rating', 'In_Stock']].corr()\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)\nplt.title('Correlation Heatmap')\nplt.show()"},
    {"type": "markdown", "source": "## 💡 Key EDA Insights:\n1. **Price Distribution:** Shows the average price range of books in the store.\n2. **Ratings:** We can observe if higher-rated books tend to be more expensive (via the boxplot).\n3. **Correlations:** The heatmap reveals if there is any linear relationship between Price, Rating, and Stock Availability."}
]

create_notebook('level_1_basic/00_web_scraping.ipynb', nb0)
create_notebook('level_1_basic/01_data_cleaning.ipynb', nb1)
create_notebook('level_1_basic/02_eda.ipynb', nb2)

print("Unified Level 1 pipeline generated successfully.")
