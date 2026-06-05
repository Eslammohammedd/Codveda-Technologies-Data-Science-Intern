import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells_content = [
    {"type": "markdown", "source": "# 🕸️ Data Collection and Web Scraping\n**Task:** Scrape data from a target website using BeautifulSoup and requests.  \n**Requirements:** Inspect structure, extract data, handle pagination, and store as CSV/JSON.  \n**Tools:** `Python`, `BeautifulSoup`, `requests`, `pandas`"},
    {"type": "code", "source": "import requests\nfrom bs4 import BeautifulSoup\nimport pandas as pd\nimport time\nimport os\n\nprint('✅ Libraries imported successfully')"},
    {"type": "code", "source": "# Target Website Configuration (Using a public sandbox site for scraping)\nBASE_URL = 'http://books.toscrape.com/catalogue/page-{}.html'\nMAX_PAGES = 3  # Set to a small number for demonstration\n\nall_books = []\n\nprint(f'🔍 Starting web scraping across {MAX_PAGES} pages...')\n\nfor page in range(1, MAX_PAGES + 1):\n    url = BASE_URL.format(page)\n    response = requests.get(url)\n    \n    if response.status_code == 200:\n        soup = BeautifulSoup(response.content, 'html.parser')\n        books = soup.find_all('article', class_='product_pod')\n        \n        for book in books:\n            title = book.h3.a['title']\n            price = book.find('p', class_='price_color').text.replace('£', '')\n            stock = book.find('p', class_='instock availability').text.strip()\n            \n            all_books.append({\n                'Title': title,\n                'Price (£)': float(price),\n                'Availability': stock\n            })\n        print(f'✅ Page {page} scraped successfully.')\n    else:\n        print(f'⚠️ Failed to retrieve page {page}. Status Code: {response.status_code}')\n    \n    # Politeness delay to avoid overloading the server\n    time.sleep(1)\n\nprint(f'\\n🎉 Total books scraped: {len(all_books)}')"},
    {"type": "code", "source": "# Store scraped data into a DataFrame\ndf_scraped = pd.DataFrame(all_books)\n\ndisplay(df_scraped.head())"},
    {"type": "code", "source": "# Save the scraped data to a CSV file\nos.makedirs('data', exist_ok=True)\noutput_file = 'data/scraped_books.csv'\ndf_scraped.to_csv(output_file, index=False)\n\nprint(f'💾 Scraped data successfully saved to: {output_file}')\nprint(f'📊 Final Dataset Shape: {df_scraped.shape}')"}
]

cells = []
for c in cells_content:
    if c['type'] == 'markdown':
        cells.append(nbf.v4.new_markdown_cell(c['source']))
    else:
        cells.append(nbf.v4.new_code_cell(c['source']))

nb['cells'] = cells

with open('level_1_basic/00_web_scraping.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Web scraping notebook generated successfully.")
