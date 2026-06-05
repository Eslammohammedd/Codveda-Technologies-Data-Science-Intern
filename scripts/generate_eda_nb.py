import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells_content = [
    {"type": "markdown", "source": "# 📊 Exploratory Data Analysis (EDA)\n**المهمة:** استكشاف بيانات Iris النظيفة وفهم العلاقات بين المتغيرات  \n**الأدوات:** `pandas`, `matplotlib`, `seaborn`"},
    {"type": "code", "source": "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# إعداد ستايل الرسوم البيانية\nsns.set_theme(style='whitegrid', palette='muted')\n\n# تحميل البيانات النظيفة\ndata_path = 'outputs/cleaned_iris.csv'\n\ntry:\n    df = pd.read_csv(data_path)\n    print('✅ تم تحميل البيانات النظيفة بنجاح')\nexcept FileNotFoundError:\n    print('⚠️ لم يتم العثور على البيانات النظيفة. نقوم بتحميل البيانات الأصلية كبديل مؤقت.')\n    df = pd.read_csv('data/iris.csv')\n\ndf.head()"},
    {"type": "markdown", "source": "## 1. الإحصائيات الوصفية (Descriptive Statistics)"},
    {"type": "code", "source": "print('📋 معلومات عامة:')\ndf.info()\nprint('\\n🔍 إحصائيات وصفية للمتغيرات العددية:')\ndisplay(df.describe())"},
    {"type": "markdown", "source": "## 2. التحليل أحادي المتغير (Univariate Analysis)"},
    {"type": "code", "source": "# رسم التوزيعات لكل المتغيرات العددية\nnum_cols = df.select_dtypes(include='number').columns\n\nplt.figure(figsize=(12, 8))\nfor i, col in enumerate(num_cols, 1):\n    plt.subplot(2, 3, i)\n    sns.histplot(df[col], kde=True, bins=20)\n    plt.title(f'توزيع {col}')\n    plt.xlabel('')\n    plt.ylabel('')\n\nplt.tight_layout()\nplt.show()"},
    {"type": "markdown", "source": "## 3. التحليل ثنائي المتغيرات (Bivariate Analysis)\nمقارنة المتغيرات العددية مع الفئات (الأنواع)."},
    {"type": "code", "source": "cat_col = 'species_encoded' if 'species_encoded' in df.columns else ('species' if 'species' in df.columns else None)\n\nif cat_col:\n    plt.figure(figsize=(12, 8))\n    for i, col in enumerate([c for c in num_cols if c != cat_col], 1):\n        plt.subplot(2, 2, i)\n        sns.boxplot(x=cat_col, y=col, data=df)\n        plt.title(f'{col} حسب النوع')\n    plt.tight_layout()\n    plt.show()\nelse:\n    print('⚠️ لا يوجد عمود فئات للرسم')"},
    {"type": "markdown", "source": "## 4. تحليل العلاقات المتعددة (Multivariate Analysis)\nاستخدام Pairplot و Heatmap لاكتشاف الارتباطات."},
    {"type": "code", "source": "# مصفوفة الارتباط (Correlation Matrix)\ncorr_matrix = df.select_dtypes(include='number').corr()\n\nplt.figure(figsize=(8, 6))\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)\nplt.title('مصفوفة الارتباط (Correlation Heatmap)')\nplt.show()"},
    {"type": "code", "source": "# رسم العلاقات الثنائية (Pairplot)\nif cat_col:\n    sns.pairplot(df, hue=cat_col, diag_kind='kde')\nelse:\n    sns.pairplot(df, diag_kind='kde')\nplt.suptitle('العلاقات بين المتغيرات (Pairplot)', y=1.02)\nplt.show()"},
    {"type": "markdown", "source": "## 💡 أهم الاستنتاجات من الـ EDA:\n1. **الارتباط (Correlation):** هناك ارتباط إيجابي قوي بين ... (مثلاً طول البتلة وعرضها).\n2. **التوزيعات (Distributions):** بعض المتغيرات تظهر توزيعات طبيعية، والبعض الآخر يمتلك قمم متعددة (Bimodal) مما يشير إلى وجود مجموعات مختلفة.\n3. **القدرة على الفصل:** يمكننا ملاحظة أن بعض الخصائص (مثل Petal Length) قادرة على فصل فئات الـ Species بشكل واضح جداً."}
]

cells = []
for c in cells_content:
    if c['type'] == 'markdown':
        cells.append(nbf.v4.new_markdown_cell(c['source']))
    else:
        cells.append(nbf.v4.new_code_cell(c['source']))

nb['cells'] = cells

with open('level_1_basic/02_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
