import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells_content = [
    {"type": "markdown", "source": "# 🧹 Data Cleaning & Preprocessing Pipeline\n**المهمة:** تنظيف مجموعة بيانات Iris وإعدادها للنمذجة  \n**المتطلبات:** معالجة القيم المفقودة، كشف الشذوذ، الترميز، التطبيع  \n**الأدوات:** `pandas`, `scikit-learn`, `numpy`"},
    {"type": "code", "source": "import pandas as pd\nimport numpy as np\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom scipy import stats\nimport os\n\n# تحميل البيانات\ndata_path = 'data/iris.csv'\ndf = pd.read_csv(data_path)\n\nprint('✅ تم تحميل البيانات بنجاح')\nprint(f'📊 الأبعاد: {df.shape}')\ndf.head()"},
    {"type": "code", "source": "# معلومات أساسية عن البيانات\nprint('📋 معلومات الأعمدة وأنواع البيانات:')\ndf.info()\nprint('\\n🔍 إحصائيات وصفية:')\ndf.describe()\nprint('\\n⚠️ عدد القيم المفقودة لكل عمود:')\nprint(df.isnull().sum())"},
    {"type": "code", "source": "# إنشاء نسخة للعمل عليها دون تعديل الأصلي\ndf_clean = df.copy()\n\n# استخدام Imputer للتعامل مع القيم المفقودة (متوسط للأعداد، الأكثر تكراراً للفئات)\nnum_cols = df_clean.select_dtypes(include='number').columns\ncat_cols = df_clean.select_dtypes(include='object').columns\n\nnum_imputer = SimpleImputer(strategy='median')\ncat_imputer = SimpleImputer(strategy='most_frequent')\n\ndf_clean[num_cols] = num_imputer.fit_transform(df_clean[num_cols])\ndf_clean[cat_cols] = cat_imputer.fit_transform(df_clean[cat_cols])\n\nprint('✅ تم معالجة القيم المفقودة بنجاح')"},
    {"type": "code", "source": "# استخدام طريقة IQR (الربيعات) لكشف الشذوذ\ndef cap_outliers(df, cols):\n    for col in cols:\n        Q1 = df[col].quantile(0.25)\n        Q3 = df[col].quantile(0.75)\n        IQR = Q3 - Q1\n        lower_bound = Q1 - 1.5 * IQR\n        upper_bound = Q3 + 1.5 * IQR\n        # استبدال القيم خارج الحدود بأقرب حد\n        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)\n    return df\n\ndf_clean = cap_outliers(df_clean, num_cols)\nprint('✅ تم قص/ضبط القيم الشاذة باستخدام IQR Method')"},
    {"type": "code", "source": "# تحويل المتغيرات الفئوية من نصي إلى رقمي\ncat_col = df_clean.select_dtypes(include='object').columns\nif len(cat_col) > 0:\n    col = cat_col[0]\n    le = LabelEncoder()\n    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])\n    mapping = dict(zip(le.classes_, le.transform(le.classes_)))\n    print(f'🔤 الترميز المستخدم: {mapping}')\n    print(f'✅ تم الترميز بنجاح للعمود {col}')\nelse:\n    print('لا يوجد أعمدة فئوية للترميز')"},
    {"type": "code", "source": "# توحيد المقياس (Standardization) للمتغيرات العددية فقط\nscaler = StandardScaler()\ndf_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])\n\nprint('✅ تم تطبيق StandardScaler على البيانات العددية')\ndf_clean.head()"},
    {"type": "code", "source": "# حفظ النتيجة\noutput_path = 'outputs/cleaned_iris.csv'\nos.makedirs('outputs', exist_ok=True)\ndf_clean.to_csv(output_path, index=False)\n\nprint(f'💾 تم حفظ البيانات النظيفة في: {output_path}')\nprint('\\n✅ التحقق النهائي:')\nprint(df_clean.isnull().sum().sum(), 'قيم مفقودة متبقية')\nprint('📦 الأعمدة النهائية:', df_clean.columns.tolist())"}
]

cells = []
for c in cells_content:
    if c['type'] == 'markdown':
        cells.append(nbf.v4.new_markdown_cell(c['source']))
    else:
        cells.append(nbf.v4.new_code_cell(c['source']))

nb['cells'] = cells

with open('level_1_basic/01_data_cleaning.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
