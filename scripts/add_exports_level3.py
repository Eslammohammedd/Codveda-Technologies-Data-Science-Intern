import nbformat
import os

# Notebook 1: 01_time_series.ipynb
nb_path_1 = 'level_3_advanced/01_time_series.ipynb'
nb1 = nbformat.read(nb_path_1, as_version=4)
export_code_1 = """
import joblib
import os
import pandas as pd

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save ARIMA model
joblib.dump(model_fit, 'models/arima_model.pkl')

# Save predictions
output_df = pd.DataFrame({'Actual': test, 'Forecast': predictions})
output_df.to_csv('outputs/arima_forecast.csv')
print('✅ Saved models and outputs successfully!')
"""
nb1.cells.append(nbformat.v4.new_code_cell(export_code_1))
nbformat.write(nb1, nb_path_1)

# Notebook 2: 02_nlp_classification.ipynb
nb_path_2 = 'level_3_advanced/02_nlp_classification.ipynb'
nb2 = nbformat.read(nb_path_2, as_version=4)
export_code_2 = """
import joblib
import os
import pandas as pd

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save NLP models and vectorizer
joblib.dump(models['Logistic Regression'], 'models/logistic_nlp_model.pkl')
joblib.dump(models['Naive Bayes'], 'models/naive_bayes_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

# Save predictions
output_df = pd.DataFrame({'Text': df.loc[y_test.index, text_col], 'Actual': y_test, 'Predicted': preds})
output_df.to_csv('outputs/nlp_predictions.csv', index=False)
print('✅ Saved models and outputs successfully!')
"""
nb2.cells.append(nbformat.v4.new_code_cell(export_code_2))
nbformat.write(nb2, nb_path_2)

# Notebook 3: 03_neural_networks.ipynb (Just in case it ever runs)
nb_path_3 = 'level_3_advanced/03_neural_networks.ipynb'
try:
    nb3 = nbformat.read(nb_path_3, as_version=4)
    export_code_3 = """
import os
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
try:
    model.save('models/nn_model.h5')
    print('✅ Saved neural network model successfully!')
except:
    pass
"""
    nb3.cells.append(nbformat.v4.new_code_cell(export_code_3))
    nbformat.write(nb3, nb_path_3)
except:
    pass
