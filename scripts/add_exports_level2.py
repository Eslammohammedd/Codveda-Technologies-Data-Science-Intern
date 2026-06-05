import nbformat
import os

# Notebook 1: 01_regression_modeling.ipynb
nb_path_1 = 'level_2_intermediate/01_regression_modeling.ipynb'
nb1 = nbformat.read(nb_path_1, as_version=4)
export_code_1 = """
import joblib
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save best model
joblib.dump(best_model, 'models/best_regression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save results
results_df.to_csv('outputs/regression_results.csv', index=False)
print('✅ Saved models and outputs successfully!')
"""
nb1.cells.append(nbformat.v4.new_code_cell(export_code_1))
nbformat.write(nb1, nb_path_1)

# Notebook 2: 02_classification.ipynb
nb_path_2 = 'level_2_intermediate/02_classification.ipynb'
nb2 = nbformat.read(nb_path_2, as_version=4)
export_code_2 = """
import joblib
import os
import pandas as pd

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save logistic regression and scaler
joblib.dump(log_reg, 'models/logistic_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# Save test predictions
output_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_log})
output_df.to_csv('outputs/classification_predictions.csv', index=False)
print('✅ Saved models and outputs successfully!')
"""
nb2.cells.append(nbformat.v4.new_code_cell(export_code_2))
nbformat.write(nb2, nb_path_2)

# Notebook 3: 03_clustering.ipynb
nb_path_3 = 'level_2_intermediate/03_clustering.ipynb'
nb3 = nbformat.read(nb_path_3, as_version=4)
export_code_3 = """
import joblib
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save kmeans model and scaler
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(pca, 'models/pca_model.pkl')

# Save clustered data
df_num.to_csv('outputs/clustered_data.csv', index=False)
print('✅ Saved models and outputs successfully!')
"""
nb3.cells.append(nbformat.v4.new_code_cell(export_code_3))
nbformat.write(nb3, nb_path_3)
