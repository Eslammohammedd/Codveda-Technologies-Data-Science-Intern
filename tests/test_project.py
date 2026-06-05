import pandas as pd
import pytest
import os

def test_data_loading():
    """
    Test if the Iris dataset exists and can be loaded.
    """
    path = 'level_2_intermediate/data/iris.csv'
    assert os.path.exists(path), f"Dataset not found at {path}"
    
    df = pd.read_csv(path)
    assert not df.empty, "Dataset is empty"
    assert 'species' in df.columns or 'Species' in df.columns, "Target column 'species' missing"

def test_model_artifact_exists():
    """
    Check if Level 2 models were exported correctly.
    """
    model_path = 'level_2_intermediate/models/classifier_svm.pkl'
    # This might fail if notebooks haven't been run yet, 
    # but serves as a placeholder for pipeline testing.
    if os.path.exists('level_2_intermediate/models'):
        assert os.path.exists(model_path), "SVM model artifact missing in models/ directory"

if __name__ == "__main__":
    # Allow running directly for quick check
    try:
        test_data_loading()
        print("✅ Data loading test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
