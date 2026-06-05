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
    # Make sure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

# --- Notebook: Neural Networks ---
nb_nn = [
    {"type": "markdown", "source": "# 🧠 Neural Networks with TensorFlow/Keras\n**Task:** Build and train a feed-forward neural network to classify images.  \n**Requirements:** Load MNIST, design architecture, train, evaluate loss/accuracy curves, and tune hyperparameters.  \n**Tools:** `Python`, `TensorFlow`, `Keras`, `matplotlib`"},
    {"type": "code", "source": "import tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, Flatten, Dropout\nfrom tensorflow.keras.optimizers import Adam\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport warnings\nwarnings.filterwarnings('ignore')\n\nprint(f'✅ TensorFlow Version: {tf.__version__}')"},
    {"type": "code", "source": "# 1. Load and Preprocess Dataset (MNIST Digits)\nmnist = tf.keras.datasets.mnist\n(x_train, y_train), (x_test, y_test) = mnist.load_data()\n\n# Normalize pixel values to be between 0 and 1\nx_train, x_test = x_train / 255.0, x_test / 255.0\n\nprint(f'✅ MNIST Data Loaded. Train shape: {x_train.shape}, Test shape: {x_test.shape}')\n\n# Visualize a sample\nplt.figure(figsize=(3,3))\nplt.imshow(x_train[0], cmap='gray')\nplt.title(f'Label: {y_train[0]}')\nplt.axis('off')\nplt.show()"},
    {"type": "code", "source": "# 2. Design Neural Network Architecture\ndef build_model(learning_rate=0.001):\n    model = Sequential([\n        Flatten(input_shape=(28, 28)),\n        Dense(128, activation='relu'),\n        Dropout(0.2), # Prevent overfitting\n        Dense(64, activation='relu'),\n        Dense(10, activation='softmax') # 10 classes for digits 0-9\n    ])\n    \n    optimizer = Adam(learning_rate=learning_rate)\n    model.compile(optimizer=optimizer,\n                  loss='sparse_categorical_crossentropy',\n                  metrics=['accuracy'])\n    return model\n\nmodel = build_model()\nmodel.summary()"},
    {"type": "code", "source": "# 3. Train the Model\nprint('\\n🚀 Starting Training...')\nhistory = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))\nprint('✅ Training Completed')"},
    {"type": "code", "source": "# 4. Evaluate with Accuracy and Loss Curves\nplt.figure(figsize=(12, 5))\n\n# Plot Accuracy\nplt.subplot(1, 2, 1)\nplt.plot(history.history['accuracy'], label='Train Accuracy')\nplt.plot(history.history['val_accuracy'], label='Validation Accuracy')\nplt.title('Model Accuracy')\nplt.ylabel('Accuracy')\nplt.xlabel('Epoch')\nplt.legend()\n\n# Plot Loss\nplt.subplot(1, 2, 2)\nplt.plot(history.history['loss'], label='Train Loss')\nplt.plot(history.history['val_loss'], label='Validation Loss')\nplt.title('Model Loss')\nplt.ylabel('Loss')\nplt.xlabel('Epoch')\nplt.legend()\n\nplt.tight_layout()\nplt.show()"},
    {"type": "code", "source": "# 5. Hyperparameter Tuning (Example: Changing learning rate and batch size)\nprint('🔄 Tuning Hyperparameters (LR=0.01, Batch Size=128)...')\ntuned_model = build_model(learning_rate=0.01)\ntuned_history = tuned_model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))\n\n# Compare final accuracies\nbase_acc = history.history['val_accuracy'][-1]\ntuned_acc = tuned_history.history['val_accuracy'][-1]\nprint(f'\\n📊 Baseline Val Accuracy: {base_acc:.4f}')\nprint(f'📊 Tuned Val Accuracy: {tuned_acc:.4f}')"}
]

create_notebook('level_3_advanced/03_neural_networks.ipynb', nb_nn)
print("Task 3 Neural Networks notebook added successfully.")
