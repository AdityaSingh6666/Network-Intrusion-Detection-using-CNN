# scripts/predict.py

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import load_nsl_kdd, preprocess_data


def predict(input_csv):
    # Load your trained model
    model_path = os.path.join('models', 'cnn_nids_model.keras')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")

    # Load and preprocess data
    df = load_nsl_kdd(input_csv)
    X, y = preprocess_data(df)

    try:
        X = X.reshape(-1, 12, 12, 1)  # Adjust shape to match the trained CNN
    except:
        print(f"Error reshaping input to (-1, 12, 12, 1). Check your input file format and columns.")
        return

    print(f"Input data shape for prediction: {X.shape}")

    # Make predictions
    predictions = model.predict(X)
    predicted_labels = np.argmax(predictions, axis=1)

    # Save predictions to CSV
    df['predicted_label'] = predicted_labels
    df['predicted_label'] = df['predicted_label'].map({0: 'normal', 1: 'attack'})

    output_path = 'results/predictions.csv'
    os.makedirs('results', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Display summary
    unique, counts = np.unique(predicted_labels, return_counts=True)
    result_summary = dict(zip(unique, counts))
    print("Prediction Summary:")
    for label, count in result_summary.items():
        label_name = 'normal' if label == 0 else 'attack'
        print(f"{label_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Predict network intrusion using trained CNN.")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV for prediction.')
    args = parser.parse_args()

    predict(args.input)


if __name__ == "__main__":
    main()
