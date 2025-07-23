# scripts/train_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import os
from utils import load_nsl_kdd, preprocess_data

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    data_path = os.path.join('data', 'KDDTrain+.txt')
    df = load_nsl_kdd(data_path)

    # Preprocess data
    X, y = preprocess_data(df)

    num_classes = len(np.unique(y))
    print(f"Detected num_classes: {num_classes}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model
    model = build_cnn_model((12, 12, 1), num_classes)

    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64
    )

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/cnn_nids_model.keras')
    print("Model saved to models/cnn_nids_model.keras")

if __name__ == "__main__":
    main()
