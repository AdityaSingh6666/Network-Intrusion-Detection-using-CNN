import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_nsl_kdd, preprocess_data

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def main():
    # Load test data
    test_data_path = os.path.join('data', 'KDDTest+.txt')
    df_test = load_nsl_kdd(test_data_path)
    X_test, y_test = preprocess_data(df_test)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    X_test = X_test[:, :144]                 
    X_test = X_test.reshape(-1, 12, 12, 1)

    # Load trained model
    model = load_model('models/cnn_nids_model.keras')

    print(f"X_test shape before evaluation: {X_test.shape}")
    print(f"y_test shape before evaluation: {y_test.shape}")

    # Evaluate on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_test

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:\n", cm)

    # Print classification report
    report = classification_report(y_true, y_pred_classes)
    print("Classification Report:\n", report)

    # Save results
    if not os.path.exists('results'):
        os.makedirs('results')
    np.savetxt('results/test_accuracy.txt', [accuracy])
    np.savetxt('results/confusion_matrix.txt', cm, fmt='%d')
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)

    # Optional: Plot and save confusion matrix
    class_names = ['Normal', 'Attack']  # Update if using more classes
    plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix.png')
    print("Evaluation completed and saved in the 'results' folder.")

if __name__ == "__main__":
    main()
