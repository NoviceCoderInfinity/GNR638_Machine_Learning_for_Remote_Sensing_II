import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def mlp_classify(train_image_feats, train_labels, test_image_feats, val_image_feats, test_labels, val_labels):
    """
    Train a three-layer MLP classifier and predict categories for test and validation sets.

    Args:
        train_image_feats: Numpy array of training features.
        train_labels: List of training labels.
        test_image_feats: Numpy array of test features.
        val_image_feats: Numpy array of validation features.
        test_labels: List of test labels.
        val_labels: List of validation labels.

    Returns:
        Tuple of predicted labels for test and validation sets, and their respective accuracies.
    """
    
    # Standardize features
    scaler = StandardScaler()
    train_image_feats = scaler.fit_transform(train_image_feats)
    test_image_feats = scaler.transform(test_image_feats)
    val_image_feats = scaler.transform(val_image_feats)

    # Initialize MLP classifier with ReLU activation and early stopping
    mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', 
                        max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1, 
                        learning_rate_init=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Train the classifier
    mlp.fit(train_image_feats, train_labels)

    # Predict
    test_pred_labels = mlp.predict(test_image_feats)
    val_pred_labels = mlp.predict(val_image_feats)

    # Calculate accuracy
    test_accuracy = accuracy_score(test_labels, test_pred_labels)
    val_accuracy = accuracy_score(val_labels, val_pred_labels)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Plot confusion matrix for test set
    cm_test = confusion_matrix(test_labels, test_pred_labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test Set')
    plt.colorbar()
    plt.show()

    # Plot confusion matrix for validation set
    cm_val = confusion_matrix(val_labels, val_pred_labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Validation Set')
    plt.colorbar()
    plt.show()

    return test_pred_labels, val_pred_labels
