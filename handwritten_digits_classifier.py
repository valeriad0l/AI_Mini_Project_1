import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("""==================================================
Mini Project 1 - COMP 472 (AI)

Handwritten Digits Classifier
==================================================""")

# Loading and exploring dataset
digits = load_digits()
data = digits.data / 16.0  # Normalized pixel values of flattened array (1797 x 64)
labels = digits.target # Correct digits for each image
images = digits.images  # Original 8x8 images for plots and visuals

print("\nDataset summary:")
print(f"Image shape in pixels -> {images[0].shape}")
print(f"Total number of images in dataset -> {len(images)}")
print(f"All unique labels -> {np.unique(labels)}")

# Split data into training (80%) and test sets (20%)
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate performance of classifier
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix (see in Plots):")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Examples of predictions (with actual and predicted labels)
print("Printing 3 prediction samples (see in Plots):")
for i in range(3):
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()