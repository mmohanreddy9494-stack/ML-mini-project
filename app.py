import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

digits = datasets.load_digits()
X = digits.data
y = digits.target

print("Dataset Loaded!")
print("Total Samples:", len(X))

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i].reshape(8, 8), cmap='gray')
    plt.title(f"Digit: {y[i]}")
    plt.axis('off')
plt.suptitle("Sample Digits from Dataset")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

model = SVC(kernel='rbf', gamma='scale')
model.fit(X_train, y_train)
print("\nModel Training Completed!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

difficulty_scores = np.std(X_test, axis=1)
threshold = np.mean(difficulty_scores)

easy_count = 0
hard_count = 0

for score in difficulty_scores:
    if score < threshold:
        easy_count += 1
    else:
        hard_count += 1

print("\nDifficulty Analysis:")
print("Easy Digits:", easy_count)
print("Hard Digits:", hard_count)

plt.figure()
plt.hist(difficulty_scores, bins=25)
plt.axvline(threshold, linestyle='dashed')
plt.title("Difficulty Score Distribution")
plt.xlabel("Difficulty Score")
plt.ylabel("Frequency")
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

digit_accuracy = []
for digit in range(10):
    idx = (y_test == digit)
    acc = accuracy_score(y_test[idx], y_pred[idx])
    digit_accuracy.append(acc)

plt.figure()
plt.bar(range(10), digit_accuracy)
plt.title("Accuracy per Digit")
plt.xlabel("Digit")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"T:{y_test[i]} P:{y_pred[i]}")
    plt.axis('off')
plt.suptitle("Sample Predictions (True vs Predicted)")
plt.show()

hardest_indices = np.argsort(difficulty_scores)[-10:]
plt.figure(figsize=(10, 5))
for i, idx in enumerate(hardest_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    plt.title(f"Hard ({y_test[idx]})")
    plt.axis('off')
plt.suptitle("Most Difficult Digits")
plt.show()

print("\nProject Execution Completed Successfully!")

