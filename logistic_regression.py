#Multi-class classification
# baseline logistic regression - KNN - SVM - Neural Network

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

digits = datasets.load_digits()
X = digits.data
y = digits.target

# Chia train/test 1 lần duy nhất, dùng chung cho tất cả model để so sánh công bằng
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42),
}

results = []

def run_model(name, model):
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {pre:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

    results.append({"Model": name, "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1})

for name, model in models.items():
    run_model(name, model)

# Bảng so sánh tổng hợp
df_results = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n=== So sánh tổng hợp ===")
print(df_results.to_string(index=False))

df_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(
    kind='bar', figsize=(9, 5)
)
plt.title("So sánh các mô hình trên tập Digits")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()