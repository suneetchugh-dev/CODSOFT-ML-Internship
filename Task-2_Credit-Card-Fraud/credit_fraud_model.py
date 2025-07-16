import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

df = pd.read_csv('creditcard.csv')
df.dropna(inplace=True)

plt.figure(figsize=(6, 4))
df['Class'].value_counts().plot(kind='pie', autopct='%1.2f%%', labels=['Legit (0)', 'Fraud (1)'], colors=['lightgreen', 'tomato'])
plt.title('Distribution of Legit vs Fraud Transactions')
plt.ylabel('')
plt.show()

print("Class distribution:\n", df['Class'].value_counts())

X = df.drop(['Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

y_proba = rf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Credit Card Fraud Detection")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

importances = rf.feature_importances_
indices = np.argsort(importances)[-10:][::-1]
features = X.columns[indices]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=features, palette='viridis')
plt.title('Top 10 Important Features in Fraud Detection')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()
