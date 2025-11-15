import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv(r"C:\FILES\projects\Customer_chatbot\data\tickets_700.csv")

# Combine issue + solution for better understanding
df["text"] = df["issue"].astype(str) + " " + df["solution"].astype(str)

X = df["text"]
y = df["department"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Improved TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Balancing + More iterations
clf = LogisticRegression(max_iter=400, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

# Report
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save
joblib.dump(clf, "department_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained & saved successfully")
