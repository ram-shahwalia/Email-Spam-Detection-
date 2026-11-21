
import os
import re
import email
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def read_emails(folder, label):
    emails = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, "rb") as file:
                msg = email.message_from_bytes(file.read())

            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            body += payload.decode(charset, errors="ignore")
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="ignore")

            body = re.sub(r"\s+", " ", body).strip()

            if body:
                emails.append({"text": body, "label": label})

        except Exception:
            continue

    return emails


print("Loading email dataset...")

ham_emails = read_emails("dataset/easy_ham_2", "Not Spam")
spam_emails = read_emails("dataset/spam_2", "Spam")

data = pd.DataFrame(ham_emails + spam_emails)

if data.empty:
    raise ValueError("No emails found. Please check dataset folders.")

print("\nDataset loaded successfully!")
print("Total emails:", len(data))
print(data["label"].value_counts())

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000
    )),
    ("classifier", MultinomialNB())
])

print("\nTraining model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "spam_model.pkl")

print("\nModel saved successfully as spam_model.pkl")
