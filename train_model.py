import pandas as pd
data = pd.read_csv("spam.csv")
print(data.head())
print(data.columns)

print(data.columns)
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("spam.csv")

# Remove empty rows
data = data[['label', 'text']]

X = data["text"]
y = data["label"]
print("Columns:")
print(data.columns)
print("\nFirst 5 Rows:")
print(data.head())

print("\nLabel Count:")
print(data['label'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully")