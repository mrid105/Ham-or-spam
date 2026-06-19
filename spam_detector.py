"""Mridul Mridul, 40279215
Reema Aboudraz, 40253549
COMP-472 Summer 2026
Mini-Project 2 Submission: Spam/ Ham Email and Message Classifier"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB          # alternative classifier
from sklearn.metrics import accuracy_score, confusion_matrix


class SpamDetector:
    """An email spam filter built on TF-IDF features + a classifier."""

    def __init__(self, csv_path):
        self.csv_path = csv_path

        self.df = None

        self.vectorizer = TfidfVectorizer(stop_words="english")

        # we used class_weight="balanced" to correct the spam/ham imbalance since only ~13%
        # of messages are spam, so the model is more willing to flag spam
        # instead of leaning toward "ham" for everything.
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.accuracy = None
        self.conf_matrix = None

    def load_data(self):
        """Read spam.csv with pandas and clean it up."""
        try:
            df = pd.read_csv(self.csv_path, encoding="latin-1")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find the dataset '{self.csv_path}'. "
                f"Make sure spam.csv is in the same folder as this program."
            )

        df = df.iloc[:, :2]
        df.columns = ["label", "message"]

        df = df.dropna()
        df = df[df["label"].isin(["ham", "spam"])]
        df = df.reset_index(drop=True)

        self.df = df

        spam_count = int((df["label"] == "spam").sum())
        ham_count = int((df["label"] == "ham").sum())
        print(f"Loaded {len(df)} messages  ->  {ham_count} ham, {spam_count} spam.")
        return df

    def prepare_features(self, test_size=0.2, random_state=42):
        """Convert text to TF-IDF numbers and split 80% train / 20% test."""

        X = self.vectorizer.fit_transform(self.df["message"])
        y = self.df["label"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(
            f"Split data: {self.X_train.shape[0]} for training, "
            f"{self.X_test.shape[0]} for testing."
        )

    def train(self):
        """Teach the model using the training data (supervised learning)."""
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")


    def evaluate(self):
        """Measure accuracy and build a confusion matrix on the test set."""
        predictions = self.model.predict(self.X_test)

        self.accuracy = accuracy_score(self.y_test, predictions)
        self.conf_matrix = confusion_matrix(
            self.y_test, predictions, labels=["spam", "ham"]
        )

        spam_spam, spam_ham = self.conf_matrix[0]
        ham_spam, ham_ham = self.conf_matrix[1]

        print(f"\nAccuracy: {self.accuracy * 100:.1f}%")
        print("Confusion Matrix:")
        print("                    Predicted")
        print("                  Spam      Ham")
        print(f"Actual Spam   {spam_spam:>7}  {spam_ham:>7}")
        print(f"Actual Ham    {ham_spam:>7}  {ham_ham:>7}")

        return self.accuracy, self.conf_matrix

    def predict_message(self, text):
        """Predict whether ONE message is spam or ham.

        Returns the predicted label ('spam' or 'ham').
        """
        features = self.vectorizer.transform([text])
        label = self.model.predict(features)[0]

        # CONFIDENCE SCORE  --  to be implemented 

        return label

    def plot_distribution(self):
        """Draw/save a bar chart of how many spam vs ham messages exist."""

        # DATA VISUALIZATION  --  to be implemented 
        
        pass
