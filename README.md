# Ham or Spam — AI Spam Detection System

COMP 472 (Artificial Intelligence) — Mini Project 2

### Created By:

- Mridul Mridul (40279215)
- Reema Aboudraz (40253549)

### What it is

An intelligent email/SMS filter that classifies messages as **spam** or
**ham** (not spam). It learns from the SMS Spam Collection dataset using
supervised machine learning, reports its own accuracy, and lets you type new
messages to classify them with a confidence score.

## What it does and how

1. Loads a dataset of labelled messages (`spam.csv`) with **pandas**.
2. Converts text into numbers using **TF-IDF** (`TfidfVectorizer`).
3. Splits the data **80% training / 20% testing**.
4. Trains a **Logistic Regression** classifier.
5. Reports **accuracy** and a **confusion matrix**.
6. Draws a **bar chart** of spam vs ham counts (see
   `class_distribution.png` -> Created on every run).
7. Runs an **interactive loop**: type a message, get a prediction + confidence;
   type `quit` to exit.

## Project files

| File               | Purpose                                           |
| ------------------ | ------------------------------------------------- |
| `spam.csv`         | The SMS Spam Collection dataset (label, message). |
| `spam_detector.py` | `SpamDetector` class — the full ML pipeline.      |
| `main.py`          | Program entry point + interactive loop.           |
| `requirements.txt` | Python libraries required to run the project.     |

## How to run

1. Make sure `spam.csv` is in this folder.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the program:
   ```bash
   python main.py
   ```

## Libraries used

pandas, numpy, scikit-learn, matplotlib, seaborn
