# Email Spam Classifier

A beginner-friendly machine learning project that classifies an email message as `spam` or `ham` (not spam). It uses TF-IDF text features and compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM. The model with the highest spam-class F1 score on the held-out test split is saved.

## Project structure

```text
email-spam-classifier/
├── dataset/
│   └── emails.csv
├── models/                 # generated after training
├── app.py                 # Streamlit web app
├── train.py               # model training and evaluation
├── requirements.txt
└── README.md
```

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Train the models

```bash
python train.py
```

This creates `models/spam_classifier.pkl` and `models/metrics.json`.

## Run the web app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, paste an email message, and click **Classify email**.

## Dataset format

`dataset/emails.csv` must contain two columns:

- `label`: `spam` or `ham`
- `text`: email subject/body text

Replace the included small demonstration dataset with a larger, representative, correctly labeled email dataset before drawing conclusions about model quality. Keep a separate untouched test set for a realistic final evaluation. Do not include private email content without appropriate permission and data protection.

## Important limitations

The included dataset is a small, hand-written teaching sample, not a production dataset. Scores from it are only a demonstration and will not reliably predict performance on real inbox traffic. The classifier is not a complete phishing detector and can make false positives and false negatives. Never automatically delete or trust emails based only on this model.
