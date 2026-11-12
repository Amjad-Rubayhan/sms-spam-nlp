# SMS Spam Detection

This is our project for the NLP course (DS483).
We want to build models that decide if an SMS message is ham (normal) or spam (unwanted).

Dataset: SMS Spam Collection from UCI
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## About the data
- 5,574 messages: 4,827 ham and 747 spam.
- Week 1: we deleted 403 messages that were exactly the same. We had 5,171.
- Version 2: we found 43 more copies with small changes
  (capital letters, spaces or dots). We deleted them too.
- Now we have 5,128 messages: 4,502 ham and 626 spam.
- Train: 4,102 messages. Test: 1,026 messages.
- All our models use the same test messages (data/test_v2.csv).

## What we will do
1. Clean the text
2. TF-IDF and cosine similarity
3. Naive Bayes and Logistic Regression
4. LSTM
5. BERT

## Folders
- data: the dataset and the train and test files
- notebooks: our Jupyter notebooks
- report: the report and the charts
- docs: our weekly plan

## Data files
| File | What it is |
|---|---|
| data/SMSSpamCollection | The original data |
| data/train_v2.csv, data/test_v2.csv | Train and test (we use these) |
| data/train_clean_v2.csv, data/test_clean_v2.csv | Text after cleaning |
| data/train_prep_v2.csv, data/test_prep_v2.csv | Text after cleaning, stopwords and stemming |
| data/train_v1.csv, data/test_v1.csv | Old split from week 1 (not used now) |
| data/pred_nb.csv | What the Naive Bayes model said about each test message |
| data/pred_lr.csv | What the Logistic Regression model said about each test message |
| data/pred_lstm.csv | What the LSTM model said about each test message |
| data/lstm_history.csv | Loss and accuracy of the LSTM after each epoch |
| data/lstm_variants.csv | Results of the three LSTM types on the validation data |
| data/model_comparison.csv | Accuracy, precision, recall and F1 of all models |
| data/all_predictions.csv | The answers of all models for each test message |

## How to run
1. Install Python.
2. Run: pip install pandas scikit-learn nltk matplotlib tensorflow
3. Open the notebooks in this order and click Run All in each one:
   - notebooks/01_load_and_split_v2.ipynb
   - notebooks/02_cleaning_v2.ipynb
   - notebooks/03_stopwords_stemming.ipynb
   - notebooks/04_charts.ipynb
   - notebooks/05_tfidf_similarity.ipynb
   - notebooks/06_naive_bayes.ipynb
   - notebooks/07_logistic_regression.ipynb
   - notebooks/08_compare_errors.ipynb
   - notebooks/09_lstm.ipynb
   - notebooks/10_lstm_variants.ipynb
   - notebooks/11_lstm_errors.ipynb
   - notebooks/12_lstm_charts.ipynb

Notes:
- We ran notebooks 09 and 10 on Google Colab. If you run them again,
  the numbers can change a little, because the LSTM starts with random numbers.
- Notebooks 11 and 12 use the saved files in the data folder.

## Results so far
All models use the same test file (data/test_v2.csv): 1,026 messages.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 98.9 | 100.0 | 91.2 | 95.4 |
| Logistic Regression | 99.1 | 97.5 | 95.2 | 96.4 |
| LSTM | 98.9 | 95.2 | 96.0 | 95.6 |

Mistakes on the test data:

| Model | Ham called spam | Spam missed |
|---|---|---|
| Naive Bayes | 0 | 11 |
| Logistic Regression | 3 | 6 |
| LSTM | 6 | 5 |

We also tried three LSTM types. We compared them on the validation data
(a part of the train data), not on the test data.

| LSTM type | F1 on validation |
|---|---|
| Simple | 90.2 |
| Bidirectional | 93.3 |
| Stacked | 92.1 |

BERT is not done yet.

## Team
| Student ID | Student Name |
|---|---|
| S220008217 | Amjad Rubayhan |
| S220045072 | Mohammed Baaqil |
| S230008796 | Abdulrahman Alrimi |
| S230013941 | Abdullah Alshubayli |
