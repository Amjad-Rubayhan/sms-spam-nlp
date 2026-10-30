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
- report: the report
- docs: our weekly plan

## Data files
| File | What it is |
|---|---|
| data/SMSSpamCollection | The original data |
| data/train_v2.csv, data/test_v2.csv | Train and test (we use these) |
| data/train_clean_v2.csv, data/test_clean_v2.csv | Text after cleaning |
| data/train_prep_v2.csv, data/test_prep_v2.csv | Text after cleaning, stopwords and stemming |
| data/train.csv, data/test.csv | Old split from week 1 (not used now) |
| data/train_clean.csv, data/test_clean.csv | Old cleaning (not used now) |

## How to run
1. Install Python.
2. Run: pip install pandas scikit-learn nltk
3. Open the notebooks in this order and click Run All in each one:
   - notebooks/01_load_and_split_v2.ipynb
   - notebooks/02_cleaning_v2.ipynb
   - notebooks/03_stopwords_stemming.ipynb

## Team
| Student ID | Student Name |
|---|---|
| S220008217 | Amjad Rubayhan |
| S220045072 | Mohammed Baaqil |
| S230008796 | Abdulrahman Alrimi |
| S230013941 | Abdullah Alshubayli |
