# SMS Spam Detection

This is our project for the NLP course (DS483).
We want to build models that decide if an SMS message is ham (normal) or spam (unwanted).

Dataset: SMS Spam Collection from UCI
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## About the data
- 5,574 messages: 4,827 ham and 747 spam.
- We found 403 repeated messages and deleted them. Now we have 5,171.
- We used 4,136 messages for training and 1,035 for testing.
- All our models use the same test messages.

## What we will do
1. Clean the text
2. TF-IDF and cosine similarity
3. Naive Bayes and Logistic Regression
4. LSTM
5. BERT

## Folders
- data: the dataset and the train and test files
- notebooks: our Jupyter notebook
- report: the report
- docs: our weekly plan

## How to run
1. Install Python.
2. Run: pip install pandas scikit-learn
3. Open notebooks/01_load_and_split.ipynb and click Run All.

## Team
| Student ID | Student Name |
|---|---|
| (S220008217) | Amjad Rubayhan |
| (S220045072) | Mohammed Baaqil |
| (S230008796) | (Abdulrahman Alrimi) |
| (S230013941) | (Abdullah Alshubayli) |
