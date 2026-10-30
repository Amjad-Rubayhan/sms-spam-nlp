# SMS Spam Detection Report

Version 1.1

## 1. Introduction

Every day, people receive many SMS messages.
Most of them are normal, like messages from friends and family.
But some are unwanted, like ads, fake prizes, and
messages that try to trick people.
Normal messages are called "ham".
Unwanted messages are called "spam".

Imagine a telecommunications company that gets a huge
number of SMS messages every day.
The company wants a system that can read a message
and decide if it is ham or spam, without a person
checking it.

In this project, we work on this problem using NLP
(Natural Language Processing).
We use the SMS Spam Collection dataset from UCI.
It has 5,574 messages, and each one is labeled
as ham or spam.

Our plan has these steps:
- Look at the data and clean the text.
- Turn the messages into numbers and find similar messages.
- Build simple models to find spam.
- Build more advanced models.
- Compare all the models and choose the best one.

This version of the report covers the first part only:
how we looked at the data, fixed problems in it,
and prepared it for the models.
We will add the other parts when we finish them.

So far, we did these things:
- We looked at the raw file and understood how it is written.
- We loaded the data and found a problem: 5,572 rows instead of 5,574.
- We found the reason (a quote mark) and fixed it.
- We checked the data: no empty values, and the labels are correct.
- We found that most messages are ham (87%) and only 13% are spam.
- We found 403 repeated messages and deleted them.
- We split the data into training (4,136) and testing (1,035).
- We saved the two parts, so everyone in the team uses the same files.
- Later, we checked the repeated messages again in a better way.
  We found 43 more copies and made version 2 of the split.

Files we have now:
- `data/SMSSpamCollection`: the original data file.
- `data/train.csv` and `data/test.csv`: the split from week 1 (old, not used now).
- `data/train_v2.csv`: the training messages, version 2 (4,102).
- `data/test_v2.csv`: the testing messages, version 2 (1,026).
- `notebooks/01_load_and_split.ipynb`: our notebook from week 1.
- `notebooks/01_load_and_split_v2.ipynb`: our notebook for version 2.

## 2. Data and Setup

### The data

The file has 5,574 lines.
Each line has a label, then a tab, then the message.
There are two labels:
- ham: 4,827 messages
- spam: 747 messages

Most messages are ham (about 87%).
No message is empty, and every label is "ham" or "spam".

### A problem when loading

When we first read the file, we got 5,572 rows.
Two rows were missing.
Three lines were joined into one row.
The reason was a quote mark (") at the start of a message.
We told pandas to ignore quote marks.
After that we got all 5,574 rows.

### Repeated messages (week 1)

We found 403 messages that were exactly the same
as another message.
The same message never had two different labels.
We deleted them. We had 5,171 messages.

### Splitting (week 1)

We split the data into two parts:
- Train: 4,136 messages (80%)
- Test: 1,035 messages (20%)

We kept the same share of spam in both parts.
We used a fixed random seed (42), so we get the same split every time.

### A better check for copies (version 2)

Later we found more copies.
They were the same message with small changes,
like a capital letter, a space, or a dot at the end.
For example:
- "... within 28 days"
- "... within 28 days."

To find them, we made a "key" for each message:
the message in small letters, with only letters and numbers.
If two messages have the same key, they are the same message.

Two messages had no letters or numbers (":)" and ":-) :-)").
For them, the key is the message itself,
because they are not copies of each other.

With the key, we found 446 copies (43 more than before).
The same key never had two different labels.
We removed the copies.
Now we have 5,128 messages:
- ham: 4,502
- spam: 626

### Splitting (version 2)

We split the data again with the same settings as before.
- Train: 4,102 messages (3,601 ham and 501 spam)
- Test: 1,026 messages (901 ham and 125 spam)

We checked the keys: no message is in both train and test.
We saved the two parts in `data/train_v2.csv` and `data/test_v2.csv`.
From now on, all our work and all our models use these two files.
