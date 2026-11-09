# Dataset

This folder stores additional labeled spam/ham text datasets used to extend the training corpus.

## Required CSV schema

All additional datasets should be normalized to two columns:

- `label`: `spam` or `ham`
- `text`: the message body

Do not upload private or personal emails. Remove email addresses, phone numbers, names, account details, and other personal information before sharing data publicly.

## Important scope note

The current `spam.csv` is based on SMS messages. SMS and email datasets can be combined for a general text-message spam classifier, but the project should be described accurately. For an email-specific classifier, use a labeled email corpus such as SpamAssassin and evaluate on a separate email test set.
