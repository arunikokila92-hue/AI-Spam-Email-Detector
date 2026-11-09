# Additional Email Dataset

This project can train on the original `spam.csv` SMS dataset plus an additional real-email corpus.

## Download and prepare the email corpus

From the repository root, run:

```powershell
python dataset/download_spamassassin.py
```

The script downloads the public Apache SpamAssassin corpus archives, extracts message subject/body text, removes exact normalized duplicates, and creates:

`dataset/spamassassin_emails.csv`

The output schema is:

- `label`: `spam` or `ham`
- `text`: email subject and body text

The selected official corpus parts describe 6,047 messages in total: 2,500 easy ham, 250 hard ham, 1,400 easy ham 2, 500 spam, and 1,397 spam 2. The exact number written to CSV can be lower after empty-message filtering and deduplication.

Official source: https://spamassassin.apache.org/old/publiccorpus/  
Corpus description: https://spamassassin.apache.org/old/publiccorpus/readme.html

## Retrain the model

After the CSV has been created, run:

```powershell
python train_model.py
```

The training script automatically combines `spam.csv` with `dataset/spamassassin_emails.csv` when the latter exists, removes duplicate messages, performs a stratified train/test split, prints accuracy/precision/recall/F1 metrics, and saves the refreshed `model.pkl`.

## Important notes

- The original `spam.csv` contains SMS messages. The additional corpus contains emails. Combining them makes this a mixed text-message spam classifier. For an email-only evaluation, evaluate separately on email messages.
- The SpamAssassin corpus README says copyright remains with the original message senders. The generated email CSV is therefore kept local and excluded from Git by `.gitignore`; review permissions before redistributing it.
- Never upload private emails or personal data to a public repository.
- The SpamAssassin README explicitly warns not to send corpus emails into a live email system.
