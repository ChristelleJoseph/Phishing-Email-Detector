from phising_detector_LSTM import PhishingEmailDetector
import pandas as pd

detector = PhishingEmailDetector()

phishing_email_df = pd.read_csv('data/Phishing_Email.csv')

phishing_email_df = phishing_email_df.drop(columns=[phishing_email_df.columns[0]])
phishing_email_df = detector.clean_data(phishing_email_df)


texts = phishing_email_df['Email Text'].values
labels = phishing_email_df['Email Type'].values


detector.fit_tokenizer(texts)
detector.train(texts, labels)


accuracy = detector.evaluate(texts, labels)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

detector.save_model('phishing_email_detector')
