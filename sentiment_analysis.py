import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

df = pd.read_csv("books_data.csv")

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Title"].apply(get_sentiment)

print(df[["Title", "Sentiment"]].head())

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

sentiment_counts = df["Sentiment"].value_counts()

plt.figure()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
plt.title("Sentiment Distribution of Book Titles")
plt.show()