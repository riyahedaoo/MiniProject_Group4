import pandas as pd
import numpy as np
from nltk.corpus import stopwords, words
import re

df1 = pd.read_csv("J:\Mini_project\Dataset\emails.csv")
df2 = pd.read_csv("J:\Mini_project\Dataset\\fraud_email_.csv")

df1.rename(columns={"text":"Text", "spam":"Labels"}, inplace=True)
df2.rename(columns={"Class":"Labels"}, inplace=True)

df = pd.concat([df1, df2], ignore_index=True)

total_spams = df[df["Labels"]==1].shape[0]
total_non_spams = df[df["Labels"]==0].shape[0]

#Convert all to lower alphabets
df['Text']  = df['Text'].str.lower()

#Remove punctuations
df['Text'] = df['Text'].str.replace('[^\w\s]','')

#Remove the stopwords
STOPWORDS = set(stopwords.words('english'))
def stopword(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["Text"] = df["Text"].apply(stopword)

#Remove emojis
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

df['Text'] = df['Text'].apply(remove_emoji)

#Remove urls
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

df['Text'] = df['Text'].apply(remove_urls)

#Remove non-english words
def remove_non_english_words(text):
    new_text=re.sub('[^a-zA-Z0-9]+',' ',text)
    return new_text

df['Text'] = df['Text'].apply(remove_non_english_words)  
df.set_index("Text", inplace=True)     
df.to_csv("processed_data.csv")