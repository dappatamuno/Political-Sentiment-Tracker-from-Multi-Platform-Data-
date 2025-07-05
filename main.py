# main.py
import gradio as gr
import pandas as pd
import snscrape.modules.twitter as sntwitter
import praw
from newspaper import Article
import nltk
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import requests
from bs4 import BeautifulSoup
import datetime
import io
import seaborn as sns

nltk.download('punkt')

# Load Sentiment Model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Reddit API Setup (use your own credentials)
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="sentiment-tracker"
)

LABELS = ['negative', 'neutral', 'positive']

def preprocess(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.strip()

def get_sentiment(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    output = model(**encoded_input)
    scores = softmax(output.logits[0].detach().numpy())
    return LABELS[scores.argmax()], scores.tolist()

def scrape_twitter(keyword, limit=100):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} since:{(datetime.date.today() - datetime.timedelta(days=7)).isoformat()}').get_items()):
        if i >= limit:
            break
        tweets.append(tweet.content)
    return tweets

def scrape_reddit(keyword, limit=100):
    posts = []
    for submission in reddit.subreddit("all").search(keyword, limit=limit):
        posts.append(submission.title + ' ' + (submission.selftext or ''))
    return posts

def scrape_news(keyword, source="bbc"):
    headlines = []
    url = f"https://www.{source}.com/search?q={keyword}"
    headers = {"User-Agent": "Mozilla/5.0"}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    for link in soup.find_all('a', href=True):
        href = link['href']
        if source in href and '/news/' in href:
            try:
                article = Article(href)
                article.download()
                article.parse()
                headlines.append(article.title + ' ' + article.text)
            except:
                continue
        if len(headlines) >= 10:
            break
    return headlines

def analyze_sentiments(texts):
    data = []
    for text in texts:
        label, scores = get_sentiment(text)
        data.append({"text": text, "sentiment": label})
    df = pd.DataFrame(data)
    return df

def generate_visuals(df):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sns.countplot(x='sentiment', data=df, ax=axs[0])
    axs[0].set_title("Sentiment Distribution")

    wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['text']))
    axs[1].imshow(wordcloud, interpolation='bilinear')
    axs[1].axis('off')
    axs[1].set_title("Word Cloud")

    df['date'] = datetime.date.today()
    trend = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)
    trend.plot(ax=axs[2], title='Daily Sentiment Trend')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def run_app(keyword, platform):
    if not keyword:
        return "Please enter a keyword.", None
    if platform == "Twitter":
        texts = scrape_twitter(keyword)
    elif platform == "Reddit":
        texts = scrape_reddit(keyword)
    elif platform == "BBC News":
        texts = scrape_news(keyword, source="bbc")
    elif platform == "CNN News":
        texts = scrape_news(keyword, source="cnn")
    else:
        return "Invalid platform.", None

    df = analyze_sentiments(texts)
    chart = generate_visuals(df)
    return df[['text', 'sentiment']], chart

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🗳️ Political Sentiment Tracker")
    gr.Markdown("Track sentiment from Twitter, Reddit, CNN, and BBC on any political keyword.")

    keyword = gr.Textbox(label="Enter Keyword (e.g. Biden, Labour Party)")
    platform = gr.Radio(choices=["Twitter", "Reddit", "BBC News", "CNN News"], label="Select Platform")
    run_btn = gr.Button("Track Sentiment")

    output_df = gr.Dataframe(label="Sentiment Analysis Results")
    output_plot = gr.Image(label="Sentiment Visuals")

    run_btn.click(fn=run_app, inputs=[keyword, platform], outputs=[output_df, output_plot])

if __name__ == "__main__":
    demo.launch()
