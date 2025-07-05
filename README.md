
# 🗳️ Political Sentiment Tracker

An AI-powered web app that collects, analyzes, and visualizes real-time political sentiment from multiple platforms (Twitter, Reddit, CNN, and BBC). Built using free tools end-to-end and deployable on Gradio.

## 🔧 Features
- Search political keywords across platforms
- Real-time scraping (no Twitter API needed)
- Sentiment analysis with HuggingFace transformers
- Visualizations: word cloud, pie chart, sentiment trends
- Deployed on Gradio

## 🧠 Tech Stack
- Python, pandas, nltk, transformers
- snscrape, praw, newspaper3k, BeautifulSoup
- matplotlib, seaborn, wordcloud
- Gradio for UI

## 🚀 How to Run
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set your Reddit API keys in `main.py`

3. Run the app:
```bash
python main.py
```

## 📄 License
Free to use for educational and research purposes.
