### Stock Sentiment Correlation Dashboard

This project investigates the relationship between financial news sentiment and stock price movements for top tech companies. The analysis includes exploratory data analysis, technical indicators, sentiment scoring, correlation studies, and an interactive dashboard (ongoing).

##Project Structure
├── .vscode/

│   └── settings.json

├── .github/

│   └── workflows

│       ├── unittests.yml

├── .gitignore

├── requirements.txt

├── README.md

├── src/

│   ├── __init__.py

├── notebooks/

│   ├── __init__.py

│   └── README.md

├── tests/

│   ├── __init__.py

└── scripts/

    ├── __init__.py

    └── README.md


## Tasks Overview
Task 1: Clean & explore news data.
Task 2: Compute stock indicators (SMA, EMA, RSI).
Task 3: Run sentiment analysis, align dates, compute daily returns, and test correlation.

## Tools & Libraries
- pandas, numpy
- matplotlib, seaborn
- textblob (sentiment)
- pynance (indicators)
- scipy.stats (correlation)
- jupyterlab

##Setup
```bash
git clone https://github.com/WaleTg/week1_new.git
cd stock-news-sentiment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
jupyter lab  # or python scripts/sentiment_correlation.py
