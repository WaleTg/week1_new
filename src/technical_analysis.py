import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import yfinance as yf
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsapplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import talib as ta

# --- Global Configuration (Adjust paths as needed) ---
NEWS_FILE_PATH = os.path.join('data', 'raw_analyst_ratings.csv')
STOCK_DATA_FOLDER = os.path.join('data', 'yfinance_data')
MAIN_TICKER_FOR_VIZ = 'AAPL' # Stock ticker for detailed visualizations

# --- SpaCy NLP Model Loading (for Named Entity Recognition) ---
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("SpaCy model 'en_core_web_sm' not found. Attempting download (requires internet and sufficient permissions).")
    try:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        print("SpaCy model downloaded and loaded successfully.")
    except Exception as e:
        print(f"Failed to download SpaCy model: {e}. Named entity extraction will be skipped.")
        nlp = None

# --- Main Technical Analysis Class ---
class TechnicalAnalysis:
    def __init__(self, news_file_path, stock_data_folder, main_ticker_for_viz):
        self.news_file_path = news_file_path
        self.stock_data_folder = stock_data_folder
        self.main_ticker_for_viz = main_ticker_for_viz
        self.df_news = None
        self.all_stock_data = {}
        self.daily_news_sentiment = None

    # --- Data Loading and Cleaning ---
    def load_and_clean_data(self):
        """Loads and cleans both news and stock data."""
        
        # Load News Data
        print(f"\n--- Loading News Data from {self.news_file_path} ---")
        self.df_news = pd.read_csv(self.news_file_path, encoding='utf-8', on_bad_lines='skip')
        print(f"✅ Loaded {len(self.df_news)} news rows.")
        self.df_news = self.df_news.dropna(subset=['headline']).reset_index(drop=True)

        if 'url' in self.df_news.columns:
            self.df_news['publisher'] = self.df_news['url'].apply(lambda x: x.split('/')[2] if pd.notna(x) and 'http' in x else 'unknown')
        elif 'author' in self.df_news.columns:
            self.df_news['publisher'] = self.df_news['author']
        else:
            self.df_news['publisher'] = 'unknown'

        if 'published_date' in self.df_news.columns:
            self.df_news['date'] = pd.to_datetime(self.df_news['published_date'], errors='coerce')
        elif 'date' in self.df_news.columns:
            self.df_news['date'] = pd.to_datetime(self.df_news['date'], errors='coerce')
        else:
            self.df_news['date'] = pd.NaT
        self.df_news = self.df_news.dropna(subset=['date'])

        # Load Stock Data
        print(f"\n--- Loading Stock Data from {self.stock_data_folder} ---")
        for file_name in os.listdir(self.stock_data_folder):
            if file_name.endswith('.csv'):
                ticker = file_name.split('_')[0]
                file_path = os.path.join(self.stock_data_folder, file_name)
                print(f"📈 Loading CSV: {file_path}...")
                df_stock = pd.read_csv(file_path, parse_dates=['Date'])
                df_stock.set_index('Date', inplace=True)
                df_stock.columns = [col.lower().replace(' ', '_') for col in df_stock.columns]
                df_stock = df_stock.dropna(subset=['close']).sort_index()
                self.all_stock_data[ticker] = df_stock
                print(f"✅ Loaded {len(df_stock)} rows for {ticker}.")

    # --- News Analysis (Task 1 components) ---
    def analyze_news(self):
        """Performs descriptive stats, time trends, topic modeling, and NER on news."""
        print("\n--- Phase 1: News Data Analysis (Task 1) ---")
        
        # Descriptive Statistics
        print("\n🧠 Headline Word Count Stats:")
        sample = self.df_news.sample(min(len(self.df_news), 10000), random_state=42)
        sample['headline_length'] = sample['headline'].apply(lambda x: len(str(x).split()))
        print(sample['headline_length'].describe())
        print("\n📰 Top 5 Publishers:")
        print(self.df_news['publisher'].value_counts().head())

        # Time Series Analysis
        print("\n--- News Publication Time Trends ---")
        if 'date' not in self.df_news.columns or self.df_news['date'].isna().all():
            print("⚠️ No usable 'date' column found for news time analysis.")
        else:
            self.df_news['day_of_week'] = self.df_news['date'].dt.day_name()
            self.df_news['month'] = self.df_news['date'].dt.to_period('M')

            plt.figure(figsize=(14, 4))
            self.df_news['date'].dt.date.value_counts().sort_index().plot()
            plt.title("🗓️ Articles Published Per Day")
            plt.ylabel("Count")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 4))
            self.df_news['day_of_week'].value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).plot(kind='bar', color='skyblue')
            plt.title("📆 Articles by Day of Week")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 4))
            self.df_news['month'].value_counts().sort_index().plot(kind='bar', color='lightgreen')
            plt.title("📅 Monthly Publishing Trend")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # Topic Modeling
        print("\n🧪 Running Topic Modeling (Overall News)...")
        sample_texts = self.df_news['headline'].astype(str).sample(min(len(self.df_news), 10000), random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        X = vectorizer.fit_transform(sample_texts)
        lda = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='batch', random_state=42)
        lda.fit(X)
        feature_names = vectorizer.get_feature_names_out()
        topics = {f"Topic {i+1}": [feature_names[j] for j in topic.argsort()[:-10-1:-1]] for i, topic in enumerate(lda.components_)}
        print("\n🔍 Topics Discovered (Overall News):")
        for topic, words in topics.items():
            print(f"{topic}: {', '.join(words)}")

        # Named Entity Recognition
        print("\n🧠 Extracting Named Entities (ORG, PERSON)...")
        if nlp is None:
            print("⚠️ spaCy NLP model not loaded. Skipping entity extraction.")
        else:
            sample_texts_ner = self.df_news['headline'].dropna().sample(min(len(self.df_news), 300), random_state=42)
            entities = {'ORG': [], 'PERSON': []}
            for doc in nlp.pipe(sample_texts_ner, disable=["parser", "tagger"]):
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text)
            top_entities = {k: Counter(v).most_common(10) for k, v in entities.items()}
            print("\n🏷️ Named Entities (Top 10):")
            for ent_type, items in top_entities.items():
                print(f"{ent_type}: {[ent for ent, _ in items]}")

        # Publisher Analysis (incorporating topic modeling per publisher)
        print("\n--- Publisher Analysis ---")
        top_publishers = self.df_news['publisher'].value_counts().head(3).index
        for pub in top_publishers:
            pub_df = self.df_news[self.df_news['publisher'] == pub].copy()
            if not pub_df.empty:
                print(f"\n--- Publisher: {pub} ---")
                sample_pub_texts = pub_df['headline'].astype(str).sample(min(len(pub_df), 1000), random_state=42)
                pub_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
                pub_X = pub_vectorizer.fit_transform(sample_pub_texts)
                pub_lda = LatentDirichletAllocation(n_components=3, max_iter=5, learning_method='batch', random_state=42)
                pub_lda.fit(pub_X)
                pub_feature_names = pub_vectorizer.get_feature_names_out()
                pub_topics = {f"Topic {i+1}": [pub_feature_names[j] for j in topic.argsort()[:-3-1:-1]] for i, topic in enumerate(pub_lda.components_)}
                print("  Topics Discovered:")
                for topic_name, words in pub_topics.items():
                    print(f"  {topic_name}: {', '.join(words)}")
            else:
                print(f"--- No data for Publisher: {pub} ---")

    # --- Financial Analysis (Task 2 components) ---
    def analyze_financial_data(self):
        """Performs technical indicator calculation and visualization for selected stock."""
        print("\n--- Phase 2: Financial Data Analysis (Task 2) ---")
        if self.main_ticker_for_viz not in self.all_stock_data:
            print(f"⚠️ Stock data for {self.main_ticker_for_viz} not found for detailed financial analysis.")
            return

        df_main_ticker = self.all_stock_data[self.main_ticker_for_viz].copy()
        
        # Ensure sufficient data for TA-Lib
        if len(df_main_ticker) < 50: # Most TA-Lib indicators need at least 50 periods
            print(f"⚠️ Insufficient data ({len(df_main_ticker)} rows) for {self.main_ticker_for_viz} to calculate robust technical indicators. Skipping.")
            return

        # Calculate Technical Indicators
        print(f"\n📊 Calculating technical indicators for {self.main_ticker_for_viz} with TA-Lib...")
        df_main_ticker['close'] = pd.to_numeric(df_main_ticker['close'], errors='coerce')
        df_main_ticker['open'] = pd.to_numeric(df_main_ticker['open'], errors='coerce')
        df_main_ticker['high'] = pd.to_numeric(df_main_ticker['high'], errors='coerce')
        df_main_ticker['low'] = pd.to_numeric(df_main_ticker['low'], errors='coerce')
        df_main_ticker['volume'] = pd.to_numeric(df_main_ticker['volume'], errors='coerce')
        df_main_ticker = df_main_ticker.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        df_main_ticker['SMA_20'] = ta.SMA(df_main_ticker['close'], timeperiod=20)
        df_main_ticker['SMA_50'] = ta.SMA(df_main_ticker['close'], timeperiod=50)
        df_main_ticker['EMA_20'] = ta.EMA(df_main_ticker['close'], timeperiod=20)
        df_main_ticker['EMA_50'] = ta.EMA(df_main_ticker['close'], timeperiod=50)
        df_main_ticker['RSI'] = ta.RSI(df_main_ticker['close'], timeperiod=14)
        macd, macd_signal, macd_hist = ta.MACD(df_main_ticker['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df_main_ticker['MACD'] = macd
        df_main_ticker['MACD_Signal'] = macd_signal
        df_main_ticker['MACD_Hist'] = macd_hist
        upper_bb, middle_bb, lower_bb = ta.BBANDS(df_main_ticker['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df_main_ticker['Upper_BB'] = upper_bb
        df_main_ticker['Middle_BB'] = middle_bb
        df_main_ticker['Lower_BB'] = lower_bb
        df_main_ticker['ADX'] = ta.ADX(df_main_ticker['high'], df_main_ticker['low'], df_main_ticker['close'], timeperiod=14)
        df_main_ticker['ATR'] = ta.ATR(df_main_ticker['high'], df_main_ticker['low'], df_main_ticker['close'], timeperiod=14)
        df_main_ticker = df_main_ticker.dropna(subset=['SMA_50', 'MACD', 'RSI', 'Upper_BB', 'ADX'])

        # Visualize Technical Indicators
        print(f"\n📈 Plotting indicators for {self.main_ticker_for_viz}...")
        plt.figure(figsize=(14, 6))
        plt.plot(df_main_ticker.index, df_main_ticker['close'], label='Close Price', alpha=0.7)
        plt.plot(df_main_ticker.index, df_main_ticker['SMA_20'], label='SMA 20', linestyle='--')
        plt.plot(df_main_ticker.index, df_main_ticker['SMA_50'], label='SMA 50', linestyle='--')
        plt.title(f'{self.main_ticker_for_viz} - Close Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(df_main_ticker.index, df_main_ticker['RSI'], color='orange', label='RSI')
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        plt.title(f'{self.main_ticker_for_viz} - Relative Strength Index (RSI)')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(df_main_ticker.index, df_main_ticker['MACD'], label='MACD', color='blue')
        plt.plot(df_main_ticker.index, df_main_ticker['MACD_Signal'], label='Signal Line', color='magenta')
        plt.fill_between(df_main_ticker.index, df_main_ticker['MACD_Hist'], color='gray', alpha=0.3, label='Histogram')
        plt.title(f'{self.main_ticker_for_viz} - Moving Average Convergence Divergence (MACD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 6))
        plt.plot(df_main_ticker.index, df_main_ticker['close'], label='Close Price', alpha=0.7)
        plt.plot(df_main_ticker.index, df_main_ticker['Upper_BB'], label='Upper BB', linestyle=':', color='red')
        plt.plot(df_main_ticker.index, df_main_ticker['Middle_BB'], label='Middle BB (SMA 20)', linestyle='--', color='blue')
        plt.plot(df_main_ticker.index, df_main_ticker['Lower_BB'], label='Lower BB', linestyle=':', color='red')
        plt.title(f'{self.main_ticker_for_viz} - Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Get Basic Financial Metrics
        print(f"\n📈 Fetching basic financial metrics for {self.main_ticker_for_viz}...")
        df_main_ticker['daily_return'] = df_main_ticker['close'].pct_change()
        annualized_volatility = df_main_ticker['daily_return'].std() * (252**0.5) if not df_main_ticker['daily_return'].empty else 0
        print(f"  Annualized Volatility: {annualized_volatility:.4f}")
        cumulative_returns = (1 + df_main_ticker['daily_return']).cumprod().iloc[-1] - 1 if not df_main_ticker['daily_return'].empty else 0
        print(f"  Cumulative Returns: {cumulative_returns:.4f}")
        print(f"  Summary Statistics (Key Metrics):\n{df_main_ticker[['close', 'daily_return', 'SMA_20', 'RSI', 'MACD']].dropna().describe()}")

    # --- Sentiment-Stock Integration (Task 3 components) ---
    def integrate_sentiment_and_stock(self):
        """Integrates news sentiment with stock movements and performs correlation and advanced time series analysis."""
        print("\n--- Phase 3: Sentiment-Stock Integration (Task 3) ---")
        
        # Perform Sentiment Analysis on news and aggregate daily
        print("\n😊 Performing sentiment analysis on all news headlines...")
        self.df_news['headline_str'] = self.df_news['headline'].astype(str)
        self.df_news['sentiment_polarity'] = self.df_news['headline_str'].apply(lambda x: TextBlob(x).sentiment.polarity)
        self.df_news['sentiment_subjectivity'] = self.df_news['headline_str'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        self.daily_news_sentiment = self.df_news.groupby(pd.to_datetime(self.df_news['date']).dt.date)['sentiment_polarity'].mean().reset_index()
        print("✅ Daily news sentiment aggregated.")

        # Process all stocks for correlation
        print("\n--- Performing Overall Correlation Analysis ---")
        all_correlations = {}
        for ticker, df_stock_original in self.all_stock_data.items():
            df_stock_temp = df_stock_original.copy().reset_index()
            df_stock_temp['date'] = pd.to_datetime(df_stock_temp['date']).dt.date
            df_stock_returns = df_stock_temp[['date', 'close']].copy()
            df_stock_returns['daily_return'] = df_stock_returns['close'].pct_change()
            df_stock_returns = df_stock_returns.dropna()

            merged_df_for_corr = pd.merge(self.daily_news_sentiment, df_stock_returns, on='date', how='inner')
            
            if not merged_df_for_corr.empty:
                correlation = merged_df_for_corr['sentiment_polarity'].corr(merged_df_for_corr['daily_return'])
                all_correlations[ticker] = correlation
                print(f"  Correlation for {ticker}: {correlation:.4f}")
                
                # Detailed integration for the main visualization ticker
                if ticker == self.main_ticker_for_viz:
                    print(f"\n--- Detailed Sentiment-Stock Integration for {self.main_ticker_for_viz} ---")
                    merged_for_viz = merged_df_for_corr.set_index('date').sort_index()

                    # Visualize Sentiment vs. Returns
                    print(f"📈 Visualizing sentiment vs. returns for {self.main_ticker_for_viz}...")
                    plt.figure(figsize=(14, 7))
                    plt.subplot(2, 1, 1)
                    plt.plot(merged_for_viz.index, merged_for_viz['daily_return'], label='Daily Return', alpha=0.7, color='gray')
                    plt.title(f'{self.main_ticker_for_viz} Daily Returns and News Sentiment')
                    plt.ylabel('Daily Return')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()

                    plt.subplot(2, 1, 2)
                    plt.plot(merged_for_viz.index, merged_for_viz['sentiment_polarity'], label='Sentiment Polarity', color='blue', alpha=0.8)
                    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
                    plt.xlabel('Date')
                    plt.ylabel('Sentiment Polarity')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=merged_for_viz['sentiment_polarity'], y=merged_for_viz['daily_return'], alpha=0.6)
                    plt.title(f'Scatter Plot: Sentiment vs. Daily Return for {self.main_ticker_for_viz}')
                    plt.xlabel('Sentiment Polarity')
                    plt.ylabel('Daily Return')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

                    # Perform Granger Causality (Advanced Time Series)
                    print("\n🔗 Performing Granger Causality tests...")
                    data_granger = merged_for_viz[['daily_return', 'sentiment_polarity']].dropna()
                    if not data_granger.empty and len(data_granger) > 5: # Need enough data for lags
                        print("\n  Testing if Sentiment Polarity Granger-causes Daily Return:")
                        try:
                            gc_results_s_to_r = grangercausalitytests(data_granger[['daily_return', 'sentiment_polarity']], max_lags=5, verbose=False)
                            for i in range(1, 6):
                                p_value = gc_results_s_to_r[i][0]['ssr_ftest'][1]
                                print(f"    Lag {i}: p-value = {p_value:.4f} (Significant if < 0.05)")
                        except Exception as e:
                            print(f"    Error in Granger test (sentiment to returns): {e}")

                        print("\n  Testing if Daily Return Granger-causes Sentiment Polarity:")
                        try:
                            gc_results_r_to_s = grangercausalitytests(data_granger[['sentiment_polarity', 'daily_return']], max_lags=5, verbose=False)
                            for i in range(1, 6):
                                p_value = gc_results_r_to_s[i][0]['ssr_ftest'][1]
                                print(f"    Lag {i}: p-value = {p_value:.4f} (Significant if < 0.05)")
                        except Exception as e:
                            print(f"    Error in Granger test (returns to sentiment): {e}")
                    else:
                        print("  Not enough data for Granger Causality tests.")

                    # Perform ARIMA Modeling (Advanced Time Series)
                    print(f"\n📉 Performing ARIMA Time Series Modeling on {self.main_ticker_for_viz} Daily Returns...")
                    ts_data_arima = merged_for_viz['daily_return'].dropna()
                    if len(ts_data_arima) >= 30: # Minimum data for ARIMA
                        try:
                            fig, axes = plt.subplots(1, 2, figsize=(16, 4))
                            plot_acf(ts_data_arima, ax=axes[0], lags=20, title=f'ACF for {self.main_ticker_for_viz} Daily Returns')
                            plot_pacf(ts_data_arima, ax=axes[1], lags=20, title=f'PACF for {self.main_ticker_for_viz} Daily Returns')
                            plt.tight_layout()
                            plt.show()
                            print("  (Review ACF/PACF plots to choose p and q for ARIMA model)")

                            # Simple ARIMA(1,0,1) example, more complex models require parameter tuning
                            model = ARIMA(ts_data_arima, order=(1,0,1))
                            model_fit = model.fit()
                            print(model_fit.summary())
                            print(f"  ARIMA model for {self.main_ticker_for_viz} Daily Returns fitted successfully.")
                        except Exception as e:
                            print(f"  Error fitting ARIMA model: {e}")
                            print("  Consider differencing the series or trying different ARIMA orders.")
                    else:
                        print("  Not enough data points for ARIMA modeling.")

            else:
                all_correlations[ticker] = None
                print(f"  No overlapping data for {ticker} for correlation analysis.")
                
        print("\n--- Final Correlation Results Across All Stocks ---")
        for ticker, corr_val in all_correlations.items():
            print(f"  {ticker}: {corr_val:.4f}" if corr_val is not None else f"  {ticker}: No correlation (insufficient data)")

    # --- Main Analysis Runner ---
    def run_all_analysis(self):
        """Orchestrates the entire technical analysis workflow."""
        total_start_time = time.time()
        print("🚀 Starting Integrated Technical Analysis Workflow...")

        self.load_and_clean_data()
        self.analyze_news()
        self.analyze_financial_data()
        self.integrate_sentiment_and_stock()

        total_end_time = time.time()
        print(f"\n✅ All Integrated Analysis Completed in {total_end_time - total_start_time:.2f} seconds.")

# --- Execute the Analysis ---
if __name__ == "__main__":
    # Create the 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(os.path.join('data', 'yfinance_data')):
        os.makedirs(os.path.join('data', 'yfinance_data'))
        print("\nCreated 'data/yfinance_data' folder. Please place your stock CSVs here.")

    # Create a dummy news file and stock files for demonstration if not present
    # In a real scenario, you'd download these or point to your existing files.
    if not os.path.exists(NEWS_FILE_PATH):
        print(f"Creating a dummy news file at {NEWS_FILE_PATH} for demonstration.")
        dummy_news_data = {
            'published_date': pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
            'headline': [
                'Tech giant announces positive earnings outlook',
                'Market analysts bearish on sector performance',
                'Innovation leader launches new product line',
                'Global economy faces uncertainty',
                'Company X stock upgraded by major bank'
            ],
            'url': ['http://example.com/a', 'http://example.com/b', 'http://example.com/c', 'http://example.com/d', 'http://example.com/e'],
            'publisher': ['PublisherA', 'PublisherB', 'PublisherA', 'PublisherC', 'PublisherB'],
            'stock': ['AAPL', 'AAPL', 'GOOG', 'MSFT', 'NVDA']
        }
        pd.DataFrame(dummy_news_data).to_csv(NEWS_FILE_PATH, index=False)

    # Creating dummy stock data if the folder is empty
    dummy_stock_path_aapl = os.path.join(STOCK_DATA_FOLDER, 'AAPL_historical_data.csv')
    if not os.path.exists(dummy_stock_path_aapl):
        print(f"Creating a dummy stock file for AAPL at {dummy_stock_path_aapl} for demonstration.")
        dummy_aapl_data = {
            'Date': pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
            'Open': [130, 131, 132, 133, 134],
            'High': [132, 133, 134, 135, 136],
            'Low': [129, 130, 131, 132, 133],
            'Close': [131, 132.5, 131.8, 134.2, 135.5],
            'Adj Close': [131, 132.5, 131.8, 134.2, 135.5],
            'Volume': [1000000, 1200000, 900000, 1100000, 1300000],
            'Dividends': [0,0,0,0,0],
            'Stock Splits': [0,0,0,0,0]
        }
        pd.DataFrame(dummy_aapl_data).to_csv(dummy_stock_path_aapl, index=False)

    dummy_stock_path_goog = os.path.join(STOCK_DATA_FOLDER, 'GOOG_historical_data.csv')
    if not os.path.exists(dummy_stock_path_goog):
        print(f"Creating a dummy stock file for GOOG at {dummy_stock_path_goog} for demonstration.")
        dummy_goog_data = {
            'Date': pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
            'Open': [90, 91, 90.5, 92, 91.5],
            'High': [91, 92, 91.5, 93, 92.5],
            'Low': [89, 90, 89.5, 91, 90.5],
            'Close': [90.5, 91.8, 90.2, 92.8, 91.9],
            'Adj Close': [90.5, 91.8, 90.2, 92.8, 91.9],
            'Volume': [500000, 600000, 550000, 620000, 580000],
            'Dividends': [0,0,0,0,0],
            'Stock Splits': [0,0,0,0,0]
        }
        pd.DataFrame(dummy_goog_data).to_csv(dummy_stock_path_goog, index=False)


    # Instantiate and run the analysis
    analyzer = TechnicalAnalysis(NEWS_FILE_PATH, STOCK_DATA_FOLDER, MAIN_TICKER_FOR_VIZ)
    analyzer.run_all_analysis()
