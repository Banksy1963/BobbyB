import tkinter as tk
from tkinter import messagebox
import yfinance as yf, pandas as pd, ta, datetime, time
from ib_insync import *
from sklearn.ensemble import RandomForestClassifier
from pytrends.request import TrendReq

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

def get_live_price(ticker):
    return yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]

def scan_for_momentum(ticker_list):
    selected = []
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period='5d', interval='1d').dropna()
            df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            macd = ta.trend.MACD(df['Close'])
            df['macd'] = macd.macd()
            resistance = df['Close'].rolling(window=3).max().iloc[-2]
            if (
                df['Volume'].iloc[-1] > df['Volume'].mean() * 2 and
                df['rsi'].iloc[-1] > 55 and
                df['macd'].iloc[-1] > 0 and
                df['Close'].iloc[-1] > resistance
            ):
                selected.append(ticker)
        except Exception as e:
            print(f"⚠️ Skipping {ticker}: {e}")
    return selected

class StockDataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker

    def fetch(self):
        return yf.download(self.ticker, period='20d', interval='1d').dropna()

    def get_fundamentals(self):
        info = yf.Ticker(self.ticker).info
        return {
            'pe_ratio': info.get('forwardPE'),
            'earnings_growth': info.get('earningsQuarterlyGrowth'),
            'float': info.get('floatShares', 8_000_000)
        }

    def get_google_trend_score(self):
        pytrends = TrendReq()
        pytrends.build_payload([self.ticker], timeframe='now 7-d')
        interest = pytrends.interest_over_time()
        return interest[self.ticker].iloc[-1] if not interest.empty else 0

    def is_earnings_soon(self):
        cal = yf.Ticker(self.ticker).calendar
        if 'Earnings Date' in cal:
            days = (cal['Earnings Date'][0] - pd.Timestamp.now()).days
            return days < 5
        return False

class IndicatorEngine:
    def __init__(self, df):
        self.df = df

    def apply(self):
        self.df['rsi'] = ta.momentum.RSIIndicator(self.df['Close']).rsi()
        macd = ta.trend.MACD(self.df['Close'])
        self.df['macd'] = macd.macd()
        self.df['atr'] = ta.volatility.AverageTrueRange(self.df['High'], self.df['Low'], self.df['Close']).average_true_range()
        self.df['sma'] = ta.trend.SMAIndicator(self.df['Close'], window=20).sma_indicator()
        return self.df

    def is_bullish_engulfing(self):
        last, prev = self.df.iloc[-1], self.df.iloc[-2]
        return (last['Close'] > last['Open'] and prev['Close'] < prev['Open'] and
                last['Close'] > prev['Open'] and last['Open'] < prev['Close'])

class StrategyEvaluator:
    def __init__(self, df, fundamentals, trend_score):
        self.df = df
        self.fundamentals = fundamentals
        self.trend_score = trend_score

    def predict_breakout(self):
        self.df['volume_change'] = self.df['Volume'].pct_change()
        features = self.df[['rsi', 'macd', 'volume_change']].dropna()
        features['target'] = (features['macd'] > 0).astype(int)
        model = RandomForestClassifier().fit(features.drop('target', axis=1), features['target'])
        return model.predict(features.drop('target', axis=1).tail(1))[0] == 1

    def passes_filters(self):
        adv = self.df['Volume'].mean()
        adv_today = self.df['Volume'].iloc[-1]
        price_change = (self.df['Close'].iloc[-1] - self.df['Open'].iloc[-1]) / self.df['Open'].iloc[-1]
        return (
            adv_today > adv * 5 and
            self.fundamentals['float'] < 10_000_000 and
            0.05 <= price_change <= 0.06 and
            self.df['rsi'].iloc[-1] > 50 and
            self.df['macd'].iloc[-1] > 0 and
            IndicatorEngine(self.df).is_bullish_engulfing() and
            self.predict_breakout() and
            self.fundamentals['pe_ratio'] and self.fundamentals['pe_ratio'] < 30 and
            self.fundamentals['earnings_growth'] and self.fundamentals['earnings_growth'] > 0.15 and
            self.trend_score > 50
        )

class PortfolioManager:
    def __init__(self):
        self.portfolio = {}

    def rebalance(self, ticker, entry_price):
        if ticker not in self.portfolio:
            self.portfolio[ticker] = {'entry': entry_price, 'shares': 100}
        elif len(self.portfolio) > 10:
            weakest = min(self.portfolio.items(), key=lambda x: x[1]['entry'])
            del self.portfolio[weakest[0]]

    def log_story(self, ticker, reason):
        with open('stock_stories.txt', 'a') as f:
            f.write(f"{datetime.datetime.now()} - {ticker}: {reason}\n")

class OrderExecutor:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df
        self.peak_price = df['Close'].iloc[-1]

    def calculate_targets(self):
        entry_price = self.df['Close'].iloc[-1]
        atr = self.df['atr'].iloc[-1]
        stop_loss = entry_price - atr * 1.5
        trailing_stop = entry_price * 0.95
        return entry_price, stop_loss, trailing_stop

    def place_order(self, entry_price):
        contract = Stock(self.ticker, 'SMART', 'USD')
        order = LimitOrder('BUY', 100, entry_price)
        ib.placeOrder(contract, order)
        print(f"✅ Order placed for {self.ticker} at ${entry_price:.2f}")

    def monitor_and_exit(self):
        print(f"📈 Monitoring {self.ticker} for momentum exit...")
        while True:
            current_price = get_live_price(self.ticker)
            if current_price > self.peak_price:
                self.peak_price = current_price
            trailing_stop = self.peak_price * 0.95
            if current_price < trailing_stop:
                contract = Stock(self.ticker, 'SMART', 'USD')
                order = LimitOrder('SELL', 100, current_price)
                ib.placeOrder(contract, order)
                print(f"📉 Exited {self.ticker} at ${current_price:.2f} (Trailing Stop Hit)")
                break
            time.sleep(60)

def run_bot(tickers):
    pm = PortfolioManager()
    for ticker in tickers:
        try:
            fetcher = StockDataFetcher(ticker)
            if fetcher.is_earnings_soon():
                print(f"⏳ Skipping {ticker} due to upcoming earnings.")
                continue
            df = fetcher.fetch()
            fundamentals = fetcher.get_fundamentals()
            trend_score = fetcher.get_google_trend_score()
            df = IndicatorEngine(df).apply()
            evaluator = StrategyEvaluator(df, fundamentals, trend_score)
            if evaluator.passes_filters():
                executor = OrderExecutor(ticker, df)
                entry_price, stop_loss, trailing_stop = executor.calculate_targets()
                executor.place_order(entry_price)
                pm.rebalance(ticker, entry_price)
                pm.log_story(ticker, "Momentum entry triggered")
                executor.monitor_and_exit()
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

def launch_gui():
    root = tk.Tk()
    root.title("Magellan Bot Control Panel")

    tk.Label(root, text="Enter tickers (comma-separated):").pack()
    ticker_entry = tk.Entry(root