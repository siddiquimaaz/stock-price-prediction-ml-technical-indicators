import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import re
warnings.filterwarnings('ignore')

# Popular company ticker mappings for easy lookup
POPULAR_COMPANIES = {
    # Technology
    'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
    'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META',
    'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'salesforce': 'CRM',
    'oracle': 'ORCL', 'cisco': 'CSCO', 'intel': 'INTC', 'amd': 'AMD',
    
    # Finance
    'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'bank of america': 'BAC',
    'wells fargo': 'WFC', 'goldman sachs': 'GS', 'morgan stanley': 'MS',
    'visa': 'V', 'mastercard': 'MA', 'american express': 'AXP',
    
    # Healthcare
    'johnson & johnson': 'JNJ', 'pfizer': 'PFE', 'abbott': 'ABT',
    'merck': 'MRK', 'bristol myers': 'BMY', 'eli lilly': 'LLY',
    
    # Consumer
    'coca cola': 'KO', 'pepsi': 'PEP', 'procter & gamble': 'PG',
    'walmart': 'WMT', 'home depot': 'HD', 'mcdonalds': 'MCD',
    'nike': 'NKE', 'disney': 'DIS', 'starbucks': 'SBUX',
    
    # Commodities & ETFs
    'gold': 'GLD', 'silver': 'SLV', 'oil': 'USO', 'bitcoin': 'BTC-USD',
    'ethereum': 'ETH-USD', 'sp500': 'SPY', 's&p 500': 'SPY', 'nasdaq': 'QQQ'
}

def search_company_ticker(company_name):
    """
    Search for company ticker symbol
    """
    company_lower = company_name.lower().strip()
    
    # Direct match
    if company_lower in POPULAR_COMPANIES:
        return POPULAR_COMPANIES[company_lower]
    
    # Partial match
    for name, ticker in POPULAR_COMPANIES.items():
        if company_lower in name or name in company_lower:
            return ticker
    
    # If no match found, return the input (might be a ticker already)
    return company_name.upper()

def validate_ticker(ticker):
    """
    Validate if ticker exists using yfinance
    """
    try:
        test_ticker = yf.Ticker(ticker)
        info = test_ticker.info
        
        # Check if we got valid data
        if 'symbol' in info or 'shortName' in info:
            company_name = info.get('shortName', info.get('longName', ticker))
            return True, company_name
        else:
            return False, None
    except:
        return False, None

def get_user_input():
    """
    Interactive function to get user input for company analysis
    """
    print("🎯 STOCK/COMMODITY PRICE PREDICTION SYSTEM")
    print("=" * 50)
    print("Enter a company name or ticker symbol to analyze.")
    print("Examples: 'Apple', 'Tesla', 'Google', 'AAPL', 'BTC-USD', 'Gold'")
    print()
    
    # Show some popular options
    print("💡 Popular choices:")
    popular_display = [
        "Apple", "Microsoft", "Tesla", "Google", "Amazon", 
        "Gold", "Bitcoin", "S&P 500", "Oil", "Netflix"
    ]
    print(f"   {', '.join(popular_display)}")
    print()
    
    while True:
        user_input = input("📈 Enter company name or ticker: ").strip()
        
        if not user_input:
            print("❌ Please enter a company name or ticker symbol.")
            continue
        
        # Search for ticker
        ticker = search_company_ticker(user_input)
        print(f"🔍 Searching for: {user_input} → {ticker}")
        
        # Validate ticker
        is_valid, company_name = validate_ticker(ticker)
        
        if is_valid:
            print(f"✅ Found: {company_name} ({ticker})")
            
            # Ask for confirmation
            confirm = input(f"📊 Analyze {company_name} ({ticker})? (y/n): ").strip().lower()
            if confirm in ['y', 'yes', '']:
                return ticker, company_name
            else:
                continue
        else:
            print(f"❌ Could not find data for '{user_input}' (tried {ticker})")
            print("💡 Try:")
            print("   • Full company name: 'Apple Inc'")
            print("   • Common name: 'Apple'")
            print("   • Ticker symbol: 'AAPL'")
            print("   • Cryptocurrency: 'BTC-USD'")
            print("   • ETF: 'SPY' for S&P 500")
            
            retry = input("🔄 Try again? (y/n): ").strip().lower()
            if retry not in ['y', 'yes', '']:
                return None, None

def get_analysis_preferences():
    """
    Get user preferences for analysis
    """
    print("\n⚙️  ANALYSIS PREFERENCES")
    print("=" * 30)
    
    # Time period
    print("📅 Select data period:")
    print("   1. 6 months (6mo)")
    print("   2. 1 year (1y) - Recommended")
    print("   3. 2 years (2y)")
    print("   4. 5 years (5y)")
    
    period_map = {'1': '6mo', '2': '1y', '3': '2y', '4': '5y'}
    while True:
        period_choice = input("Choose period (1-4, default 2): ").strip()
        if period_choice == '':
            period_choice = '2'
        if period_choice in period_map:
            period = period_map[period_choice]
            break
        print("❌ Please enter 1, 2, 3, or 4")
    
    # Prediction horizon
    print(f"\n🎯 Select prediction horizon:")
    print("   1. Next day (1 day)")
    print("   2. Next 3 days")
    print("   3. Next week (5 days)")
    print("   4. Next 2 weeks (10 days)")
    
    days_map = {'1': 1, '2': 3, '3': 5, '4': 10}
    while True:
        days_choice = input("Choose prediction horizon (1-4, default 1): ").strip()
        if days_choice == '':
            days_choice = '1'
        if days_choice in days_map:
            target_days = days_map[days_choice]
            break
        print("❌ Please enter 1, 2, 3, or 4")
    
    return period, target_days

class PricePredictionModel:
    def __init__(self, symbol=None, period="2y", company_name=None):
        """
        Initialize the price prediction model
        
        Args:
            symbol (str): Stock/commodity symbol (e.g., 'AAPL', 'GOOGL', 'GLD')
            period (str): Data period ('6mo', '1y', '2y', '5y', 'max')
            company_name (str): Full company name for display
        """
        self.symbol = symbol
        self.period = period
        self.company_name = company_name or symbol
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
    def create_technical_indicators(self):
        """Create technical indicators as features"""
        if self.data is None:
            print("No data available. Please fetch data first.")
            return

        df = self.data.copy()

        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()

        # Price change features
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_5d'] = df['Close'].pct_change(5)
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Open_Close_ratio'] = df['Open'] / df['Close']

        # Volume features
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA_10']

        # Lag features
        for lag in [1, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)

        self.data = df
        print("📊 Technical indicators created successfully.")
        
    def fetch_data(self):
        """
        Fetch historical price data using yfinance API
        
        yfinance API endpoints used:
        - Historical data: https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
        - Real-time quotes: https://query1.finance.yahoo.com/v7/finance/quote
        """
        try:
            print(f"🌐 Connecting to Yahoo Finance API for {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            
            # This makes an API call to Yahoo Finance servers
            self.data = ticker.history(period=self.period)
            
            # Get current/latest price (most recent data point)
            current_price = self.data['Close'].iloc[-1]
            latest_date = self.data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"✅ Successfully fetched {len(self.data)} days of data for {self.symbol}")
            print(f"📊 Latest price: ${current_price:.2f} (as of {latest_date})")
            print(f"📅 Data range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
            
            return self.data
        except Exception as e:
            print(f"❌ Error fetching data from Yahoo Finance API: {e}")
            return None
    def get_realtime_quote(self):
        """
        Get real-time quote for the stock/commodity

        Returns:
            dict: Contains current price and day change percentage
        """
        try:
            print(f"🔄 Fetching real-time data for {self.symbol}...")
            ticker = yf.Ticker(self.symbol)

            # Get real-time info - this calls Yahoo Finance API
            info = ticker.info
            fast_info = ticker.fast_info

            realtime_data = {
                'current_price': fast_info.last_price,
                'previous_close': fast_info.previous_close,
                'day_change': fast_info.last_price - fast_info.previous_close,
                'day_change_percent': ((fast_info.last_price - fast_info.previous_close) / fast_info.previous_close) * 100,
                'day_high': fast_info.day_high,
                'day_low': fast_info.day_low,
                'volume': fast_info.last_volume,
                'market_cap': info.get('marketCap', 'N/A'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            print(f"💰 Real-time Price: ${realtime_data['current_price']:.2f}")
            print(f"📈 Day Change: ${realtime_data['day_change']:.2f} ({realtime_data['day_change_percent']:.2f}%)")
            print(f"📊 Day Range: ${realtime_data['day_low']:.2f} - ${realtime_data['day_high']:.2f}")
            print(f"🕐 Last Updated: {realtime_data['timestamp']}")

            return realtime_data

        except Exception as e:
            print(f"❌ Error fetching real-time data: {e}")
            return None
        


    
    def prepare_features(self, target_days=1):
        """
        Prepare features and target for machine learning
        
        Args:
            target_days (int): Number of days ahead to predict
        """
        if self.data is None:
            print("No data available. Please create technical indicators first.")
            return
        
        # Define feature columns (excluding basic OHLCV and target)
        feature_cols = [
            'MA_5', 'MA_10', 'MA_20', 'MA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'MACD_histogram', 'RSI',
            'BB_width', 'BB_position', 'Volatility',
            'Price_change', 'Price_change_5d', 'High_Low_ratio', 'Open_Close_ratio',
            'Volume_ratio', 'Close_lag_1', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10',
            'Volume_lag_1', 'Volume_lag_3', 'Volume_lag_5', 'Volume_lag_10'
        ]
        
        # Create target (future price)
        self.data['Target'] = self.data['Close'].shift(-target_days)
        
        # Remove rows with NaN values
        clean_data = self.data[feature_cols + ['Target']].dropna()
        
        self.features = clean_data[feature_cols]
        self.target = clean_data['Target']
        
        print(f"Prepared {len(self.features)} samples with {len(feature_cols)} features")
        print(f"Target: Predicting price {target_days} day(s) ahead")
    
    def split_data(self, test_size=0.2, time_series_split=True):
        """
        Split data into training and testing sets
        
        Args:
            test_size (float): Proportion of data for testing
            time_series_split (bool): Use time-based split for time series data
        """
        if time_series_split:
            # Use time-based split to avoid data leakage
            split_idx = int(len(self.features) * (1 - test_size))
            X_train = self.features.iloc[:split_idx]
            X_test = self.features.iloc[split_idx:]
            y_train = self.target.iloc[:split_idx]
            y_test = self.target.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.target, test_size=test_size, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        
        print("Training models...")
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
        
        print("All models trained successfully")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        print("\nModel Evaluation Results:")
        print("-" * 80)
        print(f"{'Model':<20} {'MAE':<10} {'MSE':<12} {'RMSE':<10} {'R²':<10}")
        print("-" * 80)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'predictions': y_pred
            }
            
            print(f"{name:<20} {mae:<10.4f} {mse:<12.4f} {rmse:<10.4f} {r2:<10.4f}")
        
        return results
    
    def plot_predictions(self, y_test, results, model_name='Random Forest'):
        """Plot actual vs predicted prices"""
        if model_name not in results:
            print(f"Model {model_name} not found in results")
            return
        
        y_pred = results[model_name]['predictions']
        
        plt.figure(figsize=(12, 8))
        
        # Time series plot
        plt.subplot(2, 2, 1)
        plt.plot(y_test.values, label='Actual', alpha=0.8)
        plt.plot(y_pred, label='Predicted', alpha=0.8)
        plt.title(f'{model_name}: Actual vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted (Scatter)')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(2, 2, 3)
        residuals = y_test.values - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # Residuals histogram
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.features.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(15), x='importance', y='feature')
            plt.title(f'Top 15 Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print(f"{model_name} doesn't have feature importance")
    
    def predict_future(self, model_name='Random Forest', days=5):
        """Predict future prices"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        last_features = self.features.iloc[-1:].values
        last_features_scaled = self.feature_scaler.transform(last_features)
        
        prediction = model.predict(last_features_scaled)[0]
        current_price = self.data['Close'].iloc[-1]
        
        print(f"\nFuture Price Prediction using {model_name}:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${prediction:.2f}")
        print(f"Expected Change: {((prediction - current_price) / current_price) * 100:.2f}%")
        
        return prediction
    
    def run_complete_analysis(self, target_days=1):
        """Run the complete price prediction analysis with real-time data"""
        print(f"\n🚀 Starting Price Prediction Analysis")
        print(f"📊 Company: {self.company_name}")
        print(f"🎯 Symbol: {self.symbol}")
        print(f"📅 Period: {self.period}")
        print(f"🔮 Predicting: {target_days} day(s) ahead")
        print("=" * 60)
        
        # Step 0: Get real-time quote first
        realtime_data = self.get_realtime_quote()
        
        # Step 1: Fetch historical data via API
        self.fetch_data()
        
        # Step 2: Create technical indicators
        self.create_technical_indicators()
        
        # Step 3: Prepare features
        self.prepare_features(target_days=target_days)
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = self.split_data()
        
        # Step 5: Train models
        self.train_models(X_train, y_train)
        
        # Step 6: Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Step 7: Plot best model results
        best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
        print(f"\n🏆 Best Model: {best_model} (Lowest RMSE)")
        self.plot_predictions(y_test, results, best_model)
        
        # Step 8: Feature importance
        self.get_feature_importance(best_model)
        
        # Step 9: Future prediction with real-time context
        prediction = self.predict_future(best_model)
        
        # Step 10: Compare with real-time data
        if realtime_data:
            print(f"\n🔥 REAL-TIME COMPARISON:")
            print(f"Current Market Price: ${realtime_data['current_price']:.2f}")
            print(f"Model Prediction: ${prediction:.2f}")
            print(f"Prediction vs Reality: {((prediction - realtime_data['current_price']) / realtime_data['current_price']) * 100:.2f}%")
            
            # Generate trading recommendation
            self.generate_trading_recommendation(realtime_data, prediction, target_days)
        
        return results, realtime_data
    
    def generate_trading_recommendation(self, realtime_data, prediction, target_days):
        """Generate trading recommendation based on prediction"""
        current_price = realtime_data['current_price']
        expected_change = (prediction - current_price) / current_price
        day_change_pct = realtime_data['day_change_percent']
        
        print(f"\n💡 TRADING RECOMMENDATION")
        print("=" * 40)
        
        # Determine signal strength
        if abs(expected_change) > 0.05:  # 5%
            strength = "STRONG"
        elif abs(expected_change) > 0.02:  # 2%
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        # Generate signal
        if expected_change > 0.02:
            signal = "BUY"
            emoji = "📈"
        elif expected_change < -0.02:
            signal = "SELL"
            emoji = "📉"
        else:
            signal = "HOLD"
            emoji = "⏸️"
        
        print(f"{emoji} Signal: {strength} {signal}")
        print(f"💰 Target Price ({target_days}d): ${prediction:.2f}")
        print(f"📊 Expected Return: {expected_change*100:.2f}%")
        print(f"📅 Today's Change: {day_change_pct:.2f}%")
        
        # Risk assessment
        volatility = abs(day_change_pct)
        if volatility > 5:
            risk = "HIGH"
        elif volatility > 2:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        print(f"⚠️ Risk Level: {risk}")
        
        # Additional insights
        print(f"\n📝 Analysis Summary:")
        print(f"   • Company: {self.company_name}")
        print(f"   • Current Trend: {'Bullish' if day_change_pct > 0 else 'Bearish'}")
        print(f"   • Prediction Confidence: {strength}")
        print(f"   • Time Horizon: {target_days} day(s)")
        
        print(f"\n⚠️  DISCLAIMER: This is for educational purposes only.")
        print(f"   Always do your own research and consider multiple factors.")
        print(f"   Past performance does not guarantee future results.")

# Example usage
if __name__ == "__main__":
    print("\n📈 STOCK PRICE PREDICTION SYSTEM")
    print("=" * 50)

    # Get company/ticker input
    while True:
        user_input = input("🔍 Enter company name or ticker (e.g., AAPL, Tesla, BTC-USD): ").strip()
        if not user_input:
            print("❌ Please enter something.\n")
            continue

        # Lookup ticker
        symbol = search_company_ticker(user_input)
        print(f"🔎 Interpreted as: {symbol}")

        # Validate ticker
        is_valid, company_name = validate_ticker(symbol)
        if is_valid:
            print(f"✅ Found: {company_name} ({symbol})")
            break
        else:
            print(f"❌ Ticker '{symbol}' not found or no data available on Yahoo Finance.\n")

    # Choose prediction horizon
    print("\n🎯 Choose prediction horizon:")
    print("   1. Next day (1 day)")
    print("   2. 3 days")
    print("   3. 5 days (recommended)")
    print("   4. 10 days")

    horizon_options = {'1': 1, '2': 3, '3': 5, '4': 10}
    while True:
        choice = input("📅 Select option (1–4, default is 3): ").strip() or '3'
        if choice in horizon_options:
            target_days = horizon_options[choice]
            break
        print("❌ Invalid input. Please enter 1, 2, 3, or 4.")

    print("\n⏳ Running full prediction analysis...")

    model = PricePredictionModel(symbol=symbol, period="2y", company_name=company_name)
    results = model.run_complete_analysis(target_days=target_days)

    if results is None:
        print("\n⚠️ Analysis aborted due to missing or invalid data.")

# Trading Strategy Recommendations
def generate_trading_signals(model, current_data, threshold=0.02):
    """
    Generate simple trading signals based on predictions
    
    Args:
        model: Trained prediction model
        current_data: Current market data
        threshold: Minimum percentage change to generate signal
    """
    prediction = model.predict_future()
    current_price = model.data['Close'].iloc[-1]
    expected_change = (prediction - current_price) / current_price
    
    if expected_change > threshold:
        return "BUY", expected_change
    elif expected_change < -threshold:
        return "SELL", expected_change
    else:
        return "HOLD", expected_change

print("\nTRADING SIGNAL GENERATION")
print("=" * 30)
print("Based on the prediction model, here are some trading considerations:")
print("1. Use multiple models for confirmation")
print("2. Consider market conditions and external factors")
print("3. Implement proper risk management")
print("4. This is for educational purposes - not financial advice")