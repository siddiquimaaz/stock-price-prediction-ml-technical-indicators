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
warnings.filterwarnings('ignore')

class PricePredictionModel:
    def __init__(self, symbol, period="2y"):
        """
        Initialize the price prediction model
        
        Args:
            symbol (str): Stock/commodity symbol (e.g., 'AAPL', 'GOOGL', 'GLD')
            period (str): Data period ('1y', '2y', '5y', 'max')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch historical price data using yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            print(f"Successfully fetched {len(self.data)} days of data for {self.symbol}")
            return self.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
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
        print("Technical indicators created successfully")
    
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
        """Run the complete price prediction analysis"""
        print(f"Starting Price Prediction Analysis for {self.symbol}")
        print("=" * 60)
        
        # Step 1: Fetch data
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
        print(f"\nBest Model: {best_model} (Lowest RMSE)")
        self.plot_predictions(y_test, results, best_model)
        
        # Step 8: Feature importance
        self.get_feature_importance(best_model)
        
        # Step 9: Future prediction
        self.predict_future(best_model)
        
        return results

# Example usage
if __name__ == "__main__":
    # Example 1: Stock prediction
    print("STOCK PRICE PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Initialize the model for Apple stock
    stock_model = PricePredictionModel('AAPL', period='2y')
    
    # Run complete analysis
    stock_results = stock_model.run_complete_analysis(target_days=1)
    
    print("\n" + "="*50)
    
    # Example 2: Commodity prediction (Gold ETF)
    print("COMMODITY PRICE PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Initialize the model for Gold ETF
    commodity_model = PricePredictionModel('GLD', period='2y')
    
    # Run complete analysis
    commodity_results = commodity_model.run_complete_analysis(target_days=1)
    
    # Additional analysis examples
    print("\n" + "="*50)
    print("ADDITIONAL ANALYSIS EXAMPLES")
    print("=" * 50)
    
    # Compare different prediction horizons
    for days in [1, 3, 5, 10]:
        print(f"\nTesting {days}-day prediction:")
        temp_model = PricePredictionModel('AAPL', period='1y')
        temp_model.fetch_data()
        temp_model.create_technical_indicators()
        temp_model.prepare_features(target_days=days)
        X_train, X_test, y_train, y_test = temp_model.split_data()
        temp_model.train_models(X_train, y_train)
        results = temp_model.evaluate_models(X_test, y_test)
        best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
        print(f"Best model for {days}-day prediction: {best_model} (RMSE: {results[best_model]['RMSE']:.4f})")

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