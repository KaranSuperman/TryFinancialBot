import yfinance as yf

def get_stock_price(symbol):
    try:
        # Fetching data for the given symbol
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="1d")
        print(stock_data)  # Debugging: check the data being fetched
        stock_price = stock_data["Close"].iloc[-1]  # Get the last closing price
        return stock_price
    except Exception as e:
        return f"Error fetching stock data for {symbol}: {str(e)}"

# Test with Google's ticker
print(get_stock_price('GOOGL'))
