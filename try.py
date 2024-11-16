import yfinance as yf
from datetime import datetime, timedelta

def get_stock_rsn(symbol):
    try:
        stock = yf.Ticker(symbol)
        news = [item for item in stock.news if 'providerPublishTime' in item and 
                datetime.fromtimestamp(item['providerPublishTime']) >= datetime.now() - timedelta(days=1)]
        return {'news': news} if news else None, None
    except Exception as e:
        return None, f"Error fetching stock data: {e}"

def generate_stock_response_rsn(symbol, data):
    if not data:
        return "Sorry, I couldn't fetch the stock data at the moment."
    
    response = f"Analysis for {symbol}:\n\n"
    if data['news']:
        response += "Recent relevant news:\n" + "\n".join([f"{i+1}. {news_item['title']}" for i, news_item in enumerate(data['news'])])
    else:
        response += "No news found in the last 24 hours."
    
    return response

def analyze_stock_movement_rsn(symbol):
    data, error = get_stock_rsn(symbol)
    return error if error else generate_stock_response_rsn(symbol, data)

# Example usage
symbol = "AAPL"
print(analyze_stock_movement_rsn(symbol))