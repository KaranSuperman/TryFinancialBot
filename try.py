import yfinance as yf
from datetime import datetime, timedelta

def get_stock_info(symbol):
    """
    Get current and historical stock information
    """
    try:
        # Get stock info
        stock = yf.Ticker(symbol)
        
        # Get news
        news = stock.news
        if not news:
            return None, "No news available for this stock"

        # Filter news from the past 24 hours
        now = datetime.now()
        one_day_ago = now - timedelta(days=1)
        recent_news = [
            item for item in news 
            if 'providerPublishTime' in item and 
            datetime.fromtimestamp(item['providerPublishTime']) >= one_day_ago
        ]
        
        return {
            'news': recent_news
        }, None
    except Exception as e:
        return None, f"Error fetching stock data: {str(e)}"

def generate_stock_response(symbol, data):
    """
    Generate a human-readable response about stock price changes
    """
    if not data:
        return "Sorry, I couldn't fetch the stock data at the moment."
    
    response = f"Analysis for {symbol}:\n\n"
    
    # Add recent news if available
    if data['news']:
        response += "Recent relevant news:\n"
        for i, news_item in enumerate(data['news'], 1):
            response += f"{i}. {news_item['title']}\n"
    else:
        response += "No news found in the last 24 hours."
    
    return response

def analyze_stock_movement(symbol):
    """
    Main function to analyze stock movement and generate response
    """
    data, error = get_stock_info(symbol)
    
    if error:
        return error
        
    return generate_stock_response(symbol, data)

# Example usage
symbol = "AAPL"
response = analyze_stock_movement(symbol)
print(response)