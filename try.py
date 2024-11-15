import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def get_stock_info(symbol):
    """
    Get current and historical stock information
    """
    try:
        # Get stock info
        stock = yf.Ticker(symbol)
        
        # Get today's data and yesterday's data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            return None, "No data available for this stock"

        
        # Get company news
        news = stock.news
        
        return {
            'news': news
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
        for i, news_item in enumerate(data['news'][:8], 1):  # Show top 10 news items
            response += f"{i}. {news_item['title']}\n"
    
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
symbol = "AAPL"  # Tesla's stock symbol
response = analyze_stock_movement(symbol)
print(response)