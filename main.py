import feedparser
import fitz  # PyMuPDF for PDF handling
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from google.oauth2 import service_account
from google.auth import credentials
from google.auth.transport.requests import Request
from google.cloud import aiplatform
from langchain_community.vectorstores import FAISS
import langid
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import HumanMessage
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import yfinance as yf
import warnings
from langchain_exa import ExaSearchRetriever
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.graph_objects as go


load_dotenv() 


exa_api_key = st.secrets["exa"]["api_key"]
openai_api_key = st.secrets["openai"]["api_key"]
 
# Ignore all warnings
warnings.filterwarnings("ignore")

def extract_text_from_pdfs(pdf_paths):
    text_contents = []
    for pdf_path in pdf_paths:
        with fitz.open(pdf_path) as doc:
            pdf_name = os.path.basename(pdf_path)  # Get the PDF name from the path
            text = f"{pdf_name}\n"  # Add the PDF name at the start
            for page in doc:
                text += page.get_text()  # Extract text, ignoring images
            text_contents.append(text)
    return text_contents


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks, batch_size=10):
    try:
        # Load the GCP credentials from Streamlit secrets
        gcp_credentials = st.secrets["gcp_service_account"]
        
        # Convert credentials to dictionary if needed
        if not isinstance(gcp_credentials, dict):
            gcp_credentials_dict = json.loads(gcp_credentials) if isinstance(gcp_credentials, str) else dict(gcp_credentials)
        else:
            gcp_credentials_dict = gcp_credentials

        # Create a temporary credentials file
        credentials_path = "temp_service_account.json"
        with open(credentials_path, "w") as f:
            json.dump(gcp_credentials_dict, f)

        # Set environment variable for authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Initialize AI Platform
        aiplatform.init(
            project=gcp_credentials_dict["project_id"],
            credentials=credentials
        )

        # Configure Gemini API
        gemini_api_key = st.secrets["gemini"]["api_key"]
        genai.configure(api_key=gemini_api_key)

        # Create embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key,
            credentials=credentials
        )

        # Process text chunks in batches
        text_embeddings = []
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            try:
                batch_embeddings = embeddings.embed_documents(batch)
                text_embeddings.extend([(text, emb) for text, emb in zip(batch, batch_embeddings)])
            except Exception as e:
                st.error(f"Error processing batch {i//batch_size}: {str(e)}")
                continue

        # Create and save vector store
        if text_embeddings:
            vector_store = FAISS.from_embeddings(
                text_embeddings,
                embedding=embeddings
            )
            vector_store.save_local("faiss_index_DS")
            return vector_store
        else:
            raise ValueError("No embeddings were successfully created")

    except Exception as e:
        st.error(f"Error in get_vector_store: {str(e)}")
        st.error("Please check your credentials and permissions")
        raise

    return None

# --------------------------------------------------------------------------------
def extract_questions_from_json(json_path):
    with open(json_path, "r") as f:
        faq_data = json.load(f)
    
    questions = []
    metadata = []
    
    for entry in faq_data:
        questions.append(entry["question"])
        metadata.append({"answer": entry["answer"]}) 
    
    return questions, metadata

def get_vector_store_faq(faq_chunks, batch_size=1):
    try:
        # Load the GCP credentials from Streamlit secrets
        gcp_credentials = st.secrets["gcp_service_account"]
        
        # Convert credentials to dictionary if needed
        if not isinstance(gcp_credentials, dict):
            gcp_credentials_dict = json.loads(gcp_credentials) if isinstance(gcp_credentials, str) else dict(gcp_credentials)
        else:
            gcp_credentials_dict = gcp_credentials

        # Create a temporary credentials file
        credentials_path = "temp_service_account.json"
        with open(credentials_path, "w") as f:
            json.dump(gcp_credentials_dict, f)

        # Set environment variable for authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path 

        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Initialize AI Platform
        aiplatform.init(
            project=gcp_credentials_dict["project_id"],
            credentials=credentials
        )

        # Configure Gemini API
        gemini_api_key = st.secrets["gemini"]["api_key"]
        genai.configure(api_key=gemini_api_key)

        # Create embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key,
            credentials=credentials
        )

        # Process text chunks in batches
        text_embeddings = []
        for i in range(0, len(faq_chunks), batch_size):
            batch = faq_chunks[i:i + batch_size]
            try:
                batch_embeddings = embeddings.embed_documents(batch)
                text_embeddings.extend([(text, emb) for text, emb in zip(batch, batch_embeddings)])
            except Exception as e:
                st.error(f"Error processing batch {i//batch_size}: {str(e)}")
                continue

        # Create and save vector store
        if text_embeddings:
            vector_store_faq = FAISS.from_embeddings(
                text_embeddings,
                embedding=embeddings
            )
            vector_store_faq.save_local("faiss_index_faq")
            return vector_store_faq
        else:
            raise ValueError("No embeddings were successfully created")

    except Exception as e:
        st.error(f"Error in get_vector_store: {str(e)}")
        st.error("Please check your credentials and permissions")
        raise

    return None

# -----------------------------------------------------------------------------------
def is_input_safe(user_input):
    disallowed_phrases = [
    "ignore previous instructions",
    "bypass security",
    "reveal confidential",
    "disregard above",
    "override protocol",
    "break character",
    "act as",
    "simulate",
    "roleplay",]
    
    # Combine all disallowed phrases into a single regex pattern
    pattern = re.compile('|'.join(map(re.escape, disallowed_phrases)), re.IGNORECASE)
    return not pattern.search(user_input)


def is_relevant(question, embeddings_model, threshold=0.55):

    # Finance-related topics or sentences
    finance_topics = [
        "Financial markets and investment strategies",
        "Banking systems and monetary policies",
        "Corporate finance and capital management",
        "Economic indicators and economic growth",
        "Financial risk management and assessment",
        "Personal finance and wealth management",
        "Accounting principles and auditing",
        "Taxation and fiscal policies",
        "Stock market analysis and trading",
        "Budgeting and financial planning",
        "Low-risk investment strategies",
        "Equity allocation and diversification",
        "S&P 500 index investment",
        "Global diversification through World ETFs",
        "Bond ETFs and fixed-income securities",
        "U.S. Aggregate Bond ETFs",
        "Short-term U.S. Treasuries",
        "Corporate bonds and yield strategies",
        "Sharpe ratio and risk-adjusted return",
        "Expense ratio and fund management costs",
        "Drawdown and risk measurement",
        "Exchange-traded funds (ETFs) and passive investing",
        "Technical indicators and chart analysis",
        "Derivatives and hedging strategies",
        "Real estate investment and property management",
        "Behavioral finance and investor psychology",
        "Cryptocurrencies and blockchain technology",
        "Financial modeling and forecasting",
        "Corporate governance and ethical investing",
        "Sustainable finance and ESG investing",
        "Global trade and international finance",
        "Venture capital and private equity",
        "Financial technology (FinTech) and innovation",
        "Interest rates and yield curves",
        "Commodities trading and market analysis",
        "Insurance products and risk assessment",
        "Retirement planning and pension funds",
        "Microfinance and social impact investing",
        "Mergers and acquisitions (M&A) and valuation techniques",
        "Stock valuation methods and intrinsic value analysis",
        "Financial statement analysis and ratio analysis",
        "Cash flow management and liquidity analysis"
    ]

    # Generate embeddings for finance topics
    finance_embeddings = embeddings_model.embed_documents(finance_topics)

    # Generate embedding for the user's question
    question_embedding = embeddings_model.embed_query(question)

    # Compute cosine similarity scores
    similarities = cosine_similarity(
        [question_embedding],
        finance_embeddings
    )[0]

    # Check if any similarity exceeds the threshold
    max_similarity = max(similarities)
    # st.write(max_similarity)
    if max_similarity >= threshold:
        return True
    else:
        return False

def is_stock_query(user_question):
    prompt = f'''Analyze the following question precisely. Determine if it's a stock-related query:

    RULES:
    1. IF the question is about CURRENT STOCK PRICE, respond: "True [STOCK_SYMBOL]"
       - Examples:
         "What is Microsoft's current stock price?" → "True MSFT"
         "How much is Tesla trading for?" → "True TSLA"
         "What is the price of google?" → "True GOOGL"

    2. IF the question is about NEWS/ANALYSIS, respond: "News [REPHRASED_QUERY]"
       - Examples:
         "Why is Apple's stock falling?" → "News Why has Apple's stock price decreased?"
         "Tesla's recent financial performance" → "News What are Tesla's recent financial trends?"
         "What's the today news? → "News What is the today news?"

    3. For NON-STOCK queries, respond: "False NONE"

    Important Stock Symbols:
    - Microsoft = MSFT
    - Apple = AAPL
    - Tesla = TSLA
    - Google = GOOGL
    - Amazon = AMZN
    - Meta = META
    - Indian stocks can use .NS or .BO suffixes

    Question: {user_question}'''

    try:
        # Use Gemini for intelligent classification
        response = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            temperature=0
        )([HumanMessage(content=prompt)]).content

        # Add detailed debugging output
        # st.write(f"DEBUG: LLM Stock Query Classification - Raw Response: {response}")

        # Validate and process LLM response
        if response.startswith("True "):
            parts = response.strip().split(maxsplit=1)
            if len(parts) == 2:
                return f"True {parts[1].upper()}"
            return "False NONE"
        
        if response.startswith("News "):
            return response.strip()
        
        return "False NONE"

    except Exception as e:
        st.write(f"DEBUG: Error in is_stock_query LLM processing: {str(e)}")
        return "False NONE"

def get_stock_price(symbol):
    try:
        # Initialize variables
        stock = yf.Ticker(symbol)
        currency_symbol = "₹" if symbol.endswith(('.NS', '.BO')) else "$"
        
        # Fetch historical data with error checking
        hist = stock.history(period="5d")
        
        # Check if we received any data
        if hist.empty:
            print(f"DEBUG: No data received for symbol {symbol}")
            return None, None, None, None, None, None
            
        # Get the most recent data points
        recent_prices = hist['Close'].tail(2)
        
        # Check if we have enough data points
        if len(recent_prices) < 2:
            print(f"DEBUG: Insufficient price data for {symbol}. Got {len(recent_prices)} days of data")
            return None, None, None, None, None, None
            
        # Get current and previous prices
        stock_price = recent_prices.iloc[-1]
        previous_day_stock_price = recent_prices.iloc[-2]
        
        # Calculate changes
        price_change = stock_price - previous_day_stock_price
        change_direction = "up" if price_change > 0 else "down"
        percentage_change = (price_change / previous_day_stock_price) * 100
        
        # Debug logging
        # st.write(f"DEBUG: Successfully fetched data for {symbol}")
        # st.write(f"DEBUG: Current price: {stock_price}")
        # st.write(f"DEBUG: Previous price: {previous_day_stock_price}")
        
        return (
            stock_price,
            previous_day_stock_price,
            currency_symbol,
            price_change,
            change_direction,
            percentage_change
        )
        
    except Exception as e:
        print(f"DEBUG: Error in get_stock_price for {symbol}: {str(e)}")
        st.write(f"DEBUG: Error in get_stock_price for {symbol}: {str(e)}")
        # Return None values for all expected return values
        return None, None, None, None, None, None

# def create_research_chain(exa_api_key: str, openai_api_key: str):
#     if not exa_api_key or not isinstance(exa_api_key, str):
#         raise ValueError("Valid Exa API key is required")
        
#     # Clean the API key
#     exa_api_key = exa_api_key.strip()
    
#     # Initialize retriever with comprehensive stock market sources
#     try:
#         retriever = ExaSearchRetriever(
#             api_key=exa_api_key,
#             k=7,  # Increased for better coverage of stock-specific info
#             highlights=True,
#             search_params={
#                 "recency_days": 1,  # More recent for stock data
#                 "use_autoprompt": True,
#                 "source_filters": {
#                     "include_domains": [
#                         # Major Financial News
#                         "moneycontrol.com",
#                         "economictimes.indiatimes.com",
#                         "livemint.com",
#                         "ndtv.com/business",
#                         "business-standard.com",
                        
#                         # Stock Market Specific
#                         "nseindia.com",
#                         "bseindia.com",
#                         "tickertape.in",
#                         "screener.in",
#                         "tradingview.com",
#                         "investing.com/indices/sensex",
#                         "investing.com/indices/s-p-cnx-nifty",
                        
#                         # Market Analysis
#                         "valueresearchonline.com",
#                         "marketsmojo.com",
#                         "trendlyne.com",
#                         "stockedge.com",
#                         "chartink.com",
                        
#                         # Trading Platforms
#                         "zerodha.com/z-connect",
#                         "upstox.com/market-talk",
#                         "angelone.in/knowledge-center",
                        
#                         # Research and Analytics
#                         "groww.in/blog",
#                         "equitymaster.com",
#                         "stocksandbonds.info",
#                         "indiainfoline.com",
#                         "capitalmarket.com"
#                     ],
#                     "content_type": ["webpage", "article", "blog_post"],
#                     "exclude_domains": [
#                         "forum.",  # Exclude forum discussions
#                         "community.",
#                         "chat."
#                     ]
#                 },
#                 "post_filter": {
#                     "min_word_count": 100  # Filter out very short content
#                 }
#             }
#         )
        
#         if hasattr(retriever, 'client'):
#             retriever.client.headers.update({
#                 "x-api-key": exa_api_key,
#                 "Content-Type": "application/json"
#             })
            
#     except Exception as e:
#         st.error(f"Error initializing retriever: {str(e)}")
#         raise

#     # Enhanced document processing with stock-specific metadata
#     def format_doc(doc):
#         metadata = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else {}
#         highlights = metadata.get('highlights', 'No highlights available.')
#         url = metadata.get('url', 'No URL available.')
#         published_date = metadata.get('published_date', 'Date not available')
#         source_name = url.split('/')[2] if url.startswith('http') else 'Unknown Source'
        
#         # Format the document text with enhanced source information
#         formatted_text = f"""
#         Source: {source_name}
#         URL: {url}
#         Published Date: {published_date}
#         Highlights: {highlights}
#         Content: {doc.page_content if hasattr(doc, 'page_content') else str(doc)}
#         """
#         return formatted_text

#     def process_docs(docs):
#         if not docs:
#             return "No recent stock market information found. Please try a different query."
#         formatted_docs = [format_doc(doc) for doc in docs]
#         return "\n\n".join(formatted_docs)

#     st.write(f"retriever: {retriever}")
#     st.write(f"RunnableLambda: {RunnableLambda(process_docs)}")

#     # Enhanced retrieval chain
#     retrieval_chain = retriever | RunnableLambda(process_docs)
#     st.write(f"retrieval_chain: {retrieval_chain}")

#     # Initialize LLM with optimized settings for stock analysis
#     try:
#         llm = ChatOpenAI(
#             api_key=openai_api_key,
#             temperature=0.1,  # Even lower for more precise stock information
#             model="gpt-4-turbo-preview",
#             max_tokens=2000  # Increased for detailed analysis
#         )
#     except Exception as e:
#         st.error(f"Error initializing LLM: {str(e)}")
#         raise

#     # Enhanced prompt template with stock market focus
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are a highly knowledgeable finance and stocks assistant for India. Your role is to provide the latest news, trends, and insights related to finance and stock markets.

#         IMPORTANT GUIDELINES:
#         1. Always check and mention the date and time of stock market information
#         2. Highlight real-time market movements and trends
#         3. Include relevant market indices (Sensex, Nifty) when discussing stocks
#         4. Specify if quoted prices are real-time, closing, or historical
#         5. Note any significant intraday movements
#         6. Include trading volumes when relevant
#         7. Mention any circuit breakers or trading halts
#         8. Reference sector-specific trends
        
#         STOCK DATA PRESENTATION:
#         1. Stock Prices: Always include
#            - Current/Last Price
#            - Day's High/Low
#            - Volume (if available)
#            - % Change
        
#         2. Technical Indicators (when available):
#            - Moving Averages
#            - Support/Resistance levels
#            - Trading patterns
        
#         3. Company Info:
#            - Market Cap
#            - P/E Ratio
#            - Latest company news/announcements
        
#         [Previous formatting rules remain the same...]
#         """),
#         ("human", """
#         Please provide the most up-to-date stock market analysis using recent information from the context. Include specific dates, times, and sources for all data points.

#         Query: {query}
        
#         Context: {context}

#         Remember to:
#         1. Specify the timestamp for each price quote
#         2. Note market hours and trading status
#         3. Highlight any breaking news affecting stocks
#         4. Include relevant sector-specific context
#         5. Mention data sources and their timestamps
#         """)
#     ])

#     # Final chain with error handling
#     chain = (
#         RunnableParallel({
#             "query": RunnableLambda(lambda x: x),
#             "context": retrieval_chain
#         })
#         | prompt
#         | llm
#     )

#     return chain

def create_research_chain(exa_api_key: str, openai_api_key: str):
    if not exa_api_key or not isinstance(exa_api_key, str):
        raise ValueError("Valid Exa API key is required")
        
    # Clean the API key
    exa_api_key = exa_api_key.strip()
    
    # Initialize retriever with comprehensive stock market sources
    try:
        retriever = ExaSearchRetriever(
            api_key=exa_api_key,
            k=3,  # Increased for better coverage of stock-specific info
            highlights=True
        )
        
        if hasattr(retriever, 'client'):
            retriever.client.headers.update({
                "x-api-key": exa_api_key,
                "Content-Type": "application/json"
            })
            
    except Exception as e:
        st.error(f"Error initializing retriever: {str(e)}")
        raise

    # Create document formatting template
    document_template = """
    <source>
        <url>{url}</url>
        <highlights>{highlights}</highlights>
    </source>
    """
    document_prompt = PromptTemplate.from_template(document_template)
    
    # Create document processing chain
    document_chain = (
        RunnablePassthrough() | 
        RunnableLambda(lambda doc: {
            "highlights": doc.metadata.get("highlights", "No highlights available."),
            "url": doc.metadata.get("url", "No URL available.")
        }) | document_prompt
    )
    
    # Create retrieval chain
    retrieval_chain = (
        retriever | 
        document_chain.map() | 
        RunnableLambda(lambda docs: "\n".join(str(doc) for doc in docs))  # Convert docs to strings
    )


    # Initialize LLM with optimized settings for stock analysis
    try:
        llm = ChatOpenAI(
            api_key=openai_api_key,
            temperature=0.1,  # Even lower for more precise stock information
            model="gpt-4-turbo-preview",
            max_tokens=2000  # Increased for detailed analysis
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        raise

    # Enhanced prompt template with stock market focus
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly knowledgeable finance and stocks assistant for India. Your role is to provide the latest news, trends, and insights related to finance and stock markets.

        IMPORTANT GUIDELINES:
        1. Always check and mention the date and time of stock market information
        2. Highlight real-time market movements and trends
        3. Include relevant market indices (Sensex, Nifty) when discussing stocks
        4. Specify if quoted prices are real-time, closing, or historical
        5. Note any significant intraday movements
        6. Include trading volumes when relevant
        7. Mention any circuit breakers or trading halts
        8. Reference sector-specific trends
        
        STOCK DATA PRESENTATION:
        1. Stock Prices: Always include
           - Current/Last Price
           - Day's High/Low
           - Volume (if available)
           - % Change
        
        2. Technical Indicators (when available):
           - Moving Averages
           - Support/Resistance levels
           - Trading patterns
        
        3. Company Info:
           - Market Cap
           - P/E Ratio
           - Latest company news/announcements
        
        [Previous formatting rules remain the same...]
        """),
        ("human", """
        Please provide the most up-to-date stock market analysis using recent information from the context. Include specific dates, times, and sources for all data points.

        Query: {query}
        
        Context: {context}

        Remember to:
        1. Specify the timestamp for each price quote
        2. Note market hours and trading status
        3. Highlight any breaking news affecting stocks
        4. Include relevant sector-specific context
        5. Mention data sources and their timestamps
        """)
    ])

    # Final chain with error handling
    chain = (
        RunnableParallel({
            "query": RunnablePassthrough(),
            "context": retrieval_chain
        })
        | prompt
        | llm
    )

    return chain

def execute_research_query(question: str):
    try:
        # Get API keys
        exa_api_key = st.secrets.get("exa", {}).get("api_key") or os.getenv("EXA_API_KEY")
        openai_api_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")

        # Validate API keys
        if not exa_api_key:
            st.error("Exa API key is missing")
            return {"output_text": "Configuration error: Exa API key is not set"}
        
        if not openai_api_key:
            st.error("OpenAI API key is missing")
            return {"output_text": "Configuration error: OpenAI API key is not set"}

        # Execute chain with error handling
        try:
            # st.write("Debug - Creating research chain...")
            research_chain = create_research_chain(exa_api_key, openai_api_key)
            # st.write("Debug - Research chain created successfully")
            
            # st.write(f"Debug - Executing query: {question[:50]}...")
            response = research_chain.invoke(question)
            # st.write("Debug - Query executed successfully")
            
            # Handle response
            if hasattr(response, 'content'):
                return {"output_text": response.content}
            else:
                return {"output_text": str(response)}
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"Chain execution error: {error_msg}")
            
            # Extract detailed error information
            if hasattr(e, 'response'):
                try:
                    error_details = e.response.json() if hasattr(e.response, 'json') else e.response.text
                    st.error(f"API Response details: {error_details}")
                except:
                    st.error(f"Raw response: {e.response}")
                    
            return {"output_text": f"Error during execution: {error_msg}"}

    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        return {"output_text": f"An unexpected error occurred: {str(e)}"}
        
def plot_stock_graph(symbol):
    try:
        # Period selection
        period = st.selectbox(
            "Select Time Period", 
            ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd', 'max'], 
            index=2  # Default to 1 month
        )
        
        # Validate period input
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd', 'max']
        if period not in valid_periods:
            st.error(f"Invalid period. Choose from {', '.join(valid_periods)}")
            return False
        
        # Get stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            st.error(f"No data found for {symbol}")
            return False
            
        # Determine currency symbol based on exchange
        currency_symbol = "₹" if symbol.endswith(('.NS', '.BO')) else "$"
        
        # Calculate price changes
        price_change = hist['Close'][-1] - hist['Close'][0]
        price_change_pct = (price_change / hist['Close'][0]) * 100
        is_positive = price_change >= 0
        
        # Create period label
        period_labels = {
            '1d': '1 Day',
            '5d': '5 Days', 
            '1mo': '1 Month', 
            '3mo': '3 Months', 
            '6mo': '6 Months', 
            '1y': '1 Year', 
            '2y': '2 Years', 
            '5y': '5 Years', 
            'ytd': 'Year to Date', 
            'max': 'Maximum'
        }
        period_label = period_labels.get(period, period)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['Close'],
            mode='lines+markers',  # Add markers to show data points
            name='Close Price',
            line=dict(
                color='#00C805' if is_positive else '#FF3E2E',
                width=2
            ),
            marker=dict(
                size=8,
                color='#ffffff',  # Set marker color to white
                line=dict(
                    color='#00C805' if is_positive else '#FF3E2E',
                    width=2
                )
            ),
            hovertemplate='Date: %{x}<br>Price: ' + currency_symbol + '%{y:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{symbol} Stock Price | {period_label}',
                x=0.5,  # Center title
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            xaxis_title='Date',
            yaxis_title=f'Price ({currency_symbol})',
            hovermode='x unified',
            template='plotly_dark',  # Dark theme
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=50, b=80),  # Increased bottom margin for annotation
        )
        
        # Adjust axis range for 1-day period
        if period == '1d':
            fig.update_xaxes(
                range=[hist.index[0], hist.index[-1]],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                rangeslider_visible=False
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickprefix=currency_symbol
            )
        else:
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                rangeslider_visible=True
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickprefix=currency_symbol
            )
        
        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        return True
        
    except Exception as e:
        st.error(f"Error plotting graph: {str(e)}")
        return False

# ----------------------------------------------------------------------------------------------------------

def user_input(user_question):
    try:
        MAX_INPUT_LENGTH = 500

        # Check for input length
        if len(user_question) > MAX_INPUT_LENGTH:
            st.error(f"Input is too long. Please limit your question to {MAX_INPUT_LENGTH} characters.")
            return {"output_text": f"Input exceeds the maximum length of {MAX_INPUT_LENGTH} characters."}

        # Sanitize user input
        if not is_input_safe(user_question):
            st.error("Your input contains disallowed content. Please modify your question.")
            return {"output_text": "Input contains disallowed content."}

        # Initialize embeddings model
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Check if question is relevant to finance
        # if not is_relevant(user_question, embeddings_model, threshold=0.5):
        #     st.error("Your question is not relevant to Paasa or finance. Please ask a finance-related question.")
        #     return {"output_text": "Your question is not relevant to Paasa or finance. Please ask a finance-related question."}

        # Check for stock query
        result = is_stock_query(user_question)
        print(f"DEBUG: Processed query - Result: {result}")
        
        # Handle current stock price query
        if result.startswith("True "):
            _, symbol = result.split(maxsplit=1)
            try:
                st.info("Using Stocks response")
                stock_price, previous_day_stock_price, currency_symbol, price_change, change_direction, percentage_change = get_stock_price(symbol)
                if stock_price is not None:
                    output_text = (
                        f"**Stock Update for {symbol}**\n\n"
                        f"- Current Price: {currency_symbol}{stock_price:.2f}\n\n"
                        f"\n- Previous Close: {currency_symbol}{previous_day_stock_price:.2f}\n\n"
                        # f"{'📈' if change_direction == 'up' else '📉'} "
                        # f"The share price has {change_direction} by {currency_symbol}{abs(price_change):.2f} "
                        # f"({percentage_change:+.2f}%) compared to the previous close!"
                    )
                    
                    # Generate and return graph after text
                    return {
                        "output_text": output_text,
                        "graph": plot_stock_graph(symbol),
                        "display_order": ["text", "graph"]  # Optional: add explicit ordering
                    }

                else:
                    return {
                        "output_text": f"Sorry, I was unable to retrieve the current stock price for {symbol}."
                    }
            except Exception as e:
                print(f"DEBUG: Stock price error: {str(e)}")
                return {
                    "output_text": f"An error occurred while trying to get the stock price for {symbol}: {str(e)}"
                }
        
        # Handle stock news/analysis query
        elif result.startswith("News "):
            try:
                # Remove "News " prefix to get the original research query
                research_query = result[5:]
                
                # Directly use Exa research for news-type queries
                st.info("Using Exa Research response")

                # exa_api_key = st.secrets["news"]["EXA_API_KEY"]
                # openai_api_key = st.secrets["news"]["OPENAI_API_KEY"]

                # exa_api_key = st.secrets["exa"]["api_key"]
                # openai_api_key = st.secrets["openai"]["api_key"]


                exa_api_key = st.secrets.get("exa", {}).get("api_key", os.getenv("EXA_API_KEY"))
                openai_api_key = st.secrets.get("openai", {}).get("api_key", os.getenv("OPENAI_API_KEY"))

                if not exa_api_key or not openai_api_key:
                    raise ValueError("API keys are missing. Ensure they are in Streamlit secrets or environment variables.")

                # research_chain = create_research_chain(exa_api_key, openai_api_key)
                # Execute the research query
                # research_result = execute_research_query(research_chain, research_query)
                research_result = execute_research_query(research_query)
                return research_result

            except Exception as e:
                print(f"DEBUG: Research query error: {str(e)}")
                return {
                    "output_text": f"An error occurred while researching your query: {str(e)}"
                }
        
        # Instead, use a more direct approach
        # else:
        #     st.info("Using LLM response")
        #     prompt1 = user_question + """ In the context of Finance       
        #     (STRICT NOTE: DO NOT PROVIDE ANY ADVISORY REGARDS ANY PARTICULAR STOCKS AND MUTUAL FUNDS
        #         for example, 
        #         - which are the best stocks to invest 
        #         - which stock is worst
        #         - Suggest me best stocks )"""
    
        #     response = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)([HumanMessage(content=prompt1)])
        #     return {"output_text": response.content} if response else {"output_text": "No response generated."}


        
        # Generate embedding for the user question
        question_embedding = embeddings_model.embed_query(user_question)
        
        # -----------------------------------------------------
        # Retrieve documents from FAISS for PDF content
        new_db1 = FAISS.load_local("faiss_index_DS", embeddings_model, allow_dangerous_deserialization=True)
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=new_db1.as_retriever(search_kwargs={'k': 3}),
            llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        )
        
        docs = mq_retriever.get_relevant_documents(query=user_question)
        
        # Compute similarity scores for PDF content
        pdf_similarity_scores = []
        for doc in docs:
            doc_embedding = embeddings_model.embed_query(doc.page_content)
            score = cosine_similarity([question_embedding], [doc_embedding])[0][0]
            pdf_similarity_scores.append(score)

        max_similarity_pdf = max(pdf_similarity_scores) if pdf_similarity_scores else 0
        
        # ----------------------------------------------------------
        # Retrieve FAQs from FAISS
        new_db2 = FAISS.load_local("faiss_index_faq", embeddings_model, allow_dangerous_deserialization=True)
        mq_retriever_faq = MultiQueryRetriever.from_llm(
            retriever=new_db2.as_retriever(search_kwargs={'k': 3}),
            llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        )
        
        faqs = mq_retriever_faq.get_relevant_documents(query=user_question)
        
        # Compute similarity scores for FAQ content and store with their metadata
        faq_similarity_scores = []
        faq_with_scores = []
        for faq in faqs:
            faq_embedding = embeddings_model.embed_query(faq.page_content)
            score = cosine_similarity([question_embedding], [faq_embedding])[0][0]
            faq_similarity_scores.append(score)
            faq_with_scores.append((score, faq))

        max_similarity_faq = max(faq_similarity_scores) if faq_similarity_scores else 0
        
        # ---------------------------------------------------------------------------
        max_similarity = max(max_similarity_pdf, max_similarity_faq)

        # -------------------------------------------------------------------------------------------

        # Process based on similarity scores
        # if max_similarity < 0.65:
        #     st.info("Using LLM response")
        #     prompt1 = user_question + """ In the context of Finance       
        #     (STRICT NOTE: DO NOT PROVIDE ANY ADVISORY REGARDS ANY PARTICULAR STOCKS AND MUTUAL FUNDS
        #         for example, 
        #         - which are the best stocks to invest 
        #         - which stock is worst
        #         - Suggest me best stocks )"""
    
        #     response = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)([HumanMessage(content=prompt1)])
        #     return {"output_text": response.content} if response else {"output_text": "No response generated."}

        # -------------------------------------------------------------------------------------------


        # Handle FAQ and PDF responses
        try:
            with open('./faq.json', 'r') as f:
                faq_data = json.load(f)

            # Create a dictionary to map questions to answers
            faq_dict = {entry['question']: entry['answer'] for entry in faq_data}

            if max_similarity_faq >= max_similarity_pdf and max_similarity_faq >= 0.85:
                st.info("Using FAQ response")
                best_faq = max(faq_with_scores, key=lambda x: x[0])[1]
                
                if best_faq.page_content in faq_dict:
                    answer = faq_dict[best_faq.page_content]
                    prompt_template = """
                    Question: {question}

                    The provided answer is:
                    {answer}

                    Based on this information, let me expand on the response:

                    {context}

                    Please let me know if you have any other questions about Paasa or its services. I'm happy to provide more details or clarification.
                    """
                    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "answer", "context"])
                    chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0), chain_type="stuff", prompt=prompt)
                    response = chain({"input_documents": docs, "question": user_question, "answer": answer, "context": """
                    Paasa is a financial platform that enables global market access and portfolio diversification without hassle. It was founded by the team behind the successful US digital bank, SoFi. Paasa offers cross-border flows, tailored portfolios, and individualized guidance for worldwide investing. Their platform helps users develop wealth while simplifying the complexity of global investing.
                    """}, return_only_outputs=True)
                    return response
                elif hasattr(best_faq, 'metadata') and 'answer' in best_faq.metadata:
                    return {"output_text": best_faq.metadata['answer']}
                else:
                    return {"output_text": best_faq.page_content}
            else:
                st.info("Using PDF response")
                prompt_template = """ About the company: 
                Paasa believes location shouldn't impede global market access. Without hassle, our platform lets anyone diversify their capital internationally. We want to establish a platform that helps you expand your portfolio globally utilizing the latest technology, data, and financial tactics.
                Formerly SoFi, we helped develop one of the most successful US all-digital banks. Many found global investment too complicated and unattainable. So we departed to fix it.
                Paasa offers cross-border flows, tailored portfolios, and individualized guidance for worldwide investing. Every component of our platform, from dollar-denominated accounts to tax-efficient tactics, helps you develop wealth while disguising complexity.
                Answer the Question in brief and should be within 100 words only.
                Background:\n{context}?\n
                Question:\n{question}. + Explain in detail.\n
                Answer:
                """ 
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0), chain_type="stuff", prompt=prompt)
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                return response

        except Exception as e:
            print(f"DEBUG: Error in FAQ/PDF processing: {str(e)}")
            return {"output_text": "I apologize, but I encountered an error while processing your question. Please try again."}

    except Exception as e:
        print(f"DEBUG: Error in user_input: {str(e)}")
        return {"output_text": "An error occurred while processing your request. Please try again."}