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
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta



load_dotenv() 


exa_api_key = st.secrets["exa"]["api_key"]
# openai_api_key = st.secrets["openai"]["api_key"]
 
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
    prompt = f'''Analyze the following question precisely. Determine if it's a stock-related or finance related query Only:
    SPECIAL NOTE: DO NOT RESPONSE IF OTHER THAN STOCKS OR FINANCE RELATED NEWS/QUESTION ASK. ALSO [PAASA] is a fintech company if 
    any user ask query related to the company then donot response to that query.

    SPECIAL NOTES: 
    - ALL news queries MUST be rephrased to focus ONLY on financial markets, economy, or business news
    - ANY general news query should be automatically converted to a financial news query
    - DO NOT include general news, political news, or non-financial news in rephrasing

    RULES:
    1. IF the question is about STOCK PRICE then Generate only [Yahoo Finance] compatible symbol, respond: "True [STOCK_SYMBOL]"
       - Examples:
         "What is Microsoft's current stock price?" → "True MSFT"
         "How much is Tesla trading for?" → "True TSLA"
         "What is the price of google?" → "True GOOGL"
         "What is price of cspx" → "True CSPX.L"
         "csndx price" → "True CSNDX.SW"
         "What is bitcoin price"  → "True BTC-USD"

    2. IF the question is about NEWS/ANALYSIS (ANY GENERAL NEWS QUERY), respond: "News [FINANCIAL_REPHRASED_QUERY]"
       - All general news queries MUST be rephrased to focus on financial markets/economy
       - Always add specific financial context when rephrasing
       Examples:
         "What's happening today?" → "News Give 5 finance news of today?"
         "Give me latest news" → "News What are the latest financial market updates?"
         "Top stories" → "News What are today's top financial market stories?"
         "US news" → "News What are the major US financial market developments?"
         "Why is Apple's stock falling?" → "News Why has Apple's stock price decreased?"
         "Tesla's recent financial performance" → "News What are Tesla's recent financial trends?"
         "What's the today news?" → "News What are today's top financial market updates?"
         "What happened today?" → "News What are the major financial market movements today?"
         "what is top 10 news?" → "News What are the top 10 financial market developments?"
         "What is the updates of US" → "News What are the latest US financial market updates?"
         "What happened to nifty50 down today?" → "News What caused the decline in Nifty50 today?"

    3. IF the question is about Finance or tax related information, respond: "News [REPHRASED_QUERY]"
       - Examples:
         "What is the market cap to gdp ratio of India?" → "News What is the market cap to gdp ratio of India?"
         "What is the tax I pay on debt ETF's overseas?" → "News What is the tax I pay on debt ETF's overseas?"

    4. Do not response on financial terms , respond: "False NONE"
        - Example: 
        "What is PE ratio?"
        "What is high risk portfolio?"


    Important Stock Symbols:
    - Microsoft = MSFT
    - Apple = AAPL
    - Tesla = TSLA
    - Google = GOOGL
    - Amazon = AMZN
    - Meta = META
    - Bitcoin = BTC-USD
    - Sensex = ^BSESN
    - Nifyt = ^NSEI
 

    COMPREHENSIVE GLOBAL STOCK SYMBOL GENERATION RULES:
    EXCHANGE SUFFIXES:
    - US Exchanges:
      * No suffix for NYSE/NASDAQ (AAPL, MSFT)
    
    NOTE: Append appropriate exchange suffix if needed
    - International Exchanges:
      - .L = London Stock Exchange (UK)
      - .SW = SIX Swiss Exchange (Switzerland)
      - .NS = National Stock Exchange (India)
      - .BO = Bombay Stock Exchange (India)
      - .JK = Indonesia Stock Exchange
      - .SI = Singapore Exchange
      - .HK = Hong Kong Stock Exchange
      - .T = Tokyo Stock Exchange (Japan)
      - .AX = Australian Securities Exchange
      - .SA = São Paulo Stock Exchange (Brazil)
      - .TO = Toronto Stock Exchange (Canada)
      - .MX = Mexican Stock Exchange
      - .KS = Korea Exchange
      - .DE = Deutsche Börse (Germany)
      - .PA = Euronext Paris
      - .AS = Euronext Amsterdam
      - .MI = Milan Stock Exchange (Italy)
      - .MC = Madrid Stock Exchange (Spain)


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
        currency_symbol = "₹" if symbol.endswith(('.NS', '.BO')) or symbol in ('^NSEI', '^BSESN') else "$"
        
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

def create_research_chain(exa_api_key: str, gemini_api_key: str):
    if not exa_api_key or not isinstance(exa_api_key, str):
        raise ValueError("Valid Exa API key is required")
    
    exa_api_key = exa_api_key.strip()
    
    try:
        # Change to 1 days (24 hours) to get very recent news
        start_date = (datetime.now() - timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')

        # Enhanced Retriever Configuration
        retriever = ExaSearchRetriever(
            api_key=exa_api_key,
            k=5,
            highlights=True,
            start_published_date=start_date,
            type="news",
            sort="date"  # Ensure sorting by date
        )

        # Ensure the API key is set in the headers
        if hasattr(retriever, 'client'):
            retriever.client.headers.update({
                "x-api-key": exa_api_key,
                "Content-Type": "application/json"
            })
        
        # Verify Gemini API key
        if not gemini_api_key or not isinstance(gemini_api_key, str):
            raise ValueError("Valid Gemini API key is required")

        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        
        # Enhanced LLM Configuration
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.2,
            google_api_key=gemini_api_key,
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )

        # Detailed Document Template
        document_template = """
        <financial_news>
            <headline>{title}</headline>
            <date>{date}</date>
            <key_insights>{highlights}</key_insights>
            <source_url>{url}</source_url>
        </financial_news>
        """
        document_prompt = PromptTemplate.from_template(document_template)
        
        document_chain = (
            RunnablePassthrough() | 
            RunnableLambda(lambda doc: {
                "title": doc.metadata.get("title", "Untitled Financial Update"),
                "date": doc.metadata.get("published_date", "Today"),
                "highlights": doc.metadata.get("highlights", "No key insights available."),
                "url": doc.metadata.get("url", "No source URL")
            }) | document_prompt
        )
        
        retrieval_chain = (
            retriever | 
            document_chain.map() | 
            RunnableLambda(lambda docs: "\n\n".join(str(doc) for doc in docs))
        )

        # Improved Financial News Prompt with Better Formatting
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior financial analyst specializing in Indian and global markets with expertise in:

            Core Areas:
            - Indian equity markets and sectoral analysis
            - Cryptocurrency markets and blockchain technology
            - Global market correlations and trends
            - Technical and fundamental analysis
            - Macroeconomic indicators and ratios
            - Market valuations and metrics

            Response Style:
            - Time-sensitive: Prioritize the most recent information
            - Quantitative: Include specific numbers, percentages, and time periods
            - Evidence-based: Support insights with recent data points
            - Comprehensive: Cover both traditional and digital assets
            - Forward-looking: Include potential market implications
            - Risk-aware: Highlight key risks and uncertainties"""),
            
            ("human", """Analyze this financial query within the given context:

            Query: {query}
            Context: {context}
            
            Structure your response based on user query.
            For time-sensitive queries (today/latest), focus only on the most recent updates.
            If no specific recent news is found, clearly state that no recent updates are available.

            IMPORTANT: For source citations, use this exact format:
            "Your news statement. [sourcename](source_url)"
            Ensure the source name is clickable.

            Maximum response length: 200 words
            Focus on actionable insights relevant to the query context.""")
        ])

        chain = (
            RunnableParallel({
                "query": RunnablePassthrough(),  
                "context": retrieval_chain,  
            }) 
            | generation_prompt 
            | llm
        )
        
        return chain

    except Exception as e:
        st.error(f"Error in create_research_chain: {str(e)}")
        raise

     

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
        currency_symbol = "₹" if symbol.endswith(('.NS', '.BO')) or symbol in ('^NSEI', '^BSESN') else "$"
        
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

        # Initial validation checks
        if len(user_question) > MAX_INPUT_LENGTH:
            st.error(f"Input is too long. Please limit your question to {MAX_INPUT_LENGTH} characters.")
            return {"output_text": f"Input exceeds the maximum length of {MAX_INPUT_LENGTH} characters."}

        if not is_input_safe(user_question):
            st.error("Your input contains disallowed content. Please modify your question.")
            return {"output_text": "Input contains disallowed content."}

        # First determine the type of query
        result = is_stock_query(user_question)
        
        # Initialize embeddings model
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        question_embedding = embeddings_model.embed_query(user_question)
        
        # 1. Check PDF content
        new_db1 = FAISS.load_local("faiss_index_DS", embeddings_model, allow_dangerous_deserialization=True)
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=new_db1.as_retriever(search_kwargs={'k': 5}),
            llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        )
        docs = mq_retriever.get_relevant_documents(query=user_question)
        
        pdf_similarity_scores = []
        for doc in docs:
            doc_embedding = embeddings_model.embed_query(doc.page_content)
            score = cosine_similarity([question_embedding], [doc_embedding])[0][0]
            pdf_similarity_scores.append(score)
        max_similarity_pdf = max(pdf_similarity_scores) if pdf_similarity_scores else 0
        
        # 2. Check FAQ content
        new_db2 = FAISS.load_local("faiss_index_faq", embeddings_model, allow_dangerous_deserialization=True)
        mq_retriever_faq = MultiQueryRetriever.from_llm(
            retriever=new_db2.as_retriever(search_kwargs={'k': 3}),
            llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        )
        faqs = mq_retriever_faq.get_relevant_documents(query=user_question)
        
        faq_similarity_scores = []
        faq_with_scores = []
        for faq in faqs:
            faq_embedding = embeddings_model.embed_query(faq.page_content)
            score = cosine_similarity([question_embedding], [faq_embedding])[0][0]
            faq_similarity_scores.append(score)
            faq_with_scores.append((score, faq))
        max_similarity_faq = max(faq_similarity_scores) if faq_similarity_scores else 0

        # If we have good matches in PDF or FAQ (similarity >= 0.67), use them
        if max(max_similarity_pdf, max_similarity_faq) >= 0.67:
            try:
                with open('./faq.json', 'r') as f:
                    faq_data = json.load(f)
                faq_dict = {entry['question']: entry['answer'] for entry in faq_data}

                # Use FAQ if it has higher similarity
                if max_similarity_faq >= max_similarity_pdf and max_similarity_faq >= 0.65:
                    # st.info("Using FAQ response")
                    best_faq = max(faq_with_scores, key=lambda x: x[0])[1]
                    
                    if best_faq.page_content in faq_dict:
                        answer = faq_dict[best_faq.page_content]
                        prompt_template = """
                        Question: {question}

                        The provided answer is:
                        {answer}

                        Based on this information, let me response within 100 words:

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
                    # Use PDF response
                    # st.info("Using PDF response")
                    prompt_template = """
                    Use the information from the provided PDF context to answer the question in detail.

                    Context:\n{context}

                    Question: {question}

                    Provide a comprehensive answer, including all relevant details and explanations. Ensure the response is clear and informative, using the factual information available in the document.
                    """
                    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                    chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0), chain_type="stuff", prompt=prompt)
                    return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            except Exception as e:
                print(f"DEBUG: Error in FAQ/PDF processing: {str(e)}")
        
        # If no good matches in PDF/FAQ, use alternative processing
        else:
            # Handle stock price queries
            if result.startswith("True "):
                # st.info("Using Stocks response")
                _, symbol = result.split(maxsplit=1)
                try:
                    stock_price, previous_day_stock_price, currency_symbol, price_change, change_direction, percentage_change = get_stock_price(symbol)
                    if stock_price is not None:
                        output_text = (
                            f"**Stock Update for {symbol}**\n\n"
                            f"- Current Price: {currency_symbol}{stock_price:.2f}\n\n"
                            f"\n- Previous Close: {currency_symbol}{previous_day_stock_price:.2f}\n\n"
                        )
                        
                        return {
                            "output_text": output_text,
                            "graph": plot_stock_graph(symbol),
                            "display_order": ["text", "graph"]
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
            
            # Handle news/analysis queries only if PDF/FAQ didn't have good matches
            elif result.startswith("News "):
                # st.info("Using Exa response")
                research_query = result[5:]
                exa_api_key = st.secrets.get("exa", {}).get("api_key", os.getenv("EXA_API_KEY"))
                gemini_api_key = st.secrets.get("gemini", {}).get("api_key", os.getenv("GEMINI_API_KEY"))

                if not exa_api_key or not gemini_api_key:
                    raise ValueError("API keys are missing")

                research_chain = create_research_chain(exa_api_key, gemini_api_key)
                response = research_chain.invoke(research_query)
                
                if hasattr(response, 'content'):
                    return {"output_text": response.content.strip()}
                else:
                    return {"output_text": "Sorry! No information avaiable for this question."}
            
            # Finally, fall back to LLM response
            else:
                # st.info("Using LLM response")
                prompt1 = user_question + """\
                Don't response if the user_question is rather than financial terms.
                If other question ask response with 'Please tell only finance related queries' .
                Finance Term Query Guidelines:
                1. Context: Finance domain
                2. Response Requirements:
                - Focus exclusively on defining finance-related terms
                - Provide clear, concise explanations of financial terminology

                Examples of Acceptable Queries:
                - What is PE ratio?
                - Define market capitalization
                - Explain book value
                - What does EBITDA mean?

                Note: Responses must be purely informative and educational about financial terms. Try to give response within 100 words with solid answer.\
                """
                response = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)([HumanMessage(content=prompt1)])
                return {"output_text": response.content} if response else {"output_text": "No response generated."}

    except Exception as e:
        print(f"DEBUG: Error in user_input: {str(e)}")
        return {"output_text": "An error occurred while processing your request. Please try again."}