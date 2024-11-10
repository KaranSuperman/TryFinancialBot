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
    prompt3 = '''Analyze the following question and respond with exactly two words, following these rules:
    1. First word must be either "True" or "False" indicating if the question asks for a stock price
    2. Second word must be the stock ticker formatted as follows:
    - For Indian stocks on NSE: Add ".NS" (Example: RELIANCE.NS)
    - For Indian stocks on BSE: Add ".BO" (Example: RELIANCE.BO)
    - For non-Indian stocks: Just the ticker without any prefix/suffix (Example: AAPL)
    - Never include currency symbols ($ ^ etc.) in the ticker
    - No spaces or special characters except the dot in .NS/.BO suffix

    Question: ''' + user_question
    response = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)([HumanMessage(content=prompt3)]).content
    return response


def get_stock_price(symbol):
    try:
        # If the symbol is for an Indian company, check if it ends with '.NS' or '.BO'
        if symbol.endswith('.NS') or symbol.endswith('.BO'):
            stock = yf.Ticker(symbol)
        else:
            # For global companies, ensure the symbol is valid for global exchanges
            stock = yf.Ticker(symbol)
        
        # Fetch the latest closing price
        stock_price = stock.history(period="1d")["Close"].iloc[-1]
        return stock_price
    except Exception as e:
        return f"Stock data not available for {symbol}. Error: {str(e)}"


# def extract_stock_symbol(user_question):
#     # Look for a stock symbol with a pattern: 1-5 uppercase letters
#     # Adjust this if you want a more specific format for symbols
#     match = re.search(r'\b[A-Z]{1,5}\b', user_question)
#     return match.group(0) if match else None


def user_input(user_question):
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
    if not is_relevant(user_question, embeddings_model, threshold=0.5):
        st.error("Your question is not relevant to Paasa or finance. Please ask a finance-related question.")
        return {"output_text": "Your question is not relevant to Paasa or finance. Please ask a finance-related question."}

    result = is_stock_query(user_question)
    check, symbol = result.split() if len(result.split()) == 2 else (result, None)
    print(f"DEBUG: Ticker symbol extracted: {symbol}")  # Debugging line

    if check == "True" and symbol:
        try:
            st.info("Using Stocks response")
            stock_price = get_stock_price(symbol)
            if isinstance(stock_price, float):
                return {"output_text": f"The current stock price of {symbol} is {stock_price:.2f}."}
            else:
                return {"output_text": f"Sorry, I was unable to retrieve the current stock price for {symbol}."}
        except Exception as e:
            return {"output_text": f"An error occurred while trying to get the stock price for {symbol}: {str(e)}"}


        
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
    # st.write(f"Maximum similarity score for PDF: {max_similarity_pdf}")

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
    # st.write(f"Maximum similarity score for FAQ: {max_similarity_faq}")

    # ---------------------------------------------------------------------------
    
    max_similarity = max(max_similarity_pdf, max_similarity_faq)

    # Fallback mechanism: use LLM directly if both similarities are below threshold
    if max_similarity < 0.65:
        st.info("Using LLM response")
        prompt1 = user_question + " In the context of Finance"
        response = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)([HumanMessage(content=prompt1)])
        return {"output_text": response.content} if response else {"output_text": "No response generated."}
    else:
        with open('./faq.json', 'r') as f:
            faq_data = json.load(f)

        # Create a dictionary to map questions to answers
        faq_dict = {entry['question']: entry['answer'] for entry in faq_data}

        # Update the code that handles the FAQ response
        if max_similarity_faq >= max_similarity_pdf and max_similarity_faq >= 0.85:
            st.info("Using FAQ response")
            # Get the FAQ with the highest similarity score
            best_faq = max(faq_with_scores, key=lambda x: x[0])[1]
            
            # Check if the FAQ question matches any in the JSON data
            if best_faq.page_content in faq_dict:
                # Retrieve the answer from the dictionary
                answer = faq_dict[best_faq.page_content]
                
                # Use the answer as a reference to generate a more comprehensive response
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
                # If the question is not in the dictionary, use the metadata answer
                return {"output_text": best_faq.metadata['answer']}

            else:
                # Fallback to using the FAQ content if metadata is not available
                return {"output_text": best_faq.page_content}

        else:
            st.info("Using PDF response")
            # Use PDF content with normal prompt template
            prompt_template = """ About the company: 
            Paasa believes location shouldn't impede global market access. Without hassle, our platform lets anyone diversify their capital internationally. We want to establish a platform that helps you expand your portfolio globally utilizing the latest technology, data, and financial tactics.

            Formerly SoFi, we helped develop one of the most successful US all-digital banks. Many found global investment too complicated and unattainable. So we departed to fix it.

            Paasa offers cross-border flows, tailored portfolios, and individualized guidance for worldwide investing. Every component of our platform, from dollar-denominated accounts to tax-efficient tactics, helps you develop wealth while disguising complexity.

            Answer the Question in brief.
            Background:\n{context}?\n
            Question:\n{question}. + Explain in detail.\n
            Answer:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0), chain_type="stuff", prompt=prompt)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            return response