import streamlit as st
import json
from main import (extract_text_from_pdfs, get_text_chunks, get_vector_store, 
                  get_faq_embeddings, user_input, extract_text_from_faq)
from supabase import create_client, Client

# Initialize Supabase client
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

# Store chat data in Supabase
def store_chat_data(user_message, bot_response):
    data = {"user_message": user_message, "bot_response": bot_response}
    try:
        response = supabase.table("chat_data").insert(data).execute()
        return response
    except Exception as e:
        st.error(f"Error storing chat data: {e}")
        return None

# Load and process FAQ data
def load_and_process_faq(file_path='./faq.json'):
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        faq_content = extract_text_from_faq(json_data)
        faq_text_chunks = [chunk for faq in faq_content for chunk in get_text_chunks(faq)]
        return get_faq_embeddings(faq_text_chunks)
    except Exception as e:
        st.error(f"Error loading FAQ data: {e}")
        return None

# Load and process PDF data
def load_and_process_pdfs(pdf_paths):
    try:
        content = extract_text_from_pdfs(pdf_paths)
        text_chunks = [chunk for pdf_content in content for chunk in get_text_chunks(pdf_content)]
        return get_vector_store(text_chunks)
    except Exception as e:
        st.error(f"Error loading PDF data: {e}")
        return None

# Initialize the Streamlit app
st.title("Finance Chatbot")

# Load FAQ embeddings
faq_vector_store = load_and_process_faq()

# Load PDF embeddings
pdf_paths = ['Low_risk_portfolio.pdf', 'Medium_risk_portfolio.pdf', 'High_risk_portfolio.pdf']
vector_store = load_and_process_pdfs(pdf_paths)

# User input for question
user_question = st.text_input("Ask a question about finance:")

if user_question:
    # Get the response from the chatbot
    response = user_input(user_question)
    bot_response = response.get("output_text", "No response generated.")
    
    # Display the response
    st.subheader("Response:")
    st.write(bot_response)
    
    # Store chat data in Supabase
    store_chat_data(user_question, bot_response)
