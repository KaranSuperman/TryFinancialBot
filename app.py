import streamlit as st
from main import extract_text_from_pdfs, get_text_chunks, initialize_vector_stores, check_faq_match, user_input
from supabase import create_client, Client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize Supabase client
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

def store_chat_data(user_message, bot_response):
    data = {
        "user_message": user_message,
        "bot_response": bot_response,
    }
    try:
        response = supabase.table("chat_data").insert(data).execute()
        return response
    except Exception as e:
        st.error(f"Error storing chat data: {e}")
        return None

# Title of the application
st.title("Finance Chatbot")

# Initialize session state for vector stores
if 'vector_stores_initialized' not in st.session_state:
    st.session_state.vector_stores_initialized = False
    st.session_state.main_vector_store = None
    st.session_state.faq_vector_store = None

# Load PDF paths
pdf_paths = ['Low_risk_portfolio.pdf', 'Medium_risk_portfolio.pdf', 'High_risk_portfolio.pdf']

# Initialize vector stores if not already done
if not st.session_state.vector_stores_initialized:
    main_store, faq_store = initialize_vector_stores(pdf_paths)
    if main_store is not None and faq_store is not None:
        st.session_state.main_vector_store = main_store
        st.session_state.faq_vector_store = faq_store
        st.session_state.vector_stores_initialized = True

# User input for question
user_question = st.text_input("Ask a question about finance:")

if user_question:
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if faq_vector_store exists
    if st.session_state.faq_vector_store is None:
        st.error("FAQ vector store is not initialized.")
    else:
        # First check FAQ matches
        faq_answer = check_faq_match(user_question, embeddings_model, st.session_state.faq_vector_store)
        
        if faq_answer:
            st.subheader("Response:")
            st.write(faq_answer)
            store_chat_data(user_question, faq_answer)
        else:
            # Get response from main chatbot flow
            response = user_input(user_question)
            st.subheader("Response:")
            bot_response = response.get("output_text")
            st.write(bot_response)
            # store_chat_data(user_question, bot_response)