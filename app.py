import streamlit as st
from main import extract_text_from_pdfs, get_text_chunks, get_vector_store, get_faq_embeddings, user_input

#changes
# ---------------------------------------------------------
from supabase import create_client, Client

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
# ---------------------------------------------------------

                                                                                                                                                                        
# Title of the application
st.title("Finance Chatbot")

# Load PDF paths
pdf_paths = ['Low_risk_portfolio.pdf', 'Medium_risk_portfolio.pdf', 'High_risk_portfolio.pdf']  

# Extract text from the PDFs and chunk it
content = extract_text_from_pdfs(pdf_paths)

text_chunks = []
for c in content:
    tc = get_text_chunks(c)
    text_chunks.extend(tc)

# Create the vector store
vector_store = get_vector_store(text_chunks)

# User input for question
user_question = st.text_input("Ask a question about finance:")

if user_question:
    # Check FAQ embeddings for the user input
    faq_answer = get_faq_embeddings(json_path="./faq.json")

    if faq_answer:
        # Display the FAQ-based answer
        st.subheader("FAQ Response:")
        st.write(faq_answer)
    else:
        # Proceed with the general user input processing
        response = user_input(user_question)
        st.subheader("Response:")
        st.write(response['output_text'])

    # Store chat data in Supabase
    # store_chat_data(user_question, faq_answer if faq_answer else response['output_text'])