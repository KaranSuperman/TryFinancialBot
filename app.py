import streamlit as st
from main import extract_text_from_pdfs, get_text_chunks, get_vector_store,get_faq_embeddings, user_input

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

def get_closest_faq_answer(user_question, vector_store):
    # Perform a similarity search in the vector store
    closest_match = vector_store.query(user_question, top_k=1)  # Adjust query method as per vector store library
    if closest_match:
        # Extract the most similar chunk text from the match result
        return closest_match[0]['text']  # Ensure to access text field as per your vector store's response format
    else:
        return None  # Return None if no close match is found

# Paths to the PDFs containing FAQ-like content
pdf_paths = ['Low_risk_portfolio.pdf', 'Medium_risk_portfolio.pdf', 'High_risk_portfolio.pdf']  

# Step 1: Extract and Chunk Text from PDFs
content = extract_text_from_pdfs(pdf_paths)

text_chunks = []
for c in content:
    tc = get_text_chunks(c)
    text_chunks.extend(tc)

# Step 2: Create the vector store from PDF chunks
vector_store = get_vector_store(text_chunks)

# User input for a question
user_question = st.text_input("Ask a question about finance:")

if user_question:
    # Step 3: Try to find the best matching FAQ chunk response for the user question
    pdf_response = get_closest_faq_answer(user_question, vector_store)
    
    if pdf_response:
        # Use the closest PDF chunk as the response
        bot_response = pdf_response
    else:
        # Fallback to a chatbot response if no relevant chunk is found
        response = user_input(user_question)
        bot_response = response.get("output_text", "No response generated.")

    # Display the response
    st.subheader("Response:")
    st.write(bot_response)
    
    # Store the chat data in Supabase
    store_chat_data(user_question, bot_response)