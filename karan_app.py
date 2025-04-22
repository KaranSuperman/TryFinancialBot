import streamlit as st
from karan_main import extract_text_from_pdfs, get_text_chunks, get_vector_store, extract_questions_from_json, get_vector_store_faq, user_input

#changes
# ---------------------------------------------------------
# from supabase import create_client, Client

# Initialize Supabase client
# supabase_url = st.secrets["supabase"]["url"]
# supabase_key = st.secrets["supabase"]["key"]
# supabase: Client = create_client(supabase_url, supabase_key)

# def store_chat_data(user_message, bot_response):
#     data = {
#         "user_message": user_message,
#         "bot_response": bot_response,
#     }
#     try:
#         response = supabase.table("chat_data").insert(data).execute()
#         return response
#     except Exception as e:
#         st.error(f"Error storing chat data: {e}")
#         return None

# def store_questions(user_message):
#     data = {
#         "questions": user_message
#     }
#     try:
#         response = supabase.table("questions_data").insert(data).execute()
#         return response
#     except Exception as e:
#         st.error(f"Error storing chat data: {e}")
#         return None

# def store_answers(bot_response):
#     data = {
#         "answers": bot_response
#     }
#     try:
#         response = supabase.table("answers_data").insert(data).execute()
#         return response
#     except Exception as e:
#         st.error(f"Error storing chat data: {e}")
#         return None
# ---------------------------------------------------------

                                                                                                                                                                        
# Title of the application
st.title("Finance Chatbot")

# Load PDF paths
pdf_paths = ['Customer_pitch_3.pdf', 'Customer_pitch.pdf', 'Protect.pdf', 'tax.pdf','Low_risk_portfolio.pdf', 'Medium_risk_portfolio.pdf', 'High_risk_portfolio.pdf', 'pdf_bro_understanding_mfs_p.pdf', 'financial-terms.pdf']  

# ------------------------------------------------------
# For PDF
# Extract text from the PDFs and chunk it
content = extract_text_from_pdfs(pdf_paths)

text_chunks = []
for c in content:
    tc = get_text_chunks(c)
    text_chunks.extend(tc)

# Create the vector store
vector_store = get_vector_store(text_chunks)

# -----------------------------------------------------
# For FAQs
questions, metadata = extract_questions_from_json("./faq.json")

# Use only the questions for creating embeddings
faq_chunks = questions

# Create the vector store
vector_store_faq = get_vector_store_faq(faq_chunks)


# -------------------------------------------------------
# User input for question
user_question = st.text_input("Ask a question about finance:")

if user_question:
    try:
        # Get the response from the chatbot
        response = user_input(user_question)
        
        # Ensure response is a dictionary
        if not isinstance(response, dict):
            response = {"output_text": str(response)}
        
        # Display the response
        st.subheader("Response:")
        bot_response = response.get("output_text", "No response generated.")
        st.write(bot_response)

        #Store chat data in Supabase
        # store_chat_data(user_question, bot_response)
        # store_questions(user_question)
        # store_answers(bot_response)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"DEBUG: Streamlit display error: {str(e)}")