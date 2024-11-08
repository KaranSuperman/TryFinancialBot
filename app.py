import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from main import extract_text_from_pdfs, get_text_chunks, initialize_vector_stores, user_input
import os

def extract_text_from_pdfs(pdf_paths: List[str]) -> List[str]:
    """
    Placeholder for PDF text extraction - implement your PDF extraction logic here
    """
    # Implement your PDF extraction logic
    return ["Sample text from PDF"]  # Replace with actual implementation

def get_text_chunks(text: str) -> List[str]:
    """
    Placeholder for text chunking - implement your chunking logic here
    """
    # Implement your text chunking logic
    return [text]  # Replace with actual implementation

def get_vector_store(text_chunks: List[str]) -> FAISS:
    """
    Create and return a FAISS vector store from text chunks
    """
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings_model)
    return vector_store

def check_faq_match(
    user_question: str,
    embeddings_model: GoogleGenerativeAIEmbeddings,
    faq_vector_store: FAISS,
    threshold: float = 0.95
) -> Optional[str]:
    """
    Check if user question matches any FAQ with high similarity.
    """
    try:
        # Search for similar questions
        docs_and_scores = faq_vector_store.similarity_search_with_score(user_question, k=1)
        
        if docs_and_scores:
            doc, score = docs_and_scores[0]
            # Convert score to similarity (FAISS returns distance)
            similarity = 1 - score
            
            if similarity >= threshold:
                return doc.metadata.get("answer")
        
        return None
        
    except Exception as e:
        st.error(f"Error checking FAQ match: {str(e)}")
        return None

def user_input(question: str) -> dict:
    """
    Process user input and return response using ConversationalRetrievalChain
    
    Args:
        question (str): User's question
        
    Returns:
        dict: Contains the chatbot's response with key 'output_text'
    """
    try:
        if not st.session_state.get('memory'):
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

        # Create prompt template
        template = """You are a helpful financial advisor chatbot. Use the following context to answer the user's question.
        If you don't know the answer based on the context, say "I don't have enough information to answer that question."
        Always provide clear, accurate, and professional responses.

        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        
        Answer: """

        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )

        # Initialize the language model
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )

        # Create the conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.main_vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True
        )

        # Get response from the chain
        response = chain({"question": question})
        
        # Extract source documents for reference
        sources = []
        if response.get("source_documents"):
            for doc in response["source_documents"]:
                if doc.metadata.get("source"):
                    sources.append(doc.metadata["source"])
        
        # Format the response
        answer = response.get("answer", "I apologize, but I couldn't generate a response.")
        if sources:
            answer += "\n\nSources: " + ", ".join(set(sources))
            
        return {"output_text": answer}

    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}"
        st.error(error_msg)
        return {"output_text": "I apologize, but I encountered an error while processing your question. Please try again."}

# Initialize Supabase client
from supabase import create_client, Client

def main():
    # Title of the application
    st.title("Finance Chatbot")

    # Initialize session state for vector stores
    if 'vector_stores_initialized' not in st.session_state:
        st.session_state.vector_stores_initialized = False
        st.session_state.main_vector_store = None
        st.session_state.faq_vector_store = None

    # Load PDF paths
    pdf_paths = ['Low_risk_portfolio.pdf', 'Medium_risk_portfolio.pdf', 'High_risk_portfolio.pdf']

    # Initialize Supabase connection
    try:
        supabase_url = st.secrets["supabase"]["url"]
        supabase_key = st.secrets["supabase"]["key"]
        supabase: Client = create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Error connecting to Supabase: {str(e)}")
        supabase = None

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
        # Initialize embeddings model
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        try:
            # First check FAQ matches
            if st.session_state.faq_vector_store:
                faq_answer = check_faq_match(user_question, embeddings_model, st.session_state.faq_vector_store)
                if faq_answer:
                    st.subheader("Response:")
                    st.write(faq_answer)
                    if supabase:
                        store_chat_data(supabase, user_question, faq_answer)
                else:
                    # Get response from main chatbot flow
                    response = user_input(user_question)
                    st.subheader("Response:")
                    bot_response = response.get("output_text", "No response generated.")
                    st.write(bot_response)
                    # if supabase:
                    #     store_chat_data(supabase, user_question, bot_response)
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

def store_chat_data(supabase: Client, user_message: str, bot_response: str):
    """Store chat data in Supabase"""
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

if __name__ == "__main__":
    main()