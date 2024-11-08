import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from supabase import create_client, Client
import json
from typing import List, Tuple, Optional
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
    Process user input and return response using MultiQueryRetriever
    """
    try:
        # Initialize the language model
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            max_output_tokens=2048,
        )

        # Create MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=st.session_state.main_vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            llm=llm
        )

        # Generate multiple queries and get documents
        retrieved_docs = retriever.get_relevant_documents(question)

        # Create prompt template for answering
        template = """You are a helpful financial advisor chatbot. Use the following context to answer the user's question.
        If you don't know the answer based on the context, say "I don't have enough information to answer that question."
        Always provide clear, accurate, and professional responses.

        Context: {context}
        
        Question: {question}
        
        Answer: """

        PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

        # Combine all retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Get response from the model
        messages = [
            {"role": "user", "content": PROMPT.format(context=context, question=question)}
        ]
        
        response = llm.invoke(messages).content
        
        # Extract source documents for reference
        sources = []
        for doc in retrieved_docs:
            if doc.metadata.get("source"):
                sources.append(doc.metadata["source"])
        
        # Format the response
        answer = response
        if sources:
            answer += "\n\nSources: " + ", ".join(set(sources))
            
        return {"output_text": answer}

    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}"
        st.error(error_msg)
        return {"output_text": "I apologize, but I encountered an error while processing your question. Please try again."}

def initialize_vector_stores(pdf_paths: List[str]) -> Tuple[Optional[FAISS], Optional[FAISS]]:
    """Initialize both main and FAQ vector stores"""
    try:
        # Initialize embeddings model
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create main vector store from PDFs
        content = extract_text_from_pdfs(pdf_paths)
        text_chunks = []
        for c in content:
            tc = get_text_chunks(c)
            text_chunks.extend(tc)
        main_vector_store = get_vector_store(text_chunks)
        
        # Load and create FAQ vector store
        with open('faq.json', 'r') as f:
            faq_data = json.load(f)
            
        # Prepare questions and metadata
        questions = [item["question"] for item in faq_data]
        metadata_list = [{"answer": item["answer"]} for item in faq_data]
        
        # Generate embeddings for questions
        question_embeddings = embeddings_model.embed_documents(questions)
        
        # Create FAISS index for FAQs
        faq_vector_store = FAISS.from_embeddings(
            [(q, emb) for q, emb in zip(questions, question_embeddings)],
            embeddings_model,
            metadatas=metadata_list
        )
        
        # Save FAQ vector store
        faq_vector_store.save_local("faiss_index_faq")
        return main_vector_store, faq_vector_store
    
    except Exception as e:
        st.error(f"Error initializing vector stores: {str(e)}")
        return None, None

def check_faq_match(user_question: str, embeddings_model: GoogleGenerativeAIEmbeddings, 
                   faq_vector_store: FAISS, threshold: float = 0.95) -> Optional[str]:
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

    # Initialize vector stores if not already done
    if not st.session_state.vector_stores_initialized:
        main_store, faq_store = initialize_vector_stores(pdf_paths)
        if main_store is not None and faq_store is not None:
            st.session_state.main_vector_store = main_store
            st.session_state.faq_vector_store = faq_store
            st.session_state.vector_stores_initialized = True

    # Initialize Supabase connection
    try:
        supabase_url = st.secrets["supabase"]["url"]
        supabase_key = st.secrets["supabase"]["key"]
        supabase: Client = create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Error connecting to Supabase: {str(e)}")
        supabase = None

    # User input for question
    user_question = st.text_input("Ask a question about finance:")

    if user_question:
        try:
            # First check FAQ matches
            if st.session_state.faq_vector_store:
                faq_answer = check_faq_match(
                    user_question,
                    GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                    st.session_state.faq_vector_store
                )
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
                    if supabase:
                        store_chat_data(supabase, user_question, bot_response)
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main()