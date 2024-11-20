from langchain_exa import ExaSearchRetriever
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
import os

def create_research_chain(exa_api_key: str, openai_api_key: str):
    # Initialize the search retriever
    retriever = ExaSearchRetriever(
        api_key=exa_api_key,
        k=3,  # Number of documents to retrieve
        highlights=True
    )
    
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
    
    # Create generation prompt
    generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly knowledgeable finance and stocks assistant. Your role is to provide the latest news, trends, and insights related to finance and stock markets. Use the XML-formatted context to ensure your responses are accurate and informative."),
        ("human", """
        Please respond to the following query using the provided context. Ensure your answer is well-structured, concise, and includes relevant data or statistics where applicable. Cite your sources at the end of your response for verification.

        Query: {query}
        ---
        <context>
        {context}
        </context>
        """)
        ])
    
    # Initialize LLM
    llm = ChatOpenAI(api_key=openai_api_key)
    
    # Combine the chains
    return (
        RunnableParallel({
            "query": RunnablePassthrough(),  
            "context": retrieval_chain,  
        }) 
        | generation_prompt 
        | llm
    )


# Example usage
if __name__ == "__main__":
    # Use environment variables for API keys
    exa_api_key = " "
    openai_api_key = os.getenv("OPENAI_API_KEY", " ")
    
    chain = create_research_chain(exa_api_key=exa_api_key, openai_api_key=openai_api_key)
    # Pass the query as a simple string
    response = chain.invoke()
    content = response.content
    print(content)