def create_research_chain(exa_api_key: str, openai_api_key: str):
    if not exa_api_key or not isinstance(exa_api_key, str):
        raise ValueError("Valid Exa API key is required")
        
    # Clean the API key
    exa_api_key = exa_api_key.strip()
    
    # Initialize retriever with comprehensive stock market sources
    try:
        retriever = ExaSearchRetriever(
            api_key=exa_api_key,
            k=7,  # Increased for better coverage of stock-specific info
            highlights=True,
            search_params={
                "recency_days": 3,  # More recent for stock data
                "use_autoprompt": True,
                "source_filters": {
                    "include_domains": [
                        # Major Financial News
                        "moneycontrol.com",
                        "economictimes.indiatimes.com",
                        "livemint.com",
                        "ndtv.com/business",
                        "business-standard.com",
                        
                        # Stock Market Specific
                        "nseindia.com",
                        "bseindia.com",
                        "tickertape.in",
                        "screener.in",
                        "tradingview.com",
                        "investing.com/indices/sensex",
                        "investing.com/indices/s-p-cnx-nifty",
                        
                        # Market Analysis
                        "valueresearchonline.com",
                        "marketsmojo.com",
                        "trendlyne.com",
                        "stockedge.com",
                        "chartink.com",
                        
                        # Trading Platforms
                        "zerodha.com/z-connect",
                        "upstox.com/market-talk",
                        "angelone.in/knowledge-center",
                        
                        # Research and Analytics
                        "groww.in/blog",
                        "equitymaster.com",
                        "stocksandbonds.info",
                        "indiainfoline.com",
                        "capitalmarket.com"
                    ],
                    "content_type": ["webpage", "article", "blog_post"],
                    "exclude_domains": [
                        "forum.",  # Exclude forum discussions
                        "community.",
                        "chat."
                    ]
                },
                "post_filter": {
                    "min_word_count": 100  # Filter out very short content
                }
            }
        )
        
        if hasattr(retriever, 'client'):
            retriever.client.headers.update({
                "x-api-key": exa_api_key,
                "Content-Type": "application/json"
            })
            
    except Exception as e:
        st.error(f"Error initializing retriever: {str(e)}")
        raise

    # Enhanced document processing with stock-specific metadata
    def format_doc(doc):
        metadata = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else {}
        highlights = metadata.get('highlights', 'No highlights available.')
        url = metadata.get('url', 'No URL available.')
        published_date = metadata.get('published_date', 'Date not available')
        source_name = url.split('/')[2] if url.startswith('http') else 'Unknown Source'
        
        # Format the document text with enhanced source information
        formatted_text = f"""
        Source: {source_name}
        URL: {url}
        Published Date: {published_date}
        Highlights: {highlights}
        Content: {doc.page_content if hasattr(doc, 'page_content') else str(doc)}
        """
        return formatted_text

    def process_docs(docs):
        if not docs:
            return "No recent stock market information found. Please try a different query."
        formatted_docs = [format_doc(doc) for doc in docs]
        return "\n\n".join(formatted_docs)

    # Enhanced retrieval chain
    retrieval_chain = retriever | RunnableLambda(process_docs)

    # Initialize LLM with optimized settings for stock analysis
    try:
        llm = ChatOpenAI(
            api_key=openai_api_key,
            temperature=0.1,  # Even lower for more precise stock information
            model="gpt-4-turbo-preview",
            max_tokens=2000  # Increased for detailed analysis
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        raise

    # Enhanced prompt template with stock market focus
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly knowledgeable finance and stocks assistant for India. Your role is to provide the latest news, trends, and insights related to finance and stock markets.

        IMPORTANT GUIDELINES:
        1. Always check and mention the date and time of stock market information
        2. Highlight real-time market movements and trends
        3. Include relevant market indices (Sensex, Nifty) when discussing stocks
        4. Specify if quoted prices are real-time, closing, or historical
        5. Note any significant intraday movements
        6. Include trading volumes when relevant
        7. Mention any circuit breakers or trading halts
        8. Reference sector-specific trends
        
        STOCK DATA PRESENTATION:
        1. Stock Prices: Always include
           - Current/Last Price
           - Day's High/Low
           - Volume (if available)
           - % Change
        
        2. Technical Indicators (when available):
           - Moving Averages
           - Support/Resistance levels
           - Trading patterns
        
        3. Company Info:
           - Market Cap
           - P/E Ratio
           - Latest company news/announcements
        
        [Previous formatting rules remain the same...]
        """),
        ("human", """
        Please provide the most up-to-date stock market analysis using recent information from the context. Include specific dates, times, and sources for all data points.

        Query: {query}
        
        Context: {context}

        Remember to:
        1. Specify the timestamp for each price quote
        2. Note market hours and trading status
        3. Highlight any breaking news affecting stocks
        4. Include relevant sector-specific context
        5. Mention data sources and their timestamps
        """)
    ])

    # Final chain with error handling
    chain = (
        RunnableParallel({
            "query": RunnableLambda(lambda x: x),
            "context": retrieval_chain
        })
        | prompt
        | llm
    )

    return chain