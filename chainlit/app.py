import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import psycopg2
from psycopg2.extras import RealDictCursor
import json

load_dotenv()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432")
    )

@cl.on_chat_start
async def on_chat_start():
    # Initialize the LLM based on configuration
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        # Use Azure OpenAI
        llm = AzureChatOpenAI(
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
    else:
        # Use direct OpenAI
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0
        )
    
    # Create a prompt template for SQL generation
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert. Given a natural language question, generate a SQL query.
        The database is a PostgreSQL database. Generate only valid PostgreSQL SQL queries.
        Only return the SQL query, nothing else. Do not include any explanations or markdown formatting.
        Make sure to use proper table and column names as they exist in the database.
        If you're not sure about the exact table or column names, use common SQL naming conventions.
        """),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    
    # Get the SQL query
    sql_query = await chain.ainvoke({
        "question": message.content
    })
    
    # Execute the query
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql_query)
        results = cur.fetchall()
        
        # Format results as a table
        if results:
            # Get column names
            columns = results[0].keys()
            
            # Create table header
            table = "| " + " | ".join(columns) + " |\n"
            table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
            
            # Add rows
            for row in results:
                table += "| " + " | ".join(str(row[col]) for col in columns) + " |\n"
            
            response = f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n**Results:**\n{table}"
        else:
            response = f"**SQL Query:**\n```sql\n{sql_query}\n```\n\nNo results found."
        
        await cl.Message(content=response).send()
        
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        await cl.Message(content=error_msg).send()
    finally:
        if 'conn' in locals():
            conn.close()
