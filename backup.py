import os
import json
import re
from typing import Dict, List, Any
from dotenv import load_dotenv

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import text

load_dotenv()

# Environment configuration
db_uri = os.getenv("DATABASE_URL")
ollama_url = os.getenv("OLLAMA_BASE_URL")
ollama_model = os.getenv("OLLAMA_MODEL_NAME")
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model = os.getenv("GOOGLE_GEMINI_MODEL_NAME")

# Database connection
db = SQLDatabase.from_uri(db_uri)

def get_llm():
    """Get the appropriate LLM based on available API keys."""
    if google_api_key:
        print("🧠 Using Gemini model")
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            api_key=google_api_key,
            temperature=0.0,
        )
    else:
        print("🧠 Using Ollama model")
        return ChatOllama(
            base_url=ollama_url,
            model=ollama_model,
            temperature=0,
            num_predict=512,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
        )

llm = get_llm()

def get_relevant_tables(question: str) -> List[str]:
    """Get relevant table names based on the question using keyword matching."""
    print(f"🔍 Analyzing question for relevant tables: {question}")
    keywords = [word.lower() for word in question.split()]
    print(f"🔍 Keywords extracted: {keywords}")
    
    all_tables = db.get_usable_table_names()
    print(f"🔍 All available tables: {len(all_tables)} total")
    
    # First pass: Look for exact matches or highly relevant tables
    exact_matches = []
    high_priority_matches = []
    partial_matches = []
    
    for table in all_tables:
        table_lower = table.lower()
        
        # Check for exact table name in question
        if any(table_lower == keyword for keyword in keywords):
            exact_matches.append(table)
            print(f"🎯 Exact match found: {table}")
        # Check for obvious main entity tables (customer, user, vendor, etc.)
        elif any(keyword in ['customer', 'customers'] for keyword in keywords):
            if table_lower == 'customer' or table_lower == 'users' or table_lower == 'vendor_list':
                high_priority_matches.append(table)
                print(f"🎯 High-priority entity match: {table}")
        elif any(keyword in ['user', 'users'] for keyword in keywords):
            if table_lower == 'users' or table_lower == 'customer':
                high_priority_matches.append(table)
                print(f"🎯 High-priority entity match: {table}")
        elif any(keyword in ['vendor', 'vendors'] for keyword in keywords):
            if table_lower == 'vendor_list':
                high_priority_matches.append(table)
                print(f"🎯 High-priority entity match: {table}")
        # Check for partial matches but be more selective
        elif any(keyword in table_lower for keyword in keywords):
            # Prioritize simpler table names (fewer underscores = more likely to be main tables)
            if table_lower.count('_') <= 2:
                partial_matches.append(table)
                print(f"🎯 Partial match: {table}")
    
    # Use exact matches first, then high priority, then partial matches
    relevant_tables = exact_matches or high_priority_matches or partial_matches[:3]
    
    # Final fallback: if still no matches, use some common tables
    if not relevant_tables:
        common_tables = ['customer', 'users', 'vendor_list', 'dsr_request', 'assessments']
        relevant_tables = [t for t in common_tables if t in all_tables][:2]
        print(f"⚠️ No keyword matches, using common tables: {relevant_tables}")
    
    print(f"✅ Selected relevant tables: {relevant_tables}")
    return relevant_tables

def format_table_info(tables: List[str]) -> str:
    """Format table information for the prompt."""
    print(f"📋 Formatting table info for tables: {tables}")
    if not tables:
        print("⚠️ No tables provided for formatting")
        return ""
    
    table_info_parts = []
    for table in tables:
        try:
            info = db.get_table_info([table])
            table_info_parts.append(info)
            print(f"✅ Got schema info for table: {table}")
            print(f"📄 Schema preview: {info[:200]}..." if len(info) > 200 else f"📄 Schema: {info}")
        except Exception as e:
            print(f"❌ Warning: Could not get info for table {table}: {e}")
            continue
    
    formatted_info = "\n\n".join(table_info_parts)
    print(f"📋 Total schema length: {len(formatted_info)} characters")
    return formatted_info

def clean_sql_query(query: str) -> str:
    """Clean SQL query by removing any prefixes and formatting issues."""
    print(f"🧹 Raw query input: {repr(query)}")
    
    if not isinstance(query, str):
        if isinstance(query, dict):
            query = query.get('content') or query.get('text') or str(query)
        elif hasattr(query, 'content'):
            query = query.content
        else:
            query = str(query)
    
    original_query = query
    
    # First, try to extract SQL from code blocks
    sql_block_pattern = r'```sql\s*(.*?)```'
    sql_block_match = re.search(sql_block_pattern, query, re.DOTALL | re.IGNORECASE)
    if sql_block_match:
        query = sql_block_match.group(1).strip()
        print(f"🔍 Extracted from SQL code block: {repr(query)}")
    else:
        # If no code block, try to extract SQL statement patterns
        # Look for SELECT, INSERT, UPDATE, DELETE statements
        sql_pattern = r'(SELECT\s+.*?)(?=\n\n|$|This|The|Please note)'
        sql_match = re.search(sql_pattern, query, re.DOTALL | re.IGNORECASE)
        if sql_match:
            query = sql_match.group(1).strip()
            print(f"🔍 Extracted using SQL pattern: {repr(query)}")
        else:
            # Fallback: remove common explanatory text patterns
            # Remove everything before the actual SQL keywords
            before_clean = query
            query = re.sub(r'^.*?(?=SELECT|INSERT|UPDATE|DELETE)', '', query, flags=re.DOTALL | re.IGNORECASE)
            if query != before_clean:
                print(f"🔍 Removed prefix text: {repr(query)}")
    
    # Remove common prefixes that LLMs sometimes add
    query = re.sub(r'^(sql|sqlquery|query):\s*', '', query.strip(), flags=re.IGNORECASE)
    query = re.sub(r'^```sql\s*', '', query, flags=re.IGNORECASE)
    query = re.sub(r'```\s*$', '', query)
    
    # Remove explanatory text after the query
    query = re.sub(r'\n\n.*?(?:This|The|Please note).*$', '', query, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up whitespace and remove trailing semicolon
    query = query.strip()
    if query.endswith(';'):
        query = query[:-1]
    
    # Remove any remaining newlines within the query for cleaner execution
    query = ' '.join(query.split())
    
    print(f"🧹 Cleaned query: {repr(query)}")
    
    # Validate that we have a proper SQL query
    if not query or not re.match(r'^(SELECT|INSERT|UPDATE|DELETE)', query, re.IGNORECASE):
        print(f"⚠️ Warning: Cleaned query doesn't look like valid SQL!")
        print(f"⚠️ Original: {repr(original_query)}")
        print(f"⚠️ Cleaned: {repr(query)}")
    
    return query

def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    """Execute SQL query and return results as a list of dictionaries."""
    try:
        print(f"🔧 Raw query before cleaning: {repr(query)}")
        
        # Clean the query
        query = clean_sql_query(query)
        print(f"🔧 Cleaned query: {repr(query)}")
        
        # Add LIMIT if not present
        if 'limit' not in query.lower():
            query += " LIMIT 10"
            
        print(f"🔧 Final query to execute: {repr(query)}")
        
        # Execute query and get column names
        with db._engine.connect() as conn:
            result = conn.execute(text(query))
            columns = list(result.keys())
            rows = result.fetchall()
        
        # Convert to list of dictionaries
        results = []
        for row in rows:
            row_dict = {col: val for col, val in zip(columns, row)}
            results.append(row_dict)
        
        return results
    
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return []

def format_results_as_markdown(results: List[Dict[str, Any]]) -> str:
    """Format SQL results as human-readable markdown."""
    if not results:
        return "No results found."
    
    # Convert to JSON for the LLM to format
    results_json = json.dumps(results, indent=2, default=str)
    return results_json

# Create a custom prompt for better SQL generation
custom_sql_prompt = ChatPromptTemplate.from_template("""
You are a PostgreSQL expert. Given the following database schema and a question, write ONLY the SQL query to answer the question.

Database Schema:
{table_info}

IMPORTANT INSTRUCTIONS:
- Write ONLY the SQL query, no explanations or comments
- Do not include any text before or after the SQL
- Do not wrap in code blocks or markdown
- Use proper PostgreSQL syntax
- Pay careful attention to column names - use EXACTLY as shown in the schema (including quotes if present)
- For timestamp columns, look for "createdAt", "updatedAt" etc. (camelCase with quotes)
- Limit results to {top_k} rows if no specific limit is mentioned
- Use double quotes for column names that contain special characters or mixed case

Question: {input}

SQL Query:""")

# Create the SQL query generation chain with custom prompt
sql_query_chain = create_sql_query_chain(llm, db, prompt=custom_sql_prompt)

# Create the results formatting chain
results_formatting_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that converts database query results into clear, human-readable markdown format.

Here are the database query results:
{results}

Please provide a comprehensive, well-formatted markdown summary of these results. Include:
1. A brief overview of what the data shows
2. Key insights or patterns
3. A nicely formatted table or list of the results
4. Any relevant observations about the data

Make it easy to understand for someone who didn't see the original SQL query.
""")

results_formatting_chain = (
    results_formatting_prompt 
    | llm 
    | StrOutputParser()
)

def create_sql_qa_chain():
    """Create the complete SQL Q&A chain using LCEL."""
    
    # Step 1: Get relevant tables and create the modified SQL chain
    enhanced_sql_chain = (
        RunnablePassthrough.assign(
            table_names_to_use=lambda x: get_relevant_tables(x["question"])
        )
        | RunnablePassthrough.assign(
            table_info=lambda x: format_table_info(x["table_names_to_use"])
        )
        | sql_query_chain
    )
    
    # Step 2: Execute the SQL query
    sql_with_execution = (
        RunnablePassthrough.assign(
            sql_query=enhanced_sql_chain
        )
        | RunnablePassthrough.assign(
            results=lambda x: execute_sql_query(x["sql_query"])
        )
    )
    
    # Step 3: Format results as markdown
    final_chain = (
        sql_with_execution
        | RunnablePassthrough.assign(
            formatted_results=lambda x: format_results_as_markdown(x["results"])
        )
        | RunnablePassthrough.assign(
            final_answer=lambda x: results_formatting_chain.invoke({
                "results": x["formatted_results"]
            })
        )
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "sql_query": clean_sql_query(x["sql_query"]),
            "results": x["results"],
            "answer": x["final_answer"]
        })
    )
    
    return final_chain

def ask_question(question: str) -> Dict[str, Any]:
    """Ask a question and get a formatted answer using the SQL Q&A chain."""
    print(f"\n🔍 Question: {question}")
    
    try:
        chain = create_sql_qa_chain()
        result = chain.invoke({"question": question})
        
        print(f"\n� Generated SQL: {result['sql_query']}")
        print(f"\n📊 Found {len(result['results'])} results")
        print(f"\n🗣️ Answer:\n{result['answer']}")
        
        return result
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            "question": question,
            "sql_query": None,
            "results": [],
            "answer": f"Sorry, I encountered an error: {e}",
            "error": str(e)
        }

if __name__ == "__main__":
    # Example usage
    # question = "List down all the open DSR requests"
    question = "What are the recent data breaches reported?"
    # question= "List down the last 5 customers added to the system"
    result = ask_question(question)
