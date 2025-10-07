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

def extract_foreign_key_relationships(schema_info: str) -> Dict[str, List[str]]:
    """Extract foreign key relationships from schema information."""
    relationships = {}
    
    # Pattern to match FOREIGN KEY constraints
    fk_pattern = r'CONSTRAINT\s+\w+\s+FOREIGN KEY\s*\((\w+)\)\s+REFERENCES\s+(\w+)\s*\((\w+)\)'
    
    matches = re.findall(fk_pattern, schema_info, re.IGNORECASE)
    
    for local_column, referenced_table, referenced_column in matches:
        if local_column not in relationships:
            relationships[local_column] = []
        relationships[local_column].append({
            'table': referenced_table,
            'column': referenced_column
        })
    
    return relationships

def get_relevant_tables(question: str) -> List[str]:
    """Get relevant table names based on the question using intelligent schema analysis."""
    print(f"🔍 Analyzing question for relevant tables: {question}")
    
    all_tables = db.get_usable_table_names()
    print(f"🔍 All available tables: {len(all_tables)} total")
    
    # Create an enhanced prompt that considers schema relationships
    table_selection_prompt = f"""
Given this question: "{question}"

And these available database tables: {', '.join(all_tables)}

Your task is to select the most relevant tables that would contain the data needed to answer this question.

Consider:
1. Primary tables that directly contain the main entities mentioned in the question
2. Related lookup tables that contain descriptive names for IDs (like customer names, user names, etc.)
3. Tables with foreign key relationships that would need to be joined for meaningful results

For example:
- If asking about "ROPAs", you might need both "ropa_register_of_proccessing_activity" and "customer" tables
- If asking about "customers", you might need "customer" and potentially "users" tables
- If asking about "departments", you might need "departments", "customer", and "users" tables

Respond with table names separated by commas. Include both main tables and related tables needed for JOINs.
Example: ropa_register_of_proccessing_activity, customer, ropa_master_data_subject_categories
"""
    
    try:
        # Use the LLM to select relevant tables
        response = llm.invoke(table_selection_prompt)
        
        # Extract table names from response
        if hasattr(response, 'content'):
            table_names_str = response.content.strip()
        else:
            table_names_str = str(response).strip()
        
        print(f"🤖 LLM suggested tables: {table_names_str}")
        
        # Parse the response to get individual table names
        suggested_tables = [name.strip() for name in table_names_str.split(',')]
        
        # Validate that suggested tables exist in our database
        relevant_tables = []
        for table in suggested_tables:
            table_clean = table.strip()
            if table_clean in all_tables:
                relevant_tables.append(table_clean)
                print(f"✅ Validated table: {table_clean}")
            else:
                print(f"⚠️ Suggested table not found: {table_clean}")
        
        # If LLM suggestions are invalid, fall back to keyword matching
        if not relevant_tables:
            print("🔄 LLM suggestions invalid, falling back to keyword matching...")
            relevant_tables = fallback_table_selection(question, all_tables)
        
    except Exception as e:
        print(f"❌ Error with LLM table selection: {e}")
        print("🔄 Falling back to keyword matching...")
        relevant_tables = fallback_table_selection(question, all_tables)
    
    print(f"✅ Selected relevant tables: {relevant_tables}")
    return relevant_tables

def fallback_table_selection(question: str, all_tables: List[str]) -> List[str]:
    """Fallback method for table selection using keyword matching."""
    keywords = [word.lower() for word in question.split()]
    print(f"🔍 Keywords extracted: {keywords}")
    
    # Score tables based on relevance
    table_scores = {}
    
    for table in all_tables:
        table_lower = table.lower()
        score = 0
        
        # Exact match gets highest score
        if any(table_lower == keyword for keyword in keywords):
            score += 100
        
        # Partial match in table name
        for keyword in keywords:
            if keyword in table_lower:
                score += 10
        
        # Penalty for complex table names (lots of underscores)
        score -= table_lower.count('_') * 2
        
        # Bonus for common entity tables
        if any(entity in table_lower for entity in ['user', 'customer', 'request', 'assessment', 'document']):
            score += 5
        
        if score > 0:
            table_scores[table] = score
    
    # Sort by score and take top 3
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    relevant_tables = [table for table, score in sorted_tables[:3]]
    
    for table, score in sorted_tables[:3]:
        print(f"🎯 Selected {table} (score: {score})")
    
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

# Create an enhanced custom prompt for intelligent SQL generation with JOINs
custom_sql_prompt = ChatPromptTemplate.from_template("""
You are a PostgreSQL expert. Given the following database schema and a question, write a SQL query that provides human-readable, meaningful results.

Database Schema:
{table_info}

CRITICAL INSTRUCTIONS FOR USER-FRIENDLY QUERIES:

1. ANALYZE THE SCHEMA to understand foreign key relationships:
   - Look for columns ending in '_id' that reference other tables
   - Find FOREIGN KEY constraints in the schema
   - Identify which tables contain descriptive names (like customer.name, users.firstName, etc.)

2. CREATE INTELLIGENT JOINs to show meaningful names instead of IDs:
   - JOIN tables to replace customer_id with customer names
   - JOIN tables to replace user IDs with user names (firstName + lastName)
   - JOIN tables to replace category/type IDs with their descriptive names
   - Use LEFT JOIN to avoid losing records when related data might be missing

3. SELECT MEANINGFUL COLUMNS:
   - Prioritize human-readable names over raw IDs
   - Use descriptive column aliases (e.g., customer.name AS customer_name)
   - Include the main data requested but enhance it with meaningful context

4. QUERY STRUCTURE:
   - Write ONLY the SQL query, no explanations or comments
   - Do not include any text before or after the SQL
   - Do not wrap in code blocks or markdown
   - Use proper PostgreSQL syntax with double quotes for mixed-case columns
   - Limit results to {top_k} rows if no specific limit is mentioned

EXAMPLE APPROACH:
Instead of: SELECT customer_id, title FROM table_name
Prefer: SELECT c.name AS customer_name, t.title, t.description 
        FROM table_name t 
        LEFT JOIN customer c ON t.customer_id = c.id

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

IMPORTANT PRESENTATION GUIDELINES:
- Focus on business-meaningful information, not technical database details
- If the data contains names instead of IDs, present them prominently
- Avoid mentioning database technical terms like "foreign keys" or "IDs"
- Present information in a way that business users would understand
- Use clear, descriptive language

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
    question = "List down the recent data breaches into the system"
    # question= "List down the last 5 customers added to the system"
    # question= "List down all the ROPAs where status is incomplete"
    result = ask_question(question)
