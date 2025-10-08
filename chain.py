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
    
    # Create a more structured prompt that forces clean output
    table_selection_prompt = f"""
You are a database expert. Given this question and list of tables, select ONLY the most relevant table names.

Question: "{question}"

Available tables:
{', '.join(all_tables)}

INSTRUCTIONS:
1. Select 3-5 most relevant tables for answering the question
2. Consider main data tables and related lookup tables for JOINs
3. Output ONLY table names separated by commas
4. NO explanations, NO bullet points, NO additional text
5. Example format: table1, table2, table3

Table names:"""
    
    try:
        # Use the LLM to select relevant tables
        response = llm.invoke(table_selection_prompt)
        
        # Extract table names from response
        if hasattr(response, 'content'):
            table_names_str = response.content.strip()
        else:
            table_names_str = str(response).strip()
        
        print(f"🤖 LLM response: {repr(table_names_str)}")
        
        # Clean the response - remove any explanations or formatting
        # Take only the first line if there are multiple lines
        first_line = table_names_str.split('\n')[0].strip()
        
        # Remove any bullet points, numbers, or common prefixes
        first_line = re.sub(r'^[-*•]\s*', '', first_line)
        first_line = re.sub(r'^\d+\.\s*', '', first_line)
        first_line = re.sub(r'^(tables?:?|relevant tables?:?)\s*', '', first_line, flags=re.IGNORECASE)
        
        # Parse the response to get individual table names
        suggested_tables = [name.strip() for name in first_line.split(',')]
        
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
    
    # Define entity mappings for better matching
    entity_mappings = {
        'breach': ['data_breach', 'breach'],
        'data breach': ['data_breach', 'breach'],
        'customer': ['customer'],
        'user': ['user', 'users'],
        'ropa': ['ropa'],
        'request': ['request'],
        'assessment': ['assessment'],
        'department': ['department'],
        'document': ['document'],
        'dsr': ['data_subject_request', 'dsr'],
        'privacy': ['privacy'],
        'incident': ['incident', 'breach']
    }
    
    for table in all_tables:
        table_lower = table.lower()
        score = 0
        
        # Check entity mappings first
        for keyword in keywords:
            if keyword in entity_mappings:
                for entity in entity_mappings[keyword]:
                    if entity in table_lower:
                        score += 50
        
        # Exact keyword matches
        for keyword in keywords:
            if keyword in table_lower:
                score += 20
        
        # Partial matches for compound words
        for keyword in keywords:
            if len(keyword) > 3:  # Only for meaningful keywords
                if any(keyword in part for part in table_lower.split('_')):
                    score += 10
        
        # Boost for common entity tables
        if any(entity in table_lower for entity in ['customer', 'user', 'request', 'assessment', 'document', 'breach']):
            score += 5
        
        # Penalty for overly complex table names
        score -= table_lower.count('_') * 1
        
        if score > 0:
            table_scores[table] = score
    
    # Sort by score and take top tables
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 3-5 tables based on scores
    top_tables = []
    for table, score in sorted_tables[:5]:
        if score >= 5:  # Minimum relevance threshold
            top_tables.append(table)
            print(f"🎯 Selected {table} (score: {score})")
    
    # If no tables found, try a broader search
    if not top_tables:
        print("🔄 No high-scoring tables found, using broader search...")
        # Look for any table containing any keyword
        for table in all_tables:
            table_lower = table.lower()
            if any(keyword in table_lower for keyword in keywords if len(keyword) > 2):
                top_tables.append(table)
                if len(top_tables) >= 3:
                    break
    
    return top_tables[:5]  # Limit to 5 tables maximum

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
        # Look for complete SQL statements (including multi-line with FROM clauses)
        # Match from SELECT to the end, handling semicolons properly
        sql_pattern = r'(SELECT\s+.*?)(?=;?\s*$)'
        sql_match = re.search(sql_pattern, query, re.DOTALL | re.IGNORECASE)
        if sql_match:
            query = sql_match.group(1).strip()
            print(f"🔍 Extracted using complete SQL pattern: {repr(query)}")
        else:
            # Fallback: remove everything before SELECT but keep everything after
            before_clean = query
            query = re.sub(r'^.*?(?=SELECT|INSERT|UPDATE|DELETE)', '', query, flags=re.DOTALL | re.IGNORECASE)
            if query != before_clean:
                print(f"🔍 Removed prefix text, preserved full SQL: {repr(query[:100])}...")
    
    # Remove common prefixes that LLMs sometimes add
    query = re.sub(r'^(sql|sqlquery|query):\s*', '', query.strip(), flags=re.IGNORECASE)
    query = re.sub(r'^```sql\s*', '', query, flags=re.IGNORECASE)
    query = re.sub(r'```\s*$', '', query)
    
    # Clean up whitespace but preserve line structure for complex queries
    lines = query.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('--'):  # Remove SQL comments
            cleaned_lines.append(line)
    
    # Join lines with spaces, but preserve some structure
    query = ' '.join(cleaned_lines)
    
    # Remove trailing semicolon for execution
    if query.endswith(';'):
        query = query[:-1]
    
    print(f"🧹 Cleaned query: {repr(query)}")
    
    # Validate that we have a proper SQL query with FROM clause
    if not query or not re.match(r'^(SELECT|INSERT|UPDATE|DELETE)', query, re.IGNORECASE):
        print(f"⚠️ Warning: Cleaned query doesn't look like valid SQL!")
        print(f"⚠️ Original: {repr(original_query)}")
        print(f"⚠️ Cleaned: {repr(query)}")
        return original_query.strip()  # Return original if cleaning failed
    
    # Check if SELECT query has FROM clause
    if query.upper().startswith('SELECT') and 'FROM' not in query.upper():
        print(f"⚠️ Warning: SELECT query missing FROM clause!")
        return original_query.strip()  # Return original if incomplete
    
    return query

def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    """Execute SQL query and return results as a list of dictionaries."""
    try:
        print(f"🔧 Raw query before cleaning: {repr(query)}")
        
        # Clean the query
        query = clean_sql_query(query)
        print(f"🔧 Cleaned query: {repr(query)}")
        
        # Validate the query
        try:
            query = validate_and_fix_sql(query)
        except ValueError as e:
            print(f"❌ SQL validation failed: {e}")
            return []
        
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
You are a PostgreSQL expert. Given the database schema below, write a syntactically correct PostgreSQL query to answer the question.

Database Schema:
{table_info}

CRITICAL INSTRUCTIONS:

1. RESPONSE FORMAT:
   - Write ONLY the SQL query, no explanations or comments
   - Do not wrap in code blocks or markdown 
   - Do not include any text before or after the SQL
   - End with a semicolon

2. COLUMN NAMES AND QUOTING:
   - Use EXACT column names as shown in the schema
   - Use double quotes for mixed-case columns (e.g., "createdAt", "updatedAt")
   - Pay careful attention to capitalization in column names
   - Never assume column names - only use what's shown in the schema

3. QUERY CONSTRUCTION:
   - Use proper PostgreSQL syntax
   - Query for at most {top_k} results using LIMIT clause
   - Never query for all columns - select only columns needed to answer the question
   - Be careful about which column is in which table
   - Use table aliases for cleaner queries

4. JOINS AND RELATIONSHIPS:
   - Look for foreign key relationships (columns ending in '_id')
   - Use LEFT JOIN to include related descriptive names when possible
   - Replace IDs with meaningful names when possible
   - Ensure all JOIN conditions are correct

5. ORDERING AND FILTERING:
   - For "recent" data, look for timestamp columns like "createdAt", created_at, date_created
   - Use DESC order for recent data
   - Pay attention to exact column name formatting with quotes if needed

Question: {input}""")

# Add a validation function for Ollama compatibility
def validate_and_fix_sql(query: str) -> str:
    """Validate and fix common SQL issues for Ollama-generated queries."""
    print(f"🔍 Validating SQL: {repr(query[:100])}...")
    
    # Basic syntax checks
    if not query.strip():
        raise ValueError("Empty SQL query")
    
    query_upper = query.upper()
    
    # Check for basic SQL structure
    if not query_upper.startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')):
        raise ValueError("Query must start with a valid SQL command")
    
    # For SELECT queries, ensure FROM clause exists
    if query_upper.startswith('SELECT') and 'FROM' not in query_upper:
        raise ValueError("SELECT query missing FROM clause")
    
    # Check for balanced parentheses
    if query.count('(') != query.count(')'):
        print("⚠️ Warning: Unbalanced parentheses in query")
    
    # Check for proper quote matching
    double_quotes = query.count('"')
    if double_quotes % 2 != 0:
        print("⚠️ Warning: Unmatched double quotes in query")
    
    print("✅ SQL validation passed")
    return query

# Create the SQL query generation chain with custom prompt
sql_query_chain = create_sql_query_chain(llm, db, prompt=custom_sql_prompt)

# Create the results formatting chain
results_formatting_prompt = ChatPromptTemplate.from_template("""
Convert the following database query results into a clean markdown table format.

Results:
{results}

INSTRUCTIONS:
- Create a simple markdown table with column headers
- Show all the data clearly in rows
- If no results, just say "No results found"
- Do not add explanations, insights, or observations
- Just present the data in a clean table format

Table:""")

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
    question = "List down all the open DSR requests"
    # question = "List down the recent data breaches into the system"
    # question= "List down the last 5 customers added to the system"
    # question= "List down all the ROPAs where status is incomplete"
    result = ask_question(question)
