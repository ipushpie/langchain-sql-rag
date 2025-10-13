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

from app.utils.database import database_dd
from app.utils.helper import llm_result_parser
from app.config.logger import get_logger, log_with_emoji

# Initialize logger
logger = get_logger(__name__)

load_dotenv()

# Environment configuration
node_db_uri = os.getenv("NODE_DATABASE_URL")
dd_db_uri = os.getenv("DD_DATABASE_URL")
ollama_url = os.getenv("OLLAMA_BASE_URL")
ollama_model = os.getenv("OLLAMA_MODEL_NAME")
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model = os.getenv("GOOGLE_GEMINI_MODEL_NAME")

# Database connection
node_db = SQLDatabase.from_uri(node_db_uri)
dd_db = SQLDatabase.from_uri(dd_db_uri)


def get_llm():
    """Get the appropriate LLM based on available API keys."""
    if google_api_key:
        logger.info("Using Gemini model for LLM operations")
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            api_key=google_api_key,
            temperature=0.0,
        )
    else:
        logger.info("Using Ollama model for LLM operations")
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
    logger.debug(f"Analyzing question for relevant tables: {question}")
    
    all_tables = db.get_usable_table_names()
    logger.debug(f"All available tables: {len(all_tables)} total")
    
    # Create a concise table selection prompt
    table_selection_prompt = f"""
Question: {question}
Tables: {', '.join(all_tables)}

Select 3-5 most relevant tables (comma-separated, no explanations):"""
    
    try:
        # Use the LLM to select relevant tables
        response = llm.invoke(table_selection_prompt)
        
        # Extract table names from response
        if hasattr(response, 'content'):
            table_names_str = response.content.strip()
        else:
            table_names_str = str(response).strip()
        
        logger.debug(f"LLM table selection response: {repr(table_names_str)}")
        
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
                logger.debug(f"Validated table: {table_clean}")
            else:
                logger.warning(f"Suggested table not found: {table_clean}")
        
        # If LLM suggestions are invalid, fall back to keyword matching
        if not relevant_tables:
            logger.info("LLM suggestions invalid, falling back to keyword matching")
            relevant_tables = fallback_table_selection(question, all_tables)
        
    except Exception as e:
        logger.error(f"Error with LLM table selection: {e}")
        logger.info("Falling back to keyword matching")
        relevant_tables = fallback_table_selection(question, all_tables)
    
    logger.info(f"Selected relevant tables: {relevant_tables}")
    return relevant_tables

def fallback_table_selection(question: str, all_tables: List[str]) -> List[str]:
    """Fallback method for table selection using keyword matching."""
    keywords = [word.lower() for word in question.split()]
    logger.debug(f"Keywords extracted: {keywords}")
    
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
            logger.debug(f"Selected {table} (score: {score})")
    
    # If no tables found, try a broader search
    if not top_tables:
        logger.info("No high-scoring tables found, using broader search")
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
    logger.debug(f"Formatting table info for tables: {tables}")
    if not tables:
        logger.warning("No tables provided for formatting")
        return ""
    
    table_info_parts = []
    for table in tables:
        try:
            info = db.get_table_info([table])
            table_info_parts.append(info)
            logger.debug(f"Got schema info for table: {table}")
            logger.debug(f"Schema preview: {info[:200]}..." if len(info) > 200 else f"Schema: {info}")
        except Exception as e:
            logger.warning(f"Could not get info for table {table}: {e}")
            continue
    
    formatted_info = "\n\n".join(table_info_parts)
    logger.debug(f"Total schema length: {len(formatted_info)} characters")
    return formatted_info

def clean_sql_query(query: str) -> str:
    """Clean SQL query by removing any prefixes and formatting issues."""
    logger.debug(f"Raw query input: {repr(query)}")
    
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
        logger.debug(f"Extracted from SQL code block: {repr(query)}")
    else:
        # Look for complete SQL statements (including multi-line with FROM clauses)
        # Match from SELECT to the end, handling semicolons properly
        sql_pattern = r'(SELECT\s+.*?)(?=;?\s*$)'
        sql_match = re.search(sql_pattern, query, re.DOTALL | re.IGNORECASE)
        if sql_match:
            query = sql_match.group(1).strip()
            logger.debug(f"Extracted using complete SQL pattern: {repr(query)}")
        else:
            # Fallback: remove everything before SELECT but keep everything after
            before_clean = query
            query = re.sub(r'^.*?(?=SELECT|INSERT|UPDATE|DELETE)', '', query, flags=re.DOTALL | re.IGNORECASE)
            if query != before_clean:
                logger.debug(f"Removed prefix text, preserved full SQL: {repr(query[:100])}...")
    
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
    
    logger.debug(f"Cleaned query: {repr(query)}")
    
    # Validate that we have a proper SQL query with FROM clause
    if not query or not re.match(r'^(SELECT|INSERT|UPDATE|DELETE)', query, re.IGNORECASE):
        logger.warning("Cleaned query doesn't look like valid SQL!")
        logger.warning(f"Original: {repr(original_query)}")
        logger.warning(f"Cleaned: {repr(query)}")
        return original_query.strip()  # Return original if cleaning failed
    
    # Check if SELECT query has FROM clause
    if query.upper().startswith('SELECT') and 'FROM' not in query.upper():
        logger.warning("SELECT query missing FROM clause!")
        return original_query.strip()  # Return original if incomplete
    
    return query

def execute_sql_query(query: str, customer_id: int = None) -> List[Dict[str, Any]]:
    """Execute SQL query and return results as a list of dictionaries."""
    try:
        logger.debug(f"Raw query before cleaning: {repr(query)}")
        
        # Clean the query
        query = clean_sql_query(query)
        logger.debug(f"Cleaned query: {repr(query)}")
        
        # Validate the query
        try:
            query = validate_and_fix_sql(query)
        except ValueError as e:
            logger.error(f"SQL validation failed: {e}")
            return []
        
        # Add customer filter if customer_id is provided and not already in query
        if customer_id and 'customer_id' not in query.lower():
            # Try to add customer filter intelligently
            if 'WHERE' in query.upper():
                query = query + f" AND customer_id = {customer_id}"
            else:
                # Find the table alias or name to add WHERE clause
                if 'FROM' in query.upper():
                    query = query + f" WHERE customer_id = {customer_id}"
            logger.debug(f"Added customer filter: customer_id = {customer_id}")
        
        # Add LIMIT if not present
        if 'limit' not in query.lower():
            query += " LIMIT 10"
            
        logger.debug(f"Final query to execute: {repr(query)}")
        
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
        
        logger.info(f"SQL query executed successfully, returned {len(results)} rows")
        return results
    
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        return []

def format_results_as_markdown(results: List[Dict[str, Any]]) -> str:
    """Format SQL results as human-readable markdown."""
    if not results:
        return "No results found."
    
    # Convert to JSON for the LLM to format
    results_json = json.dumps(results, indent=2, default=str)
    return results_json

# Create a concise custom prompt for intelligent SQL generation
custom_sql_prompt = ChatPromptTemplate.from_template("""
Write a PostgreSQL query to answer the question using this schema:

{table_info}

Rules:
- Use exact column names from schema
- Quote mixed-case columns: "createdAt", "updatedAt"  
- Limit to {top_k} results
- Use LEFT JOIN for related data when helpful
- Order by timestamp DESC for recent data
{customer_filter_instruction}

Question: {input}""")

# Concise navigation selection prompt
navigation_selection_prompt = ChatPromptTemplate.from_template("""
Select the best route for this question and database tables:

Question: {question}
Tables: {tables}
Routes: {routes}

Choose exactly one route:""")

# Enhanced results formatting prompt for detailed answers
results_formatting_prompt = ChatPromptTemplate.from_template("""
Convert database results to detailed JSON response:

Question: {question}
Routes: {selected_routes}
Results: {results}

Format as JSON with:
- "answer": Detailed response with specific data (not just counts)
- "routes": First route from selected routes

For multiple results, show key details for each item.
For data breaches, include: title, status, date, description
For customers, include: name, status, contact info
For requests, include: type, status, date, requester info

Example for breaches:
"answer": "Found 5 data breaches:\\n\\n**Recent Breaches:**\\n- **new breach** (Status: OPEN) - Occurred: 2025-06-05, Discovered: 2025-06-05\\n- **Another breach** (Status: CLOSED) - Occurred: 2025-05-15, Discovered: 2025-05-16\\n\\nAll breaches require immediate attention for compliance review."

Output valid JSON only:""")

results_formatting_chain = (
    results_formatting_prompt 
    | llm 
    | StrOutputParser()
)

def validate_and_fix_sql(query: str) -> str:
    """Validate and fix common SQL issues for Ollama-generated queries."""
    logger.debug(f"Validating SQL: {repr(query[:100])}...")
    
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
        logger.warning("Unbalanced parentheses in query")
    
    # Check for proper quote matching
    double_quotes = query.count('"')
    if double_quotes % 2 != 0:
        logger.warning("Unmatched double quotes in query")
    
    logger.debug("SQL validation passed")
    return query

def create_sql_qa_chain(selected_db, customer_id=None):
    """Create the complete SQL Q&A chain using LCEL with customer filtering."""
    
    # Add customer filter instruction if customer_id is provided
    customer_filter_instruction = ""
    if customer_id:
        customer_filter_instruction = f"\n- IMPORTANT: Filter results for customer_id = {customer_id} using WHERE or JOIN conditions"
    
    # Create SQL query chain with customer filtering
    sql_query_chain = create_sql_query_chain(
        llm, 
        selected_db, 
        prompt=custom_sql_prompt.partial(customer_filter_instruction=customer_filter_instruction)
    )

    # Prepare schema context
    prepared = RunnablePassthrough.assign(
        table_names_to_use=lambda x: get_relevant_tables(x["question"])
    ).assign(
        table_info=lambda x: format_table_info(x["table_names_to_use"])
    )

    # Generate SQL
    with_sql = prepared.assign(
        sql_query=lambda x: sql_query_chain.invoke(x)
    )

    # Execute SQL with customer filtering
    executed = with_sql.assign(
        results=lambda x: execute_sql_query(x["sql_query"], customer_id)
    )

    # Directly get LLM-selected routes
    with_routes = executed.assign(
        selected_routes=lambda x: llm.invoke(
            navigation_selection_prompt.format(
                question=x["question"],
                tables=", ".join(x.get("table_names_to_use", [])) or "None",
                routes="\n".join(x["routes"])
            )
        ).content
    )

    # Format results
    final_chain = (
        with_routes
        | RunnablePassthrough.assign(
            formatted_results=lambda x: format_results_as_markdown(x["results"])
        )
        | RunnablePassthrough.assign(
            final_answer=lambda x: results_formatting_chain.invoke({
                "question": x["question"],
                "selected_routes": x.get("selected_routes", ""),
                "results": x["formatted_results"],
            })
        )
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "sql_query": clean_sql_query(x["sql_query"]),
            "selected_routes": x.get("selected_routes", ""),
            "results": x["results"],
            "answer": x["final_answer"],
        })
    )
    return final_chain

def is_greeting_or_general(question: str) -> bool:
    """Check if the question is a greeting or general conversation."""
    question_lower = question.lower().strip()
    
    # Common greeting patterns
    greeting_patterns = [
        r'^(hi|hello|hey|good morning|good afternoon|good evening)',
        r'^(how are you|how\'s it going|what\'s up)',
        r'^(thanks|thank you|bye|goodbye|see you)',
        r'^(who are you|what can you do|help me)',
        r'^(how do you work|what is this|explain)'
    ]
    
    # Check if it's a greeting
    for pattern in greeting_patterns:
        if re.match(pattern, question_lower):
            return True
    
    # Check if question is too short or doesn't seem database-related
    if len(question_lower.split()) <= 2 and not any(keyword in question_lower for keyword in 
        ['show', 'list', 'find', 'get', 'search', 'count', 'total', 'recent', 'last', 'first', 'latest']):
        return True
    
    return False

def generate_greeting_response(question: str, navigation_routes: List[str], customer_id: int = None) -> Dict[str, Any]:
    """Generate a friendly greeting response with helpful information."""
    question_lower = question.lower().strip()
    
    customer_context = f" for customer {customer_id}" if customer_id else ""
    
    # Personalized greeting responses
    if any(greeting in question_lower for greeting in ['hi', 'hello', 'hey']):
        answer = f"Hello! I'm your AI assistant. I can help you find information from your database{customer_context}. You can ask me questions like 'Show me recent data breaches' or 'List all customers'."
    elif any(phrase in question_lower for phrase in ['how are you', 'how\'s it going']):
        answer = f"I'm doing great, thank you for asking! I'm here to help you query and analyze your data{customer_context}. What would you like to know?"
    elif any(phrase in question_lower for phrase in ['who are you', 'what can you do']):
        answer = f"I'm an AI assistant that can help you explore your database{customer_context}. I can answer questions about your data, generate reports, and help you find specific information. Try asking me about customers, requests, assessments, or any other data you're looking for."
    elif any(phrase in question_lower for phrase in ['thank', 'thanks']):
        answer = "You're welcome! Feel free to ask me any questions about your data whenever you need help."
    elif any(phrase in question_lower for phrase in ['bye', 'goodbye']):
        answer = "Goodbye! Come back anytime if you need help with your data queries."
    else:
        answer = f"Hello! I'm your AI data assistant. I can help you find information from your database{customer_context}. Try asking me specific questions about your data, and I'll do my best to help you!"
    
    # Return empty routes for greetings since no navigation suggestion is relevant
    return {
        "question": question,
        "sql_query": None,
        "results": [],
        "answer": {
            "answer": answer,
            "routes": ""
        }
    }

def ask_question(question: str, navigation_routes: List[str], customer_id: int = None) -> Dict[str, Any]:
    """Ask a question and get a formatted answer using the SQL Q&A chain."""
    logger.info(f"Processing question: {question}")
    if customer_id:
        logger.info(f"Customer ID: {customer_id}")
    
    # Check if it's a greeting or general conversation
    if is_greeting_or_general(question):
        logger.info("Detected greeting/general question, generating friendly response")
        return generate_greeting_response(question, navigation_routes, customer_id)
    
    try:
        selected_is_dd = database_dd(question)
        selected_db = dd_db if selected_is_dd else node_db
        global db
        db = selected_db

        logger.info(f"Selected database: {'dd_db' if selected_is_dd else 'node_db'}")

        chain = create_sql_qa_chain(selected_db, customer_id)
        # Provide routes to the chain input so selection can happen inside
        result = chain.invoke({
            "question": question,
            "routes": navigation_routes
        })

        logger.info(f"Generated SQL: {result['sql_query']}")
        logger.debug(f"Routes used: {result.get('selected_routes', [])}")
        logger.info(f"Found {len(result['results'])} results")
        logger.debug(f"Answer: {result.get('answer')}")
        
        # Parse the result
        parsed_answer = llm_result_parser(result.get("answer"))
        
        # If no results found, provide a helpful fallback
        if not result['results'] and isinstance(parsed_answer, dict):
            if parsed_answer.get("answer") in ["No results found for your query.", "No results found."]:
                customer_context = f" for customer {customer_id}" if customer_id else ""
                parsed_answer["answer"] = f"I couldn't find any specific data matching your question{customer_context}. This could mean the data doesn't exist, or you might want to try rephrasing your question. Feel free to ask about customers, requests, assessments, or other specific data you're looking for."
        
        result['answer'] = parsed_answer
        logger.info("Question processed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        customer_context = f" for customer {customer_id}" if customer_id else ""
        return {
            "question": question,
            "sql_query": None,
            "results": [],
            "answer": {
                "answer": f"I encountered an issue while processing your request{customer_context}. Please try rephrasing your question or ask about specific data like customers, requests, or assessments. I'm here to help!",
                "routes": ""
            },
            "error": str(e)
        }

