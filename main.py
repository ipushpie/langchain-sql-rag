import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()
db_uri = os.getenv("DATABASE_URL")
ollama_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("OLLAMA_MODEL_NAME")

# 1. Initialize the SQL database
db = SQLDatabase.from_uri(db_uri)
print("\n📦 Available Tables:")
print(db.get_usable_table_names())

# 2. Initialize the Ollama LLM

system_prompt = (
    "You are an expert SQL assistant. "
    "When generating SQL queries, output only raw SQL. "
    "Do not use Markdown formatting or code fences."
)
llm = OllamaLLM(
    base_url=ollama_url,
    model=model_name,
    temperature=0,
    handle_parsing_errors=True,
    system_prompt=system_prompt,  # Add this line if supported
    
)

# 3. Create the SQL toolkit (this includes all the SQL tools like query, schema, etc.)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 4. Create the agent using the toolkit (this is the correct way!)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True,
)

# 5. Ask a question
question = "list down my last 5 contracts into the system. it's stored in the ContractExtraction table."

# 6. Run the agent
try:
    response = agent_executor.invoke({"input": question})
    print("\n📊 Answer:")
    print(response["output"])
except Exception as e:
    print("❌ Error running the query:")
    print(e)
