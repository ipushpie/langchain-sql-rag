import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_ollama import ChatOllama
from langchain.agents.agent import AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

# Load environment variables
load_dotenv()
db_uri = os.getenv("DATABASE_URL")
ollama_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("OLLAMA_MODEL_NAME")
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model_name = os.getenv("GOOGLE_GEMINI_MODEL_NAME")

# 1. Initialize the SQL database
db = SQLDatabase.from_uri(db_uri)
print("\n📦 Available Tables:")
print(db.get_usable_table_names())

# 2. Initialize the Ollama LLM

llm = ChatOllama(
    base_url=ollama_url,
    model=model_name,
    temperature=0,
)

# Initialize Google Gemini model (Gemini 2.5 or others)
# llm = ChatGoogleGenerativeAI(
#     model=gemini_model_name,
#     api_key=google_api_key,
# )

# 3. Create the SQL toolkit (this includes all the SQL tools like query, schema, etc.)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 4. Create the agent using the toolkit (this is the correct way!)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# 5. Ask a question
# question = "Which processing activities(ropa) are incomplete?"
question = "What are the recent risks registered in the system for customer_id 129?"
# question = "List down the last 5 customers added to the system"

# 6. Run the agent
try:
    response = agent_executor.invoke({"input": question})
    print("\n📊 Answer:")
    print(response["output"])
except Exception as e:
    print("❌ Error running the query:")
    print(e)
