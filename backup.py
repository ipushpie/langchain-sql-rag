from dotenv import load_dotenv
import os

load_dotenv()

db_uri = os.getenv("DATABASE_URL")
ollama_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("OLLAMA_MODEL_NAME")

print(f"DB_URI={db_uri}")
print(f"OLLAMA_URL={ollama_url}")
print(f"OLLAMA_MODEL={model_name}")

from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

# Initialize LLM
llm = OllamaLLM(model=model_name, base_url=ollama_url, temperature=0)

# Test streaming output
for chunk in llm.stream("Hi how are you?"):
    print(chunk, end="")

# Connect to the DB
db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)

print("Usable table names:", db.get_usable_table_names())
print("Table info:\n", db.table_info)

# Create the SQL agent
agent_executor = create_sql_agent(
    llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = agent_executor.invoke("List down the last 5 customers added to the system")
print("Query Result:", result)
