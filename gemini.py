import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# llm = ChatOllama(
#     base_url=ollama_url,
#     model=model_name,
#     temperature=0,
# )

# Initialize Google Gemini model (Gemini 2.5 or others)
llm = ChatGoogleGenerativeAI(
    model=gemini_model_name,
    api_key=google_api_key,
)

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

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this list with your frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    user_input: str
    customer_id: int = None

def ask_question(user_input: str, customer_id: int = None):
    """
    This function takes a question as input and returns the answer from the agent.
    """
    try:
        final_question = user_input
        if customer_id is not None:
            final_question += f" for customer id {customer_id}"
        response = agent_executor.invoke({"input": final_question})
        print(f"Agent response: {response}")
        return response["output"]
    except Exception as e:
        return f"Error running the query: {e}"

@app.post("/ask")
def ask(query: Query):
    answer = ask_question(query.user_input, query.customer_id)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
