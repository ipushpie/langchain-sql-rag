# LangChain SQL MCP

This project integrates LangChain with SQL databases using Model Context Protocol (MCP). It provides a flexible interface for querying, managing, and interacting with SQL databases through LangChain's powerful language model capabilities.

## Features
- Connect to SQL databases (e.g., SQLite, PostgreSQL, MySQL)
- Use LangChain to generate and execute SQL queries
- Model Context Protocol (MCP) integration for context-aware operations
- Easily configurable via environment variables
- Extensible for custom workflows

## Requirements
- Python 3.8+
- pip
- Supported SQL database (SQLite by default)

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd langchain_sql_mcp
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the project root (see `.env.example` for template):

```
GOOGLE_API_KEY=your_google_api_key_here
DATABASE_URL=your_database_connection_string_here
OLLAMA_BASE_URL=http://localhost:11435
OLLAMA_MODEL_NAME=my_model_name_here
```

## Usage
Run the main script:
```bash
python main.py
```


## Example
The following example demonstrates how the code works:

```python
import os
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_ollama import OllamaLLM

load_dotenv()
db_uri = os.getenv("DATABASE_URL")
ollama_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("OLLAMA_MODEL_NAME")

db = SQLDatabase.from_uri(db_uri)
llm = OllamaLLM(
   base_url=ollama_url,
   model=model_name,
   temperature=0,
   handle_parsing_errors=True,
   system_prompt="You are an expert SQL assistant. When generating SQL queries, output only raw SQL. Do not use Markdown formatting or code fences."
)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
   llm=llm,
   toolkit=toolkit,
   verbose=True,
   handle_parsing_errors=True,
)
question = "list down my last 5 contracts into the system. it's stored in the ContractExtraction table."
response = agent_executor.invoke({"input": question})
print(response["output"])
```


## Environment Variables
- `GOOGLE_API_KEY`: (Optional) Google API key for integrations
- `DATABASE_URL`: Database connection string (e.g., PostgreSQL, SQLite)
- `OLLAMA_BASE_URL`: Base URL for the Ollama server
- `OLLAMA_MODEL_NAME`: Name of the Ollama model to use

Refer to `.env.example` for sample values and formatting.

## Project Structure
- `main.py`: Entry point
- `.env.example`: Example environment configuration
- `.gitignore`: Files and folders to ignore in git

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT
