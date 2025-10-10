import json
import re

def llm_result_parser(model_text: str) -> str:
    txt = model_text.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", txt).strip()
    try:
        obj = json.loads(txt)
        return obj
    except json.JSONDecodeError:
        return txt
