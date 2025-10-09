import re

database_keywords = {
    "dspm": [
        "dpsm", "pii", "piis", "policy", "policies", "file", "files",
        "access control", "access controls", "asset", "assets", 
        "metadata", "ocr", "aws", "profile", "profiles", "table", 
        "s3", "rds", "structured", "unstructured",
        "aws", "azure", "onedrive"
    ],
    "non-dspm" : [
        "cookie", "cookies", "consent", "consents", "ropa", "data breach", "dpia", "dpias", "vrm", "data retention"
    ]
}

def database_dd(question: str, keywords_map=database_keywords) -> str:
    q = question.lower()
    def contains_any(keywords):
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw.lower()) + r"\b", q):
                return True
        return False

    has_dspm = contains_any(keywords_map["dspm"])
    has_non_dspm = contains_any(keywords_map["non-dspm"])
    return True if has_dspm and not has_non_dspm else False
