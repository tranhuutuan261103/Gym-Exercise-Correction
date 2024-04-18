from services.db import get_all_docs

def get_introductions():
    result = []
    for doc in get_all_docs("Introductions"):
        result.append(doc.to_dict())
    return result