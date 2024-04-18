from services.db import get_all_docs

def get_histories():
    result = []
    for doc in get_all_docs("Histories"):
        # get the data and id of the document and append it to the result list
        data = doc.to_dict()
        data["id"] = doc.id
        result.append(data)
    return result