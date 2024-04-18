import firebase_admin
from firebase_admin import credentials, firestore

# Fetch the service account key JSON file contents
cred = credentials.Certificate('./desktop_app/services/serviceAccountKey.json')

# Initialize the app with a service account, granting admin privileges
try:
    firebase_admin.initialize_app(cred)
except ValueError as e:
    print(f"Firebase initialization failed: {e}")

db = firestore.client()

# get all documents from a collection
def get_all_docs(collection):
    docs = db.collection(collection).stream()
    return docs

# get a document by id
def get_doc(collection, doc_id):
    doc = db.collection(collection).document(doc_id).get()
    return doc