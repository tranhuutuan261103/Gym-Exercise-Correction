from services.db import get_all_docs
from services.db import db, bucket
from firebase_admin import messaging
from google.cloud import firestore
import time as Time

def get_histories():
    # Get all histories from the database acsending by time
    result = []
    ref = db.collection("Histories").order_by("Datetime", direction=firestore.Query.DESCENDING).stream()
    for doc in ref:
        result.append(doc.to_dict())
    return result

def create_history(history):
    # create a new history in the database
    try:
        history["Duration"] = 0
        history["ErrorTotalCount"] = 0
        history["SpecificErrorFrames"] = []
        result = db.collection("Histories").add(history)
    except Exception as e:
        # if there is an error, save the error to the database
        print(e)
        return None
    # return id of the new history
    return result[1].id

def save_error(error, image_data, history_id):
    try:
        blob = bucket.blob(f"ErrorImages/{history_id}/{Time.time()}{error}.png")
        blob.upload_from_string(image_data, content_type="image/png")
        blob.make_public()

        # get the history document from the database
        ref = db.collection("Histories").document(history_id)
        history = ref.get().to_dict()
        # update the history document with the new error
        is_exist = False
        for item in history["SpecificErrorFrames"]:
            if item["ErrorType"] == error["ErrorType"]:
                item["Count"] += 1
                item["ImageUrl"].append(blob.public_url)
                is_exist = True

        if not is_exist:
            history["SpecificErrorFrames"].append({
                "ErrorType": error["ErrorType"],
                "Count": 1,
                "ImageUrl": [blob.public_url]
            })
        ref.update({
            "ErrorTotalCount": history["ErrorTotalCount"] + 1,
            "SpecificErrorFrames": history["SpecificErrorFrames"]
        })
    except Exception as e:
        # if there is an error, save the error to the database
        print(f'Error here {e}')
        return None
    return True

def send_push_notification(title, body):
    # Tạo thông điệp push notification
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        data={
            'title': title,
            'body': body,
        },
        topic='54U9rc8mD9Nbm4dpRAUNNm7ZYGw2',
    )

    # Gửi thông điệp
    response = messaging.send(message)
    print("Successfully sent message:", response)