from firebase_admin import credentials, firestore, initialize_app, get_app, storage
from firebase_admin.auth import create_custom_token
import os
from dotenv import load_dotenv

load_dotenv()


def initialize_firestore():
    try:
        # Kiểm tra xem ứng dụng Firebase đã được khởi tạo chưa
        get_app()
    except Exception as e:
        # Nếu chưa, khởi tạo nó
        current_dir = os.path.dirname(os.path.realpath(__file__))
        cred = credentials.Certificate(f"{current_dir}/key.json")
        initialize_app(cred, {"storageBucket": os.getenv("STORAGE_BUCKET")})
        print("Initialize: Firebase app has been initialized.")
    return firestore.client()


def upload_video_to_fire_storage(file_path):
    file_name = os.path.basename(file_path)
    bucket = storage.bucket()
    blob_name = f"OfflineVideos/{file_name}"
    blob = bucket.blob(blob_name)

    # Tạo access token
    token = create_custom_token("<uid>")

    # Thêm token vào header của yêu cầu tải lên
    blob.metadata = {"customMetadata": {"FirebaseStorageDownloadTokens": token}}
    blob.upload_from_filename(file_path)
    blob.make_public()
    return blob.public_url
