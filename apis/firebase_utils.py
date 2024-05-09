from firebase_admin import credentials, firestore, initialize_app, get_app, storage
from firebase_admin.auth import create_custom_token
import os
import socket
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
        initialize_app(
            cred,
            {
                "storageBucket": os.getenv("STORAGE_BUCKET"),
                "databaseURL": os.getenv("DATABASE_URL"),
            },
        )
        print("Initialize: Firebase app has been initialized.")
    return firestore.client()


def upload_file_to_fire_storage(file_path, file_name="", bucket_name="OfflineVideos"):
    bucket = storage.bucket()
    if file_name == "":
        file_name = os.path.basename(file_path)
    blob_name = f"{bucket_name}/{file_name}"
    blob = bucket.blob(blob_name)

    # Tạo access token
    token = create_custom_token("<uid>")

    # Thêm token vào header của yêu cầu tải lên
    blob.metadata = {"customMetadata": {"FirebaseStorageDownloadTokens": token}}
    blob.upload_from_filename(file_path)
    blob.make_public()

    print(f"Done upload {file_name} to {bucket_name}.")
    return blob.public_url


def save_server_ip():
    hostname = socket.gethostname()
    IP_address = socket.gethostbyname(hostname)

    # Lưu vào realtime database
    server_info_ref = (
        initialize_firestore().collection("ServerInfo").document("server_ip")
    )
    server_info_ref.set({"ip": IP_address})
    print("URL server: ", IP_address)
