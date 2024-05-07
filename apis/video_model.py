import os
import threading
from flask import jsonify
from werkzeug.utils import secure_filename
from ML_models.plank.PlankModel import PlankModel
from utils import combine_frames_to_video, create_folder_if_not_exists
from datetime import datetime
from firebase_utils import initialize_firestore, upload_video_to_fire_storage
import time
import pytz

current_dir = os.path.dirname(os.path.realpath(__file__))
plank_model = PlankModel()
UPLOADED_VIDEOS = "uploaded_videos"
db = initialize_firestore()


class VideoModel:
    def handle_uploaded_video(self, file, user_id):
        filename = secure_filename(file.filename)
        current_dir = os.path.dirname(__file__)
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")

        # Tạo task để thêm video vào Firestore
        record_id = self.init_empty_handle_video_record(user_id)

        create_folder_if_not_exists(f"{current_dir}/{UPLOADED_VIDEOS}")
        uploaded_video_url = (
            f"{current_dir}/{UPLOADED_VIDEOS}/{user_id}_{formatted_time}_{filename}.mp4"
        )

        # Lưu video xuống thư mục uploaded_videos
        file.save(uploaded_video_url)

        response_info = plank_model.plank_detection_offline(uploaded_video_url)
        handled_video_url = combine_frames_to_video(user_id, response_info)

        video_thread = threading.Thread(
            target=self.process_video_and_upload,
            args=(record_id, uploaded_video_url, handled_video_url),
        )
        video_thread.start()

        return (
            jsonify({"message": "Upload successful"}),
            200,
        )

    def init_empty_handle_video_record(self, user_id):
        handled_offline_videos_ref = db.collection("HandledOfflineVideos").document()
        timezone = pytz.timezone('Asia/Ho_Chi_Minh')
        handled_offline_videos_ref.set({"UserId": user_id, "HandledVideoUrl": "", "UploadedAt": datetime.now(timezone)})
        return handled_offline_videos_ref.id

    def process_video_and_upload(
        self, record_id, uploaded_video_url, handled_video_url
    ):
        time_start = time.time()
        handled_video_public_url = upload_video_to_fire_storage(handled_video_url)

        os.remove(uploaded_video_url)
        os.remove(handled_video_url)

        # Update lại thông tin của record
        handled_offline_videos_ref = db.collection("HandledOfflineVideos").document(
            record_id
        )
        handled_offline_videos_ref.update({"HandledVideoUrl": handled_video_public_url})
        time_end = time.time()
        print(
            "Upload to fire storage done in: ",
            (time_end - time_start),
            "s",
        )
