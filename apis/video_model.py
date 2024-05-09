import os, cv2, time, pytz
from flask import jsonify
from werkzeug.utils import secure_filename
from ML_models.plank.PlankModel import PlankModel
from datetime import datetime
from utils import (
    combine_frames_to_video,
    create_folder_if_not_exists,
    create_thread_and_start,
)
from firebase_utils import (
    initialize_firestore,
    upload_file_to_fire_storage,
)

current_dir = os.path.dirname(os.path.realpath(__file__))
UPLOADED_VIDEOS = "uploaded_videos"
db = initialize_firestore()

available_models = {
    "plank": PlankModel(),
}


class VideoModel:
    def handle_upload_chunks(
        self, user_id, exercise_type, chunk, chunk_index, total_chunks, formatted_time
    ):
        create_folder_if_not_exists(f"{current_dir}/{UPLOADED_VIDEOS}")
        uploaded_video_url = (
            f"{current_dir}/{UPLOADED_VIDEOS}/{user_id}_{formatted_time}.mp4"
        )
        with open(f"{uploaded_video_url}", "wb") as f:
            f.write(chunk)

        if int(chunk_index) == int(total_chunks) - 1:
            response_info = self.handle_uploaded_video(
                user_id, exercise_type, uploaded_video_url=uploaded_video_url
            )
            return response_info
        return jsonify({"message": f"Chunk {chunk_index} uploaded successfully"}), 200

    def handle_uploaded_video(
        self, user_id, exercise_type, file=None, uploaded_video_url=None
    ):
        """
        Xử lý video đã được upload lên server khi đã thu thập đủ các chunks
        """
        # Tạo task để thêm video vào Firestore
        record_id = self.init_empty_handle_video_record(user_id, exercise_type)

        if uploaded_video_url is None:
            filename = secure_filename(file.filename)
            current_dir = os.path.dirname(__file__)
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y%m%d%H%M%S")

            create_folder_if_not_exists(f"{current_dir}/{UPLOADED_VIDEOS}")
            uploaded_video_url = f"{current_dir}/{UPLOADED_VIDEOS}/{user_id}_{formatted_time}_{filename}.mp4"

            # Lưu video xuống thư mục uploaded_videos
            file.save(uploaded_video_url)

        if exercise_type in available_models:
            response_info = available_models.get(exercise_type).plank_detection_offline(
                uploaded_video_url
            )
            handled_video_url = combine_frames_to_video(user_id, response_info)

            thread_upload_video = create_thread_and_start(
                target=self.process_video_and_upload,
                args=(
                    record_id,
                    uploaded_video_url,
                    handled_video_url,
                ),
            )

            thread_upload_error_details = create_thread_and_start(
                target=self.upload_error_details,
                args=(record_id, response_info["error_details"]),
            )

            thread_upload_video.join()
            thread_upload_error_details.join()

            return (
                jsonify({"message": "Upload successful", "record_id": record_id}),
                200,
            )
        else:
            return jsonify({"error": "Exercise type not found"}), 400

    def init_empty_handle_video_record(self, user_id, exercise_type):
        handled_offline_videos_ref = db.collection("HandledOfflineVideos").document()
        timezone = pytz.timezone("Asia/Ho_Chi_Minh")
        handled_offline_videos_ref.set(
            {
                "UserId": user_id,
                "HandledVideoUrl": "",
                "UploadedAt": datetime.now(timezone),
                "ExerciseType": exercise_type,
            }
        )
        return handled_offline_videos_ref.id

    def process_video_and_upload(
        self, record_id, uploaded_video_url, handled_video_url
    ):
        print("Uploading video to fire storage...")
        time_start = time.time()
        handled_video_public_url = upload_file_to_fire_storage(
            handled_video_url, bucket_name="OfflineVideos"
        )

        os.remove(uploaded_video_url)
        os.remove(handled_video_url)

        # Update lại thông tin của record
        handled_offline_videos_ref = db.collection("HandledOfflineVideos").document(
            record_id
        )

        handled_offline_videos_ref.update({"HandledVideoUrl": handled_video_public_url})

        time_end = time.time()
        print(
            "Upload video to fire storage done in: ",
            (time_end - time_start),
            "s",
        )

    def upload_error_details(self, record_id, error_details):
        print("Uploading error details to fire storage...")
        time_start = time.time()
        for error_type, images in error_details.items():
            urls = []
            for idx, image in enumerate(images):
                try:
                    file_name = f"{record_id}_{error_type}_{idx}.jpg"

                    # Lưu tạm ảnh xuống thư mục temp
                    file_path = f"{current_dir}/results/{file_name}"
                    cv2.imwrite(file_path, image["frame"])

                    urls.append(
                        {
                            "url": upload_file_to_fire_storage(
                                file_path,
                                file_name=f"{record_id}/{error_type}_{idx}.jpg",
                                bucket_name="ErrorImages",
                            ),
                            "frame_in_seconds": image["frame_in_seconds"],
                        }
                    )
                except Exception as e:
                    print(e)
                finally:
                    os.remove(file_path)

            error_details[error_type] = urls

        # Update lại thông tin của record
        handled_offline_videos_ref = db.collection("HandledOfflineVideos").document(
            record_id
        )
        handled_offline_videos_ref.update({"ErrorDetails": error_details})

        time_end = time.time()
        print(
            "Upload error details to fire storage done in: ",
            (time_end - time_start),
            "s",
        )
        return error_details
