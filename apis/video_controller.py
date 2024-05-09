from flask import Blueprint, jsonify, request
from video_model import VideoModel
from utils import create_thread_and_start

video_bp = Blueprint("video", __name__, url_prefix="/api/video")

video_model = VideoModel()


def allowed_file(filename) -> bool:
    # Các loại file cho phép
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@video_bp.route("upload_chunks/<exercise_type>", methods=["POST"])
def upload_video_chunks(exercise_type):
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        user_id = request.form.get("userId")
        formatted_time = request.form.get("formattedTime")
        chunk_index = request.form.get("chunkIndex")
        total_chunks = request.form.get("totalChunks")
        chunk = request.files["file"]

        # Kiểm tra xem chunk có tồn tại không
        if chunk is None:
            return jsonify({"error": "No file provided"}), 400
        chunk = chunk.read()
        return video_model.handle_upload_chunks(
            user_id, exercise_type, chunk, chunk_index, total_chunks, formatted_time
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@video_bp.route("upload/<exercise_type>", methods=["POST"])
def handle_uploaded_video(exercise_type):
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    user_id = request.form.get("userId")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "File type not allowed"}), 400

    if exercise_type not in video_model.available_models:
        return jsonify({"error": "Exercise type not found"}), 400

    record_id = video_model.init_empty_handle_video_record(user_id, exercise_type)
    
    # Lưu tạm file video xuống server để thread xử lí ngầm sau đó
    uploaded_video_url = video_model.save_file_temporarily(file, user_id)

    create_thread_and_start(
        video_model.handle_uploaded_video,
        args=(record_id, user_id, exercise_type, uploaded_video_url),
    )

    return (
        jsonify({"record_id": record_id}),
        200,
    )
