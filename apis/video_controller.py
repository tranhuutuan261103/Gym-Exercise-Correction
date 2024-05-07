from flask import Blueprint, jsonify, request
from video_model import VideoModel

video_bp = Blueprint("video", __name__, url_prefix="/api/video")

video_model = VideoModel()


def allowed_file(filename) -> bool:
    # Các loại file cho phép
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@video_bp.route("upload/<string:exercise_type>", methods=["POST"])
def handle_uploaded_video(exercise_type):
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    user_id = request.form.get("userId")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "File type not allowed"}), 400

    response_info = video_model.handle_uploaded_video(file, user_id)
    return response_info
