import cv2, os
from datetime import datetime
import os

current_dir = os.path.dirname(os.path.realpath(__file__))


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def combine_frames_to_video(user_id, frames_info) -> str:
    """
    output_path: Đường dẫn đến video xử lý xong
    """
    image_width = frames_info["image_width"]
    image_height = frames_info["image_height"]
    frames = frames_info["frames"]

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")

    create_folder_if_not_exists(f"{current_dir}/results")
    output_path = f"{current_dir}/results/{user_id}_{formatted_time}.mp4"
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"H264"),
        15,
        (image_width, image_height),
    )

    # Ghi các frame vào video
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    return output_path