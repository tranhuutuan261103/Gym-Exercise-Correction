import mediapipe as mp
import math
import numpy as np
import pandas as pd
import cv2
import copy
import warnings
import pickle
import os


class BicepCurlModel:
    def __init__(self):
        warnings.filterwarnings("ignore")
        self.mp_drawing = mp.solutions.drawing_utils
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.mp_pose = mp.solutions.pose

    def rescale_frame(self, frame, percent=50):
        """
        Rescale a frame to a certain percentage compare to its original frame
        """
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def get_color_for_landmarks(self, errors):
        if errors == "None":
            return ((255, 165, 0), (255, 140, 0))
        else:
            return ((29, 62, 199), (1, 143, 241))

    def get_image_size(self, image):
        return image.shape[1], image.shape[0]

    def calculate_angle(self, a, b, c, size_of_image):
        # Lấy tọa độ của 3 điểm
        a = (a[0] * size_of_image[0], a[1] * size_of_image[1])
        b = (b[0] * size_of_image[0], b[1] * size_of_image[1])
        c = (c[0] * size_of_image[0], c[1] * size_of_image[1])

        # Tính góc giữa 3 điểm
        ba_vector = [a[0] - b[0], a[1] - b[1]]
        bc_vector = [c[0] - b[0], c[1] - b[1]]
        ba_length = math.sqrt(ba_vector[0] ** 2 + ba_vector[1] ** 2)
        bc_length = math.sqrt(bc_vector[0] ** 2 + bc_vector[1] ** 2)

        return math.degrees(
            math.acos(
                (ba_vector[0] * bc_vector[0] + ba_vector[1] * bc_vector[1])
                / (ba_length * bc_length)
            )
        )

    def angle_2_vector(self, v1, v2):
        """
        Tính góc giữa 2 vector
        """
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot / (norm_v1 * norm_v2)

        # convert to degree
        return np.arccos(cos_theta) * 180 / np.pi

    def determine_stage(self, key_points, image_size):
        # Dựa vào góc giữa khuỷu tay, cổ tay và vai để xác định giai đoạn của động tác Bicep Curl
        left_shoulder = [key_points[11].x, key_points[11].y]
        right_shoulder = [key_points[12].x, key_points[12].y]
        left_elbow = [key_points[13].x, key_points[13].y]
        right_elbow = [key_points[14].x, key_points[14].y]
        left_wrist = [key_points[15].x, key_points[15].y]
        right_wrist = [key_points[16].x, key_points[16].y]

        # Tính góc giữa 2 vector
        left_angle = self.calculate_angle(
            left_shoulder, left_elbow, left_wrist, image_size
        )
        right_angle = self.calculate_angle(
            right_shoulder, right_elbow, right_wrist, image_size
        )

        if abs(left_angle - right_angle) < 10:
            angle = max(left_angle, right_angle)
        else:
            angle = min(left_angle, right_angle)

        if angle > 160:
            return "Down"
        elif angle > 90:
            return "Middle"
        else:
            return "Up"

    def define_errors(self, key_points, image_size):
        errors = []
        left_shoulder = [key_points[11].x, key_points[11].y]
        right_shoulder = [key_points[12].x, key_points[12].y]
        left_hip = [key_points[23].x, key_points[23].y]
        right_hip = [key_points[24].x, key_points[24].y]
        left_elbow = [key_points[13].x, key_points[13].y]
        right_elbow = [key_points[14].x, key_points[14].y]
        left_wrist = [key_points[15].x, key_points[15].y]
        right_wrist = [key_points[16].x, key_points[16].y]
        left_index = [key_points[19].x, key_points[19].y]
        right_index = [key_points[20].x, key_points[20].y]

        # Góc giữa vector vai, hông và đầu gối
        left_body_angle = self.angle_2_vector(
            np.array(
                [
                    (left_shoulder[0] - left_hip[0]) * image_size[0],
                    (left_shoulder[1] - left_hip[1]) * image_size[1],
                ]
            ),
            np.array([1, 0]),
        )
        right_body_angle = self.angle_2_vector(
            np.array(
                [
                    (right_shoulder[0] - right_hip[0]) * image_size[0],
                    (right_shoulder[1] - right_hip[1]) * image_size[1],
                ]
            ),
            np.array([1, 0]),
        )
        angle = max(left_body_angle, right_body_angle)
        if angle < 80 or angle > 100:
            errors.append("body not straight")

        # Góc giữa vai, khuỷu tay và hông
        angle = self.calculate_angle(left_shoulder, left_hip, left_elbow, image_size)
        angle = max(
            angle,
            self.calculate_angle(right_shoulder, right_hip, right_elbow, image_size),
        )
        if angle > 25:
            errors.append("arm not straight")

        # Góc giữa khuỷu tay, cổ tay và ngón tay trỏ
        angle = self.calculate_angle(left_elbow, left_wrist, left_index, image_size)
        angle = 180 - max(
            angle,
            self.calculate_angle(right_elbow, right_wrist, right_index, image_size),
        )
        if angle >= 50:
            errors.append("wrist not straight")

        if errors == []:
            return "None"
        else:
            return ", ".join(errors)

    def draw_info_on_frame(
        self,
        image,
        predicted_stage,
        counter,
        errors,
    ):
        """
        Draw information on the frame
        """
        cv2.rectangle(image, (0, 0), (750, 60), (245, 117, 16), -1)

        # Hiển thị lỗi sai
        cv2.putText(
            image,
            "STAGE, REP",
            (30, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if predicted_stage == "Up":
            cv2.putText(
                image,
                f"{predicted_stage}, {counter}",
                (40, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                f"{predicted_stage}, {counter}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            image,
            "ERRORS",
            (220, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if errors == "None":
            cv2.putText(
                image,
                "OK",
                (250, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            errors_list = errors.split(", ")
            y_position = 45
            for error in errors_list:
                cv2.putText(
                    image,
                    error,
                    (200, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 150, 255),
                    2,
                    cv2.LINE_AA,
                )
                y_position += 30

    def detection_offline(self, video_path, prediction_probability_threshold=0.5):
        cap = cv2.VideoCapture(video_path if video_path else 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS: ", fps)

        current_stage = "Unknown"

        # Số frame được bỏ qua
        image_width, image_height = 0, 0
        frame_skip = 1
        frame_count = 0

        # Số rep tập được
        counter = 0

        result_frames = []
        error_details = {}

        # Đặt thời gian bắt đầu của video để lát tính thời gian tại thời điểm của từng frame
        previous_error = {"name": "Unknown", "time": 0}

        with self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    break

                frame_count += 1

                # Bỏ qua frame nếu không phải frame được xử lý
                if frame_count % frame_skip != 0:
                    continue

                print("Running, process in seconds: ", frame_count / fps)

                # resize frame để tăng tốc độ xử lý
                if image.shape[1] != 768:
                    image = self.rescale_frame(image, percent=60)
                image_width, image_height = self.get_image_size(image)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if not results.pose_landmarks:
                    print("No human found")
                    continue

                initial_pose_landmarks = copy.deepcopy(results.pose_landmarks)
                image.flags.writeable = True

                # Cần khôi phục lại màu gốc của ảnh
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Get landmarks
                try:
                    image_size = self.get_image_size(image)
                    errors = self.define_errors(
                        results.pose_landmarks.landmark, image_size
                    )
                    predicted_stage = self.determine_stage(
                        results.pose_landmarks.landmark, image_size
                    )

                    if current_stage == "Up" and predicted_stage == "Middle":
                        direction = "Down"
                    elif (
                        current_stage == "Middle"
                        and predicted_stage == "Down"
                        and direction == "Down"
                    ):
                        counter += 1
                        direction = "Up"
                    current_stage = predicted_stage

                    colors = self.get_color_for_landmarks(errors)
                    self.mp_drawing.draw_landmarks(
                        image,
                        initial_pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(
                            color=colors[0], thickness=2, circle_radius=2
                        ),
                        self.mp_drawing.DrawingSpec(
                            color=colors[1], thickness=2, circle_radius=1
                        ),
                    )

                    self.draw_info_on_frame(
                        image,
                        current_stage,
                        counter,
                        errors,
                    )

                    # Lưu frame vào để phục vụ cho việc xuất video
                    result_frames.append(image)
                    current_time = round(frame_count / fps, 1)
                    if errors != "None" and (
                        errors != previous_error["name"]
                        or current_time - previous_error["time"] >= 1.5
                    ):
                        error_details.setdefault(errors, []).append(
                            {
                                "frame": image,
                                "frame_in_seconds": current_time,
                            }
                        )
                        previous_error = {"name": errors, "time": current_time}

                except Exception as e:
                    print(f"Error: {e}")

                cv2.imshow("CV2", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

        response_info = {
            "frame_skip": frame_skip,
            "fps": fps,
            "frames": result_frames,
            "image_width": image_width,
            "image_height": image_height,
            "error_details": error_details,
        }
        print("Done detection")

        return response_info
