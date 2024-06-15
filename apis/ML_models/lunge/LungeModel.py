import mediapipe as mp
import math
import numpy as np
import pandas as pd
import cv2
import copy
import warnings
import pickle
import os


class LungeModel:
    def __init__(self):
        warnings.filterwarnings("ignore")
        self.mp_drawing = mp.solutions.drawing_utils
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.mp_pose = mp.solutions.pose
        self.RF_model = self.load_model(f"{self.current_dir}\RF_model.pkl")
        self.input_scaler = self.load_model(f"{self.current_dir}\input_scaler.pkl")
        self.IMPORTANT_LMS = [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
            "LEFT_HEEL",
            "RIGHT_HEEL",
            "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX",
        ]
        self.HEADERS = ["label"]
        for landmark in self.IMPORTANT_LMS:
            for dim in ["x", "y", "z"]:
                self.HEADERS.append(f"{landmark.lower()}_{dim}")

    def load_model(self, file_name):
        with open(file_name, "rb") as file:
            model = pickle.load(file)
        return model

    def extract_and_recalculate_landmarks(self, pose_landmarks):
        """
        Tịnh tiến thân người vào giữa bức hình, đồng thời dời lại trục toạ độ
        """

        hip_center_x = float((pose_landmarks[23].x + pose_landmarks[24].x) / 2)
        hip_center_y = float((pose_landmarks[23].y + pose_landmarks[24].y) / 2)

        new_center = (0.5, 0.5)
        delta_x = new_center[0] - hip_center_x
        delta_y = new_center[1] - hip_center_y

        # Khởi tạo mảng NumPy với kích thước đã biết trước
        data = np.zeros((len(self.IMPORTANT_LMS), 3))

        for idx, landmark in enumerate(self.IMPORTANT_LMS):
            key_point_id = self.mp_pose.PoseLandmark[landmark].value
            key_point = pose_landmarks[key_point_id]
            data[idx] = [
                key_point.x + delta_x - 0.5,
                key_point.y + delta_y - 0.5,
                key_point.z,
            ]

        return data.flatten().tolist()

    def rescale_frame(self, frame, percent=50):
        """
        Rescale a frame to a certain percentage compare to its original frame
        """
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def get_class(self, encode_label: float):
        """
        Chuyển một label được encode thành class tương ứng
        """
        return {0: "Down", 1: "Middle", 2: "Stand"}.get(encode_label)

    def get_color_for_landmarks(self, errors):
        if errors == "None":
            return ((255, 165, 0), (255, 140, 0))
        else:
            return ((29, 62, 199), (1, 143, 241))

    def get_image_size(self, image):
        return image.shape[1], image.shape[0]

    def rescale_points_to_original(self, points, image_shape):
        return (points[0] * image_shape[0], points[1] * image_shape[1])

    def calculate_angle(self, point1, point2, point3):
        vector1 = (point1[0] - point2[0], point1[1] - point2[1])
        vector2 = (point3[0] - point2[0], point3[1] - point2[1])

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        sum_magnitude = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5 * (
            vector2[0] ** 2 + vector2[1] ** 2
        ) ** 0.5
        return math.degrees(math.acos(dot_product / sum_magnitude))

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

    def draw_knee_angle(self, mp_results, image):
        image_shape = (image.shape[1], image.shape[0])
        landmarks = mp_results.pose_landmarks.landmark
        mp_pose = self.mp_pose

        # Lấy toạ độ của 3 điểm cần thiết (hông, đầu gối, mắt cá chân) cho việc tính góc
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]

        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]

        # Tính góc tạo bởi 3 điểm trên ở bên trái và bên phải
        left_angle = self.calculate_angle(
            self.rescale_points_to_original(left_hip, image_shape),
            self.rescale_points_to_original(left_knee, image_shape),
            self.rescale_points_to_original(left_ankle, image_shape),
        )
        right_angle = self.calculate_angle(
            self.rescale_points_to_original(right_hip, image_shape),
            self.rescale_points_to_original(right_knee, image_shape),
            self.rescale_points_to_original(right_ankle, image_shape),
        )

        # Vẽ góc lên ảnh
        # trong đó 0.5 là font size, (255, 255, 255) là màu, 2 là độ dày của chữ
        cv2.putText(
            image,
            str(int(left_angle)),
            tuple(np.multiply(left_knee, image_shape).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            str(int(right_angle)),
            tuple(np.multiply(right_knee, image_shape).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def define_leg_front(self, knee_left, knee_right):
        """
        Xác định chân đưa lên trước, dựa vào vị trí của đầu gối trái và đầu gối phải
        """
        if knee_left[1] < knee_right[1]:
            return "left"
        else:
            return "right"

    def define_error(self, key_points, image_size):
        errors = []

        leg_front = self.define_leg_front(
            self.rescale_points_to_original(
                (key_points.left_knee_x[0], key_points.left_knee_y[0]), image_size
            ),
            self.rescale_points_to_original(
                (key_points.right_knee_x[0], key_points.right_knee_y[0]), image_size
            ),
        )

        if leg_front == "left":
            knee_angle_left = self.calculate_angle(
                self.rescale_points_to_original(
                    (key_points.left_hip_x[0], key_points.left_hip_y[0]), image_size
                ),
                self.rescale_points_to_original(
                    (key_points.left_knee_x[0], key_points.left_knee_y[0]), image_size
                ),
                self.rescale_points_to_original(
                    (key_points.left_ankle_x[0], key_points.left_ankle_y[0]), image_size
                ),
            )

            if knee_angle_left > 110 or knee_angle_left < 70:
                errors.append("left knee not square")

            leg_angle = self.angle_2_vector(
                np.array(
                    [
                        (key_points.left_knee_x[0] - key_points.left_ankle_x[0])
                        * image_size[0],
                        (key_points.left_knee_y[0] - key_points.left_ankle_y[0])
                        * image_size[1],
                    ]
                ),
                np.array([1, 0]),
            )

            if leg_angle < 75 or leg_angle > 105:
                errors.append("left leg not straight")
        else:
            knee_angle_right = self.calculate_angle(
                self.rescale_points_to_original(
                    (key_points.left_hip_x[0], key_points.right_hip_y[0]), image_size
                ),
                self.rescale_points_to_original(
                    (key_points.right_knee_x[0], key_points.right_knee_y[0]), image_size
                ),
                self.rescale_points_to_original(
                    (key_points.right_ankle_x[0], key_points.right_ankle_y[0]),
                    image_size,
                ),
            )

            if knee_angle_right > 110 or knee_angle_right < 70:
                errors.append("right knee not square")

            # Define errors for leg
            leg_angle = self.angle_2_vector(
                np.array(
                    [
                        (key_points.right_knee_x[0] - key_points.right_ankle_x[0])
                        * image_size[0],
                        (key_points.right_knee_y[0] - key_points.right_ankle_y[0])
                        * image_size[1],
                    ]
                ),
                np.array([1, 0]),
            )
            if leg_angle < 75 or leg_angle > 105:
                errors.append("right leg not straight")

        body = self.angle_2_vector(
            np.array(
                [
                    (key_points.left_shoulder_x[0] - key_points.left_hip_x[0])
                    * image_size[0],
                    (key_points.left_shoulder_y[0] - key_points.left_hip_y[0])
                    * image_size[1],
                ]
            ),
            np.array([1, 0]),
        )

        if body < 75 or body > 105:
            errors.append("body not straight")

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
        prediction_probability_max,
    ):
        """
        Draw information on the frame
        """
        cv2.rectangle(image, (0, 0), (750, 60), (245, 117, 16), -1)

        # Hiển thị lỗi sai
        cv2.putText(
            image,
            "REP",
            (15, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            str(counter),
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Display class
        cv2.putText(
            image,
            "STAGE",
            (100, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{predicted_stage}, {round(prediction_probability_max, 2)}",
            (60, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            "ERRORS",
            (255, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        if predicted_stage == "Down":
            if errors == "None":
                cv2.putText(
                    image,
                    "OK",
                    (270, 45),
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
        frame_skip = 3
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
                    self.draw_knee_angle(results, image)
                    key_points = self.extract_and_recalculate_landmarks(
                        results.pose_landmarks.landmark
                    )
                    X = pd.DataFrame([key_points], columns=self.HEADERS[1:])
                    X_original = copy.deepcopy(X)
                    X = self.input_scaler.transform(X)

                    predicted_stage = self.RF_model.predict(X)[0]
                    predicted_stage = self.get_class(predicted_stage)
                    prediction_probability_max = self.RF_model.predict_proba(X)[0].max()

                    if prediction_probability_max >= prediction_probability_threshold:
                        if predicted_stage == "Down" and current_stage == "Middle":
                            counter += 1
                        current_stage = predicted_stage

                    errors = "None"
                    if current_stage == "Down":
                        errors = self.define_error(
                            X_original, self.get_image_size(image)
                        )

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
                        prediction_probability_max,
                    )

                    # Lưu frame vào để phục vụ cho việc xuất video
                    result_frames.append(image)
                    current_time = frame_count / fps
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
