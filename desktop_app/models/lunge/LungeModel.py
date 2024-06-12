import mediapipe as mp
import math
import numpy as np
import pandas as pd
import cv2
import copy
import warnings
import pickle
import os
import time as Time
import threading
from services.Histories import create_history, save_error
import datetime
import soundfile as sf
import sounddevice as sd

class LungeModel:
    def __init__(self):
        warnings.filterwarnings('ignore')
        self.last_time_skip = Time.time()
        self.thredhold_time_skip = 0.25 # Số s được bỏ qua
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.RF_model = self.load_model(f'{current_dir}/best_models/RF_model.pkl')
        self.input_scaler = self.load_model(f"{current_dir}/best_models/input_scaler.pkl")

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
            "RIGHT_FOOT_INDEX"
        ]
        self.HEADERS = ["label"]
        for landmark in self.IMPORTANT_LMS:
            for dim in ['x', 'y', 'z']:
                self.HEADERS.append(f"{landmark.lower()}_{dim}")

        self.last_class = "Unknown"
        self.last_errors = "Unknown"
        self.last_state = None
        self.last_predicted_stage = "Unknown"
        self.counter = 0
        self.last_prediction_probability_max = 0

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

        data = []
        for landmark in self.IMPORTANT_LMS:
            # Lấy ra id của key point trên cơ thể người
            key_point_id = self.mp_pose.PoseLandmark[landmark].value

            key_point = pose_landmarks[key_point_id]
            key_point.x += delta_x - 0.5
            key_point.y += delta_y -0.5
            data.append([key_point.x, key_point.y, key_point.z])

        return np.array(data).flatten().tolist()

    def extract_important_key_points(self, results) -> list:
        key_points = []

        for lm in self.IMPORTANT_LMS:
            # Lấy ra id của key point trên cơ thể người
            key_point_id = self.mp_pose.PoseLandmark[lm].value
            key_point = results.pose_landmarks.landmark[key_point_id]
            key_points.append([key_point.x, key_point.y, key_point.z])

        return np.array(key_points).flatten().tolist()

    def rescale_frame(self, frame, percent=50):
        '''
        Rescale a frame to a certain percentage compare to its original frame
        '''
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

    def get_class(self, encode_label: float):
        return {
            0: "Down",
            1: "Middle",
            2: "Up"
        }.get(encode_label, "Unknown")
    
    def extract_and_recalculate_landmarks(self, pose_landmarks):
        """
        Tịnh tiến thân người vào giữa bức hình, đồng thời dời lại trục toạ độ
        """
        hip_center_x = float((pose_landmarks[23].x + pose_landmarks[24].x) / 2)
        hip_center_y = float((pose_landmarks[23].y + pose_landmarks[24].y) / 2)

        new_center = (0.5, 0.5)
        delta_x = new_center[0] - hip_center_x
        delta_y = new_center[1] - hip_center_y

        data = []
        for landmark in self.IMPORTANT_LMS:
            # Lấy ra id của key point trên cơ thể người
            key_point_id = self.mp_pose.PoseLandmark[landmark].value

            key_point = pose_landmarks[key_point_id]
            key_point.x += delta_x - 0.5
            key_point.y += delta_y -0.5
            data.append([key_point.x, key_point.y, key_point.z])

        return np.array(data).flatten().tolist()
    
    def rescale_points_to_original(self, points, image_shape):
        return (points[0] * image_shape[0], points[1] * image_shape[1])
    
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
    
    def calculate_angle(self, point1, point2, point3):
        vector1 = (point1[0] - point2[0], point1[1] - point2[1])
        vector2 = (point3[0] - point2[0], point3[1] - point2[1])

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        sum_magnitude = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5 * (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5
        return math.degrees(math.acos(dot_product / sum_magnitude))
    
    def define_leg_front(self, knee_left, knee_right):
        """
        Xác định chân đưa lên trước, dựa vào vị trí của đầu gối trái và đầu gối phải
        """
        if knee_left[1] < knee_right[1]:
            return "left"
        else:
            return "right"
    
    def define_errors(self, key_points, image_size):
        errors = []

        leg_front = self.define_leg_front(self.rescale_points_to_original((key_points.left_knee_x[0], key_points.left_knee_y[0]), image_size),
                                        self.rescale_points_to_original((key_points.right_knee_x[0], key_points.right_knee_y[0]), image_size))
        
        if leg_front == "left":
            # Define errors for knee
            knee_angle_left = 180
            knee_angle_left = self.calculate_angle(
                            self.rescale_points_to_original((key_points.left_hip_x[0], key_points.left_hip_y[0]), image_size),
                            self.rescale_points_to_original((key_points.left_knee_x[0], key_points.left_knee_y[0]), image_size),
                            self.rescale_points_to_original((key_points.left_ankle_x[0], key_points.left_ankle_y[0]), image_size))

            if knee_angle_left > 110 or knee_angle_left < 70:
                errors.append("left knee not square")

            # Define errors for leg
            leg_angle = self.angle_2_vector(
                np.array([(key_points.left_knee_x[0] - key_points.left_ankle_x[0]) * image_size[0], 
                        (key_points.left_knee_y[0] - key_points.left_ankle_y[0]) * image_size[1]]),
                np.array([1, 0])
            )

            if leg_angle < 75 or leg_angle > 105:
                errors.append("left leg not straight")
        else:
            # Define errors for knee
            knee_angle_right = 180
            knee_angle_right = self.calculate_angle(
                            self.rescale_points_to_original((key_points.left_hip_x[0], key_points.right_hip_y[0]), image_size),
                            self.rescale_points_to_original((key_points.right_knee_x[0], key_points.right_knee_y[0]), image_size),
                            self.rescale_points_to_original((key_points.right_ankle_x[0], key_points.right_ankle_y[0]), image_size)
                )

            if knee_angle_right > 110 or knee_angle_right < 70:
                errors.append("right knee not square")

            # Define errors for leg
            leg_angle = self.angle_2_vector(
                np.array([(key_points.right_knee_x[0] - key_points.right_ankle_x[0]) * image_size[0], (key_points.right_knee_y[0] - key_points.right_ankle_y[0]) * image_size[1]]),
                np.array([1, 0])
            )
            if leg_angle < 75 or leg_angle > 105:
                errors.append("right leg not straight")

        body = self.angle_2_vector(
            np.array([(key_points.left_shoulder_x[0] - key_points.left_hip_x[0]) * image_size[0], (key_points.left_shoulder_y[0] - key_points.left_hip_y[0]) * image_size[1]]),
            np.array([1, 0])
        )

        if body < 75 or body > 105:
            errors.append("body not straight")

        if errors == []:
            return "OK"
        else:
            return ", ".join(errors)
    
    def lunge_detection_realtime(self, frame, size_original, prediction_probability_threshold=0.5):
        time_skip_diff = Time.time() - self.last_time_skip
        
        # Bỏ qua frame nếu không phải frame được xử lý
        if time_skip_diff > self.thredhold_time_skip:
            self.last_time_skip = Time.time()
            t1 = threading.Thread(target=self.task, name='t1', args=(frame, size_original, prediction_probability_threshold))
            t1.start()

        image = cv2.resize(frame, size_original, interpolation =cv2.INTER_AREA)

        cv2.rectangle(image, (0, 0), (image.shape[1], 60), (245, 117, 16), -1)

        cv2.putText(image, "REP", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "STAGE", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"{self.last_predicted_stage}, {round(self.last_prediction_probability_max, 2)}", (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "ERRORS", (260, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if self.last_state == "Down":
            cv2.putText(image, f"{self.last_errors}", (260, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return image
    
    def task(self, frame, size_original, prediction_probability_threshold):
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # resize frame để tăng tốc độ xử lý
            image = self.rescale_frame(frame, percent=30)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                return

            image.flags.writeable = True

            # Cần khôi phục lại màu gốc của ảnh
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get landmarks
            try:
                key_points = self.extract_and_recalculate_landmarks(results.pose_landmarks.landmark)
                X = pd.DataFrame([key_points], columns=self.HEADERS[1:])
                errors = self.define_errors(X, self.get_image_size(image))
                X = self.input_scaler.transform(X)

                predicted_stage = self.RF_model.predict(X)[0]
                predicted_stage = self.get_class(predicted_stage)
                self.last_predicted_stage = predicted_stage
                prediction_probability_max = self.RF_model.predict_proba(X)[0].max()

                if prediction_probability_max >= prediction_probability_threshold:
                    if predicted_stage == "Down" and current_stage == "Middle":
                        self.counter += 1
                    current_stage = predicted_stage
                    self.last_state = current_stage
                    self.last_errors = errors
                    self.last_prediction_probability_max = prediction_probability_max
            except Exception as e:
                self.last_state = "Unknown"
                self.last_errors = "Unknown"
                self.last_predicted_stage = "Unknown"
                self.last_prediction_probability_max = 0
                print(f"Error: {e}")
        
    def save_error(self, error, image_frame):
        if self.history_id is not None:
            save_error({
                "ErrorType": error,
            }, cv2.imencode('.png', image_frame)[1].tostring(), self.history_id)

    def init_history(self):
        self.history_id = create_history({
            "ExcerciseName": "Lunge",
            "Datetime": datetime.datetime.now(),
            "UserID": "54U9rc8mD9Nbm4dpRAUNNm7ZYGw2"
        })

    def get_image_size(self, image):
        """
        Lấy kích thước của ảnh
        """
        return image.shape[1], image.shape[0]