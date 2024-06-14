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

class PlankModel:
    def __init__(self, thredhold_time_skip = 0.25):
        warnings.filterwarnings('ignore')
        self.last_time_skip = Time.time()
        self.thredhold_time_skip = thredhold_time_skip # Số s được bỏ qua
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.RF_model = self.load_model(f'{current_dir}/best_models/RF_model.pkl')
        self.input_scaler = self.load_model(f"{current_dir}/best_models/input_scaler.pkl")
        self.error_types_audio = self.load_audio(f"{current_dir}/audios")
        self.is_playing = False
        self.IMPORTANT_LMS = [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
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
        self.last_error = "Unknown"
        self.last_prediction_probability_max = 0
        self.history_id = None

    def load_model(self, file_name):
        with open(file_name, "rb") as file:
            model = pickle.load(file)
        return model
    
    def load_audio(self, folder_path = "audios"):
        # Khởi tạo dictionary để lưu các đối tượng audio
        error_types_audio = {}

        # Thư mục chứa các file âm thanh
        current_path = os.getcwd()

        # Duyệt qua các file trong thư mục
        for filename in os.listdir(folder_path):
            file_path = os.path.join(current_path, folder_path, filename)
            if os.path.isfile(file_path):
                data, samplerate = sf.read(file_path)
                filename = filename.replace(".wav", "")
                error_types_audio[filename] = (data, samplerate)
        return error_types_audio
    
    def play_audio(self, data, samplerate):
        self.is_playing = True
        sd.play(data, samplerate)
        sd.wait()
        self.is_playing = False

    # Hàm bắt đầu một luồng để phát âm thanh
    def start_audio_thread(self, data, samplerate):
        print("Start audio thread")
        threading.Thread(target=self.play_audio, args=(data, samplerate,), daemon=True).start()

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
            0: "C",
            1: "W"
        }.get(encode_label, "Unknown")
    
    def get_color_for_landmarks(self, label):
        if label == "C":
            return ((255, 165, 0), (255, 140, 0))
        elif label == "W":
            return ((29, 62, 199), (1, 143, 241))

    def get_image_size(self, image):
        return image.shape[1], image.shape[0]

    def calculate_angle(self, a, b, c, size_of_image):
        # Lấy tọa độ của 3 điểm
        a = (a[0] * size_of_image[0], a[1] * size_of_image[1])
        b = (b[0] * size_of_image[0], b[1] * size_of_image[1])
        c = (c[0] * size_of_image[0], c[1] * size_of_image[1])

        # Tính góc giữa 3 điểm
        angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        
        # Chuyển góc về khoảng từ 0 đến 360 độ
        if angle < 0:
            angle += 360
        return angle

    def define_error(self, key_points, image_size):
        error = []
        angle = 180

        if (key_points.nose_x[0] > 0):
            angle = self.calculate_angle((key_points.right_shoulder_x[0], key_points.right_shoulder_y[0]), 
                            (key_points.right_hip_x[0], key_points.right_hip_y[0]), 
                            (key_points.right_ankle_x[0], key_points.right_ankle_y[0]), 
                            image_size)
        else:
            angle = self.calculate_angle((key_points.left_ankle_x[0], key_points.left_ankle_y[0]), 
                            (key_points.left_hip_x[0], key_points.left_hip_y[0]), 
                            (key_points.left_shoulder_x[0], key_points.left_shoulder_y[0]), 
                            image_size)

        if angle < 170:
            error.append("high hip")
        elif angle > 190:
            error.append("low hip")

        angle = 90
        if (key_points.nose_x[0] > 0):
            angle = self.calculate_angle((key_points.right_shoulder_x[0], key_points.right_shoulder_y[0]),
                                (key_points.right_elbow_x[0], key_points.right_elbow_y[0]),
                                (key_points.right_wrist_x[0], key_points.right_wrist_y[0]),
                                image_size)
        else:
            angle = self.calculate_angle((key_points.left_wrist_x[0], key_points.left_wrist_y[0]),
                                (key_points.left_elbow_x[0], key_points.left_elbow_y[0]),
                                (key_points.left_shoulder_x[0], key_points.left_shoulder_y[0]),
                                image_size)
        
        if angle < 75:
            error.append("elbow to front")
        elif angle > 105:
            error.append("elbow to after")

        if error == []:
            return "Unknown"
        else:
            return ", ".join(error)

    def plank_detection(self, frame, size_original, prediction_probability_threshold=0.5):
        current_class = "Unknown"

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # resize frame để tăng tốc độ xử lý
            image = self.rescale_frame(frame, percent=50)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                image = cv2.resize(image, size_original, interpolation =cv2.INTER_AREA)
                # Cần khôi phục lại màu gốc của ảnh
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.rectangle(image, (0, 0), (750, 60), (245, 117, 16), -1)
                cv2.putText(image, "ERROR", (180, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(image, "No human found", (180, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                return image

            initial_pose_landmarks = copy.deepcopy(results.pose_landmarks)
            image.flags.writeable = True

            # Cần khôi phục lại màu gốc của ảnh
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get landmarks
            try:
                key_points = self.extract_and_recalculate_landmarks(results.pose_landmarks.landmark)
                X = pd.DataFrame([key_points], columns=self.HEADERS[1:])
                error = self.define_error(X, self.get_image_size(image))
                X = self.input_scaler.transform(X)

                predicted_class = self.RF_model.predict(X)[0]
                predicted_class = self.get_class(self.RF_model.predict(X)[0])
                prediction_probability_max = self.RF_model.predict_proba(X)[0].max()

                if prediction_probability_max >= prediction_probability_threshold:
                    current_class = predicted_class
                else:
                    current_class = "Unknown"

                colors = self.get_color_for_landmarks(current_class)
                self.mp_drawing.draw_landmarks(
                        image,
                        initial_pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=colors[0], thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=colors[1], thickness=2, circle_radius=1),
                    )
                
                # scale image back to original size
                image = cv2.resize(image, size_original, interpolation =cv2.INTER_AREA)

                cv2.rectangle(image, (0, 0), (750, 60), (245, 117, 16), -1)

                # Display error
                cv2.putText(image, "ERROR", (180, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                if (current_class == "Unknown"):
                    cv2.putText(image, "Unknown", (180, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                elif (current_class == "W"):
                    cv2.putText(image, error, (180, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "OK", (180, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display class
                cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, current_class, (110, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display probability
                cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(prediction_probability_max, 2)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")

        return image

    def plank_detection_realtime(self, frame, size_original, prediction_probability_threshold=0.5):
        time_skip_diff = Time.time() - self.last_time_skip
        
        # Bỏ qua frame nếu không phải frame được xử lý
        if time_skip_diff > self.thredhold_time_skip:
            self.last_time_skip = Time.time()
            t1 = threading.Thread(target=self.task, name='t1', args=(frame, size_original, prediction_probability_threshold))
            t1.start()

        image = cv2.resize(frame, size_original, interpolation =cv2.INTER_AREA)

        cv2.rectangle(image, (0, 0), (750, 60), (245, 117, 16), -1)
        cv2.putText(image, "ERROR", (180, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image, self.last_error, (180, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display class
        cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, self.last_class, (110, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display probability
        cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(self.last_prediction_probability_max, 2)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image
    
    def task(self, frame, size_original, prediction_probability_threshold=0.5):
        current_class = "Unknown"

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # resize frame để tăng tốc độ xử lý
            image = self.rescale_frame(frame, percent=50)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                image = cv2.resize(image, size_original, interpolation =cv2.INTER_AREA)
                # Cần khôi phục lại màu gốc của ảnh
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.rectangle(image, (0, 0), (750, 60), (245, 117, 16), -1)
                cv2.putText(image, "ERROR", (180, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(image, "No human found", (180, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                return image
            
            image.flags.writeable = True

            # Cần khôi phục lại màu gốc của ảnh
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.rectangle(image, (0, 0), (750, 60), (245, 117, 16), -1)

            # Get landmarks
            try:
                key_points = self.extract_and_recalculate_landmarks(results.pose_landmarks.landmark)
                X = pd.DataFrame([key_points], columns=self.HEADERS[1:])
                error = self.define_error(X, self.get_image_size(image))
                X = self.input_scaler.transform(X)

                predicted_class = self.RF_model.predict(X)[0]
                predicted_class = self.get_class(self.RF_model.predict(X)[0])
                prediction_probability_max = self.RF_model.predict_proba(X)[0].max()

                self.last_prediction_probability_max = prediction_probability_max

                if prediction_probability_max >= prediction_probability_threshold:
                    current_class = predicted_class
                else:
                    current_class = "Unknown"

                if current_class == "W" and error == "Unknown":
                    current_class = "C"
                    error = "None"

                self.last_class = current_class

                error_format = error.replace(", ", "_").replace(" ", "_").lower()
                if current_class == "W" and error_format in self.error_types_audio and not self.is_playing:
                    data, samplerate = self.error_types_audio[error_format]
                    self.start_audio_thread(data, samplerate)

                # Display class
                cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, current_class, (110, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display probability
                cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(prediction_probability_max, 2)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # set error
                cv2.putText(image, "ERROR", (180, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                if (current_class == "Unknown"):
                    cv2.putText(image, "Unknown", (180, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    self.last_error = "Unknown"
                elif (current_class == "W"):
                    self.last_error = error
                    cv2.putText(image, error, (180, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    self.save_error(error, image)
                else:
                    self.last_error = "OK"
                    cv2.putText(image, "OK", (180, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")
    def save_error(self, error, image_frame):
        if self.history_id is not None:
            save_error({
                "ErrorType": error,
            }, cv2.imencode('.png', image_frame)[1].tostring(), self.history_id)

    def init_history(self):
        self.history_id = create_history({
            "ExcerciseName": "Plank",
            "Datetime": datetime.datetime.now(),
            "UserID": "54U9rc8mD9Nbm4dpRAUNNm7ZYGw2"
        })