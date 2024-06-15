import mediapipe as mp
import math
import numpy as np
import cv2
import warnings
import pickle
import os
import time as Time
import threading
from services.Histories import create_history, save_error
import datetime
import soundfile as sf
import sounddevice as sd

class BicepCurlModel:
    def __init__(self, thredhold_time_skip = 0.25):
        warnings.filterwarnings('ignore')
        self.last_time_skip = Time.time()
        self.thredhold_time_skip = thredhold_time_skip # Số s được bỏ qua
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.RF_model = self.load_model(f'{current_dir}/best_models/RF_model.pkl')
        self.input_scaler = self.load_model(f"{current_dir}/best_models/input_scaler.pkl")

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
        ]
        self.HEADERS = ["label"]
        for landmark in self.IMPORTANT_LMS:
            for dim in ['x', 'y', 'z']:
                self.HEADERS.append(f"{landmark.lower()}_{dim}")

        self.last_errors_list = []
        self.last_errors = "Unknown"
        self.last_state = None
        self.last_predicted_stage = "Unknown"
        self.counter = 0
        self.last_prediction_probability_max = 0
        self.direction = "Up"
        self.last_angles = [0, 0]

        self.error_types_audio = self.load_audio(f"{current_dir}/audios")
        self.is_playing = False

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
            data[idx] = [key_point.x + delta_x - 0.5, key_point.y + delta_y - 0.5, key_point.z]

        return data.flatten().tolist()

    def rescale_frame(self, frame, percent=50):
        '''
        Rescale a frame to a certain percentage compare to its original frame
        '''
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    
    def get_class(self, encode_label: float):
        """
        Chuyển một label được encode thành class tương ứng
        """
        return {
            0: "Down",
            1: "Middle",
            2: "Up"
        }.get(encode_label)
    
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

        return math.degrees(math.acos((ba_vector[0] * bc_vector[0] + ba_vector[1] * bc_vector[1]) / (ba_length * bc_length)))    
    
    def get_image_size(self, image):
        """
        Lấy kích thước của ảnh
        """
        return image.shape[1], image.shape[0]
    
    def define_leg_front(self, knee_left, knee_right):
        """
        Xác định chân đưa lên trước, dựa vào vị trí của đầu gối trái và đầu gối phải
        """
        if knee_left[1] < knee_right[1]:
            return "left"
        else:
            return "right"
        
    def determine_stage(self, key_points, image_size):
        # Dựa vào góc giữa khuỷu tay, cổ tay và vai để xác định giai đoạn của động tác Bicep Curl
        left_shoulder = [key_points[11].x, key_points[11].y]
        right_shoulder = [key_points[12].x, key_points[12].y]
        left_elbow = [key_points[13].x, key_points[13].y]
        right_elbow = [key_points[14].x, key_points[14].y]
        left_wrist = [key_points[15].x, key_points[15].y]
        right_wrist = [key_points[16].x, key_points[16].y]

        # Tính góc giữa 2 vector
        left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist, image_size)
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist, image_size)
        
        if abs(left_angle - right_angle) < 10:
            angle = max(left_angle, right_angle)
        else:
            angle = min(left_angle, right_angle)
        print(left_angle, right_angle)

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
            angle, self.calculate_angle(right_shoulder, right_hip, right_elbow, image_size)
        )
        if angle > 25:
            errors.append("arm not straight")

        # Góc giữa khuỷu tay, cổ tay và ngón tay trỏ
        angle = self.calculate_angle(left_elbow, left_wrist, left_index, image_size)
        angle = 180 - max(
            angle, self.calculate_angle(right_elbow, right_wrist, right_index, image_size)
        )
        if angle >= 50:
            errors.append("wrist not straight")

        if errors == []:
            return "None"
        else:
            return ", ".join(errors)
    
    def play_audio(self, data, samplerate):
        self.is_playing = True
        sd.play(data, samplerate)
        sd.wait()
        self.is_playing = False
        
    def start_audio_thread(self, data, samplerate):
        threading.Thread(target=self.play_audio, args=(data, samplerate,), daemon=True).start()
        
    def bicep_curl_detection_realtime(self, frame, size_original, prediction_probability_threshold=0.5):
        time_skip_diff = Time.time() - self.last_time_skip
        
        # Bỏ qua frame nếu không phải frame được xử lý
        if time_skip_diff > self.thredhold_time_skip:
            self.last_time_skip = Time.time()
            t1 = threading.Thread(target=self.task, name='t1', args=(frame, size_original, prediction_probability_threshold))
            t1.start()

        image = cv2.resize(frame, size_original, interpolation =cv2.INTER_AREA)

        cv2.rectangle(image, (0, 0), (image.shape[1], 60), (245, 117, 16), -1)            
        cv2.putText(image, "STAGE, REP", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if self.last_predicted_stage == "Up":
            cv2.putText(image, f"{self.last_predicted_stage}, {self.counter}", (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, f"{self.last_predicted_stage}, {self.counter}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "ERRORS", (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Hiển thị các lỗi theo hàng dọc
        y_position = 45
        for error in self.last_errors_list:
            cv2.putText(image, error, (220, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2, cv2.LINE_AA)
            y_position += 30

        # cv2.putText(image, "ANGLES", (380, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(self.last_angles), (380, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2, cv2.LINE_AA)
        return image
    
    def task(self, frame, size_original, prediction_probability_threshold):
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # resize frame để tăng tốc độ xử lý
            image = self.rescale_frame(frame, percent=60)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                return
            
            image.flags.writeable = True
            # Cần khôi phục lại màu gốc của ảnh
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                image_size = self.get_image_size(image)
                errors = self.define_errors(results.pose_landmarks.landmark, image_size)
                # key_points = self.extract_and_recalculate_landmarks(results.pose_landmarks.landmark)
                # X = pd.DataFrame([key_points], columns=self.HEADERS[1:])
                # X = self.input_scaler.transform(X)

                # predicted_stage = self.RF_model.predict(X)[0]
                # predicted_stage = self.get_class(predicted_stage)
                # prediction_probability_max = self.RF_model.predict_proba(X)[0].max()
                predicted_stage = self.determine_stage(results.pose_landmarks.landmark, image_size)

                errors_list = errors.split(", ")
                self.last_errors_list = errors_list
                if len(errors_list) == 1:
                    error_types = errors_list[0].replace(" ", "_")
                elif len(errors_list) == 2:
                    error_types = "_".join(errors_list).replace(" ", "_").replace("_not_straight", "", 1)
                else:
                    error_types = "_".join([errors_list[0], errors_list[1]]).replace(" ", "_").replace("_not_straight", "", 1)

                if errors != "OK" and error_types in self.error_types_audio and not self.is_playing:
                    self.start_audio_thread(*self.error_types_audio[error_types])

                if self.last_state == "Up" and predicted_stage == "Middle":
                    self.direction = "Down"
                elif self.last_state == "Middle" and predicted_stage == "Down" and self.direction == "Down":
                    self.counter += 1
                    self.direction = "Up"

                self.last_state = predicted_stage
                self.last_predicted_stage = predicted_stage
                self.last_prediction_probability_max = 1
            except Exception as e:
                print(e)
                self.last_errors = "Unknown"
                self.last_state = None
                self.last_predicted_stage = "Unknown"
                self.last_prediction_probability_max = 0

        
    def save_error(self, error, image_frame):
        if self.history_id is not None:
            save_error({
                "ErrorType": error,
            }, cv2.imencode('.png', image_frame)[1].tostring(), self.history_id)

    def init_history(self):
        self.history_id = create_history({
            "ExcerciseName": "Bicep Curl",
            "Datetime": datetime.datetime.now(),
            "UserID": "54U9rc8mD9Nbm4dpRAUNNm7ZYGw2"
        })