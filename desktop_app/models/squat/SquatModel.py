import cv2
from cvzone.PoseModule import PoseDetector
from models.squat.angleFinder import angleFinder
import pickle
import os
import numpy as np
import pandas as pd
import time as Time
import datetime
import threading
from services.Histories import create_history, save_error
import mediapipe as mp
import soundfile as sf
import sounddevice as sd

class SquatModel:
    def __init__(self, thredhold_time_skip = 0.25):
        self.IMPORTANT_LMS = [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE"
        ]

        # Tạo các cột cho dữ liệu đầu vào
        self.HEADERS = ["label"]
        for landmark in self.IMPORTANT_LMS:
            for dim in ['x', 'y', 'z']:
                self.HEADERS.append(f"{landmark.lower()}_{dim}")
        
        self.mp_pose = mp.solutions.pose

        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.model = self.load_model(f"{current_dir}/best_models/SVC_model_side.pkl")
        self.input_scaler = self.load_model(f"{current_dir}/best_models/input_scaler.pkl")
        self.error_types_audio = self.load_audio(f"{current_dir}/audios")

        self.human_found = False

        self.counter = 0
        self.direction = 0
        self.last_state = None

        self.last_error_bend_forward = None
        self.last_error_deep_squat = None
        self.last_error_knees_straight = None
        self.last_error_lift_hips = None

        self.last_label_error_bend_forward = None
        self.last_label_error_deep_squat = None
        self.last_label_error_knees_straight = None
        self.last_label_error_lift_hips = None

        self.error_bend_forward_start_time = None
        self.error_deep_squat_start_time = None
        self.error_knees_straight_start_time = None
        self.error_lift_hips_start_time = None

        self.detector = PoseDetector(detectionCon=0.7, trackCon=0.7)

        # Config for skipping frame
        self.last_time_skip = Time.time()
        self.thredhold_time_skip = thredhold_time_skip

        self.is_playing = False
        self.history_id = None

    def extract_amd_recalculate_landmarks(self, pose_landmarks):
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
    
    def get_class(self, encode_label: float):
        return {
            0: "D",
            1: "M",
            2: "U",
        }.get(encode_label, "Unknown")
    
    def counting_number_of_squat(self, hands):
        # Extracting values
        left, right, leftBackAngle, rightBackAngle, leftkneeAngleLineAngle, rightkneeAngleLineAngle = hands[0:]

        # Counting number of squat
        if left > 75 and right > 75:
            if self.direction == 0:
                self.counter += 0.5
                self.direction = 1
        if left <= 70 and right <= 70:
            if self.direction == 1:
                self.counter += 0.5
                self.direction = 0
    
    def define_error(self, hands, current_time):
        # Init errors
        errors = []

        # Extracting values
        left, right, leftBackAngle, rightBackAngle, leftkneeAngleLineAngle, rightkneeAngleLineAngle = hands[0:]

        # Error 1: Bend forward
        if leftBackAngle >= 20 and rightBackAngle >= 20:
            self.last_error_bend_forward = True
            errors.append("bend_forward")
            if self.error_bend_forward_start_time is None:
                self.error_bend_forward_start_time = current_time
        elif self.last_error_bend_forward is not None:
            self.last_error_bend_forward = None

        if self.last_error_bend_forward and self.error_bend_forward_start_time and (current_time - self.error_bend_forward_start_time) < 1000:
            self.last_label_error_bend_forward = "bend_forward"
        else:
            self.last_label_error_bend_forward = None

        if self.last_error_deep_squat and self.error_deep_squat_start_time and (current_time - self.error_deep_squat_start_time) < 1000:
            self.last_label_error_deep_squat = "deep_squat"
        else:
            self.last_label_error_deep_squat = None

        if self.last_error_knees_straight and self.error_knees_straight_start_time and (current_time - self.error_knees_straight_start_time) < 1000:
            self.last_label_error_knees_straight = "knees_straight"
        else:
            self.last_label_error_knees_straight = None

        # Error 2: Deep squat
        if left >= 95 and right >= 95:
            self.last_error_deep_squat = True
            errors.append("deep_squat")
            if self.error_deep_squat_start_time is None:
                self.error_deep_squat_start_time = current_time
        elif self.error_deep_squat_start_time is not None:
            self.error_deep_squat_start_time = None

        # Error 3: Knees straight
        if leftkneeAngleLineAngle >= 30 and rightkneeAngleLineAngle >= 30:
            self.last_error_knees_straight = True
            errors.append("knees_straight")
            if self.error_knees_straight_start_time is None:
                self.error_knees_straight_start_time = current_time
        elif self.error_knees_straight_start_time is not None:
            self.error_knees_straight_start_time = None

        return ", ".join(errors)
    
    def squat_detection_realtime(self, frame, size_original = (640, 480)):
        time_skip_diff = Time.time() - self.last_time_skip
        
        # Bỏ qua frame nếu không phải frame được xử lý
        if time_skip_diff > self.thredhold_time_skip:
            self.last_time_skip = Time.time()
            t1 = threading.Thread(target=self.task, name='t1', args=(frame, size_original))
            t1.start()

        image = cv2.resize(frame, (size_original[0], size_original[1]))

        if not self.human_found:
            cv2.rectangle(image, (0, 0), (620, 120), (255, 0, 0), -1)
            cv2.putText(image, str(int(self.counter)), (1, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6)
            cv2.putText(image, "No human found", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return image

        # putting scores on the screen
        cv2.rectangle(image, (0, 0), (620, 120), (255, 0, 0), -1)
        cv2.putText(image, str(int(self.counter)), (1, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6)
        if self.last_state is not None:
            cv2.putText(image, f"State: {self.last_state}", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw errors
        if self.last_label_error_bend_forward is not None:
            cv2.rectangle(image, (380, 60), (630, 100), (64, 64, 204), -1)
            cv2.putText(image, "Bend Forward", (390, 80), cv2.FONT_HERSHEY_TRIPLEX , 0.7, (255, 255, 230), 3)
        if self.last_label_error_knees_straight is not None:
            cv2.rectangle(image, (380, 160), (630, 200), (64, 64, 204), -1)
            cv2.putText(image, "Knee falling over toes", (390, 180), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 230), 3)
        if self.last_label_error_deep_squat is not None:
            cv2.rectangle(image, (380, 210), (630, 250), (204, 122, 0), -1)
            cv2.putText(image, "Deep squats", (390, 230), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 230), 3)

        return image

    def task(self, frame, size_original = (640, 480)):
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            errors = []

            results = pose.process(frame)

            if not results.pose_landmarks:
                self.human_found = False
                return
            self.human_found = True

            # Cần khôi phục lại màu gốc của ảnh
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            key_points = self.extract_amd_recalculate_landmarks(results.pose_landmarks.landmark)
            X = pd.DataFrame([key_points], columns=self.HEADERS[1:])
            X = self.input_scaler.transform(X)

            predicted_class = self.model.predict(X)[0]
            predicted_class = self.get_class(self.model.predict(X)[0])
            self.last_state = predicted_class

            # putting predicted class on the screen
            cv2.putText(img, f"State: {predicted_class}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.detector.findPose(img, draw=0)
            lmList, bboxInfo = self.detector.findPosition(img, bboxWithHands=0, draw=False)

            angle1 = angleFinder(img, lmList, 23, 25, 27, 24, 26, 28, 11, 12, drawPoints=True)
            hands = angle1.angle()
            
            # Counting number of squat
            self.counting_number_of_squat(hands)

            current_time = Time.time()
            errors = self.define_error(hands, current_time)

            parts = errors.split(", ")[:2]
            # Nối các phần lại với nhau
            error = ", ".join(parts)
            error_type = error.replace(", ", "_").replace(" ", "_").lower()
            print("Error type: " + error_type)
            if error_type in self.error_types_audio and not self.is_playing:
                data, samplerate = self.error_types_audio[error_type]
                self.start_audio_thread(data, samplerate)
        

    def save_error(self, error, image_frame):
        if self.history_id is not None:
            save_error({
                "ErrorType": error,
            }, cv2.imencode('.png', image_frame)[1].tostring(), self.history_id)
    
    def init_history(self):
        self.history_id = create_history({
            "ExcerciseName": "Squat",
            "Datetime": datetime.datetime.now(),
            "UserID": "54U9rc8mD9Nbm4dpRAUNNm7ZYGw2"
        })