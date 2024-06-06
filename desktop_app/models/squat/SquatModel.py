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

class SquatModel:
    def __init__(self):
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
        self.model = self.load_model(f"{current_dir}/SVC_model_side.pkl")
        self.input_scaler = self.load_model(f"{current_dir}/input_scaler.pkl")

        self.human_found = False

        self.counter = 0
        self.direction = 0
        self.last_state = None

        self.error1 = False
        self.error2 = False
        self.error3 = False
        self.error4 = False
        self.error5 = False

        self.last_error1 = None
        self.last_error2 = None
        self.last_error3 = None
        self.last_error4 = None
        self.last_error5 = None

        self.error1_start_time = None
        self.error2_start_time = None
        self.error3_start_time = None
        self.error4_start_time = None
        self.error5_start_time = None

        self.detector = PoseDetector(detectionCon=0.7, trackCon=0.7)

        # Config for skipping frame
        self.last_time_skip = Time.time()
        self.thredhold_time_skip = 0.25

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
    
    def get_class(self, encode_label: float):
        return {
            0: "D",
            1: "M",
            2: "U",
        }.get(encode_label, "Unknown")
    
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
        if self.last_error1 is not None:
            cv2.rectangle(image, (620, 10), (880, 50), (0, 215, 215), -1)
            cv2.putText(image, "Bend Forward", (630, 30), cv2.FONT_HERSHEY_TRIPLEX , 0.7, (59, 59, 56), 3)
        if self.last_error2 is not None:
            cv2.rectangle(image, (620, 60), (880, 100), (0, 215, 215), -1)
            cv2.putText(image, "Bend Backwards", (630, 80), cv2.FONT_HERSHEY_TRIPLEX , 0.7, (59, 59, 56), 3)
        if self.last_error3 is not None:
            cv2.rectangle(image, (620, 110), (880, 150), (64, 64, 204), -1)
            cv2.putText(image, "Lower one's hips", (630, 130), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 230), 3)
        if self.last_error4 is not None:
            cv2.rectangle(image, (620, 160), (880, 200), (64, 64, 204), -1)
            cv2.putText(image, "Knee falling over toes", (630, 180), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 230), 3)
        if self.last_error5 is not None:
            cv2.rectangle(image, (620, 210), (880, 250), (204, 122, 0), -1)
            cv2.putText(image, "Deep squats", (630, 230), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 230), 3)

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
            left, right, leftBackAngle, rightBackAngle, leftkneeAngleLineAngle, rightkneeAngleLineAngle = hands[0:]

            # Set lại giá trị ban đầu
            self.error1 = False
            self.error2 = False
            self.error3 = False
            self.error4 = False
            self.error5 = False

            # Counting number of squat
            if left > 75 and right > 75:
                if self.direction == 0:
                    self.counter += 0.5
                    self.direction = 1
            if left <= 70 and right <= 70:
                if self.direction == 1:
                    self.counter += 0.5
                    self.direction = 0

            current_time = Time.time()

            if leftBackAngle >= 15 and rightBackAngle >= 15:
                self.error1 = True
                if self.error1_start_time is None:  # Nếu lỗi mới xuất hiện
                    self.error1_start_time = current_time
            elif self.error1_start_time is not None:
                self.error1_start_time = None  # Reset thời gian nếu lỗi không còn

            if leftBackAngle >= 45 and rightBackAngle >= 45:
                self.error2 = True
                if self.error2_start_time is None:
                    self.error2_start_time = current_time
            elif self.error2_start_time is not None:
                self.error2_start_time = None

            if leftBackAngle >= 30 and rightBackAngle >= 30 and left <= 80 and right <= 80:
                self.error3 = True
                if self.error3_start_time is None:
                    self.error3_start_time = current_time
            elif self.error3_start_time is not None:
                self.error3_start_time = None

            if leftkneeAngleLineAngle >= 30 and rightkneeAngleLineAngle >= 30:
                self.error4 = True
                if self.error4_start_time is None:
                    self.error4_start_time = current_time
            elif self.error4_start_time is not None:
                self.error4_start_time = None

            if left >= 95 and right >= 95:
                self.error5 = True
                if self.error5_start_time is None:
                    self.error5_start_time = current_time
            elif self.error5_start_time is not None:
                self.error5_start_time = None

            # putting scores on the screen
            cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
            cv2.putText(img, str(int(self.counter)), (1, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6)

            # Draw errors
            if self.error1 and self.error1_start_time  and (current_time - self.error1_start_time) < 1000:
                cv2.rectangle(img, (380, 10), (630, 50), (0, 215, 215), -1)
                cv2.putText(img, "Bend Forward", (390, 30), cv2.FONT_HERSHEY_TRIPLEX , 0.7, (59, 59, 56), 3)
                errors.append("Bend Forward")
                self.last_error1 = "Bend Forward"
            else:
                self.last_error1 = None

            if self.error2 and self.error2_start_time and (current_time - self.error2_start_time) < 1000:
                cv2.rectangle(img, (380, 60), (630, 100), (0, 215, 215), -1)
                cv2.putText(img, "Bend Backwards", (390, 80), cv2.FONT_HERSHEY_TRIPLEX , 0.7, (59, 59, 56), 3)
                errors.append("Bend Backwards")
                self.last_error2 = "Bend Backwards"
            else:
                self.last_error2 = None

            if self.error3 and self.error3_start_time and (current_time - self.error3_start_time) < 1000:
                cv2.rectangle(img, (380, 110), (630, 150), (64, 64, 204), -1)
                cv2.putText(img, "Lower one's hips", (390, 130), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 230), 3)
                errors.append("Lower one's hips")
                self.last_error3 = "Lower one's hips"
            else:
                self.last_error3 = None

            if self.error4 and self.error4_start_time and (current_time - self.error4_start_time) < 1000:
                cv2.rectangle(img, (380, 160), (630, 200), (64, 64, 204), -1)
                cv2.putText(img, "Knee falling over toes", (390, 180), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 230), 3)
                errors.append("Knee falling over toes")
                self.last_error4 = "Knee falling over toes"
            else:
                self.last_error4 = None

            if self.error5 and self.error5_start_time and (current_time - self.error5_start_time) < 1000:
                cv2.rectangle(img, (380, 210), (630, 250), (204, 122, 0), -1)
                cv2.putText(img, "Deep squats", (390, 230), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 230), 3)
                errors.append("Deep squats")
                self.last_error5 = "Deep squats"
            else:
                self.last_error5 = None

            # Converting values for rectangles
            leftval = np.interp(left, [0, 100], [400, 200])
            rightval = np.interp(right, [0, 100], [400, 200])

            # For color changing
            value_left = np.interp(left, [0, 100], [0, 100])
            value_right = np.interp(right, [0, 100], [0, 100])

            # Drawing right rectangle and putting text
            cv2.putText(img, 'R', (24, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
            cv2.rectangle(img, (8, 200), (50, 400), (0, 255, 0), 5)
            cv2.rectangle(img, (8, int(rightval)), (50, 400), (255, 0, 0), -1)

            # Drawing right rectangle and putting text
            cv2.putText(img, 'L', (604, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
            cv2.rectangle(img, (582, 200), (632, 400), (0, 255, 0), 5)
            cv2.rectangle(img, (582, int(leftval)), (632, 400), (255, 0, 0), -1)

            # Tô màu đỏ khi góc đạt đến trạng thái s3
            if value_left > 75:
                cv2.rectangle(img, (582, int(leftval)), (632, 400), (0, 0, 255), -1)

            if value_right > 75:
                cv2.rectangle(img, (8, int(rightval)), (50, 400), (0, 0, 255), -1)

            if len(errors) > 0:
                self.save_error(", ".join(errors), img)
        

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