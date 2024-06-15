import mediapipe as mp
import math
import numpy as np
import pandas as pd
import cv2
from cvzone.PoseModule import PoseDetector
import warnings
import pickle
import os
import time


class AngleFinder:
    def __init__(self, lm_list, p1, p2, p3, p4, p5, p6, p7, p8, image, draw_points):
        self.lm_list = lm_list
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.image = image
        self.draw_points = draw_points

    # finding angles
    def angle(self):
        if len(self.lm_list) != 0:
            left_hip = self.lm_list[self.p1][:2]
            left_knee = self.lm_list[self.p2][:2]
            left_ankle = self.lm_list[self.p3][:2]
            right_hip = self.lm_list[self.p4][:2]
            right_knee = self.lm_list[self.p5][:2]
            right_ankle = self.lm_list[self.p6][:2]
            left_shoulder = self.lm_list[self.p7][:2]
            right_shoulder = self.lm_list[self.p8][:2]

            if (
                len(left_hip) >= 2
                and len(left_knee) >= 2
                and len(left_ankle) >= 2
                and len(right_hip) >= 2
                and len(right_knee) >= 2
                and len(right_ankle) >= 2
                and len(left_shoulder) >= 2
                and len(right_shoulder) >= 2
            ):
                x1, y1 = left_hip[:2]
                x2, y2 = left_knee[:2]
                x3, y3 = left_ankle[:2]
                x4, y4 = right_hip[:2]
                x5, y5 = right_knee[:2]
                x6, y6 = right_ankle[:2]
                x7, y7 = left_shoulder[:2]
                x8, y8 = right_shoulder[:2]

                vertical_line_angle = 90  # Góc của đường thẳng đứng so với trục x

                # calculating angle for left and right hands
                left_hand_angle = math.degrees(
                    math.atan2(y1 - y2, x1 - x2) - math.atan2(y1 - y2, 0)
                )
                right_hand_angle = math.degrees(
                    math.atan2(y4 - y5, x4 - x5) - math.atan2(y4 - y5, 0)
                )

                left_back_angle = math.degrees(
                    math.atan2(y1 - y2, 0) - math.atan2(y7 - y1, x7 - x1)
                )
                right_back_angle = math.degrees(
                    math.atan2(y4 - y5, 0) - math.atan2(y8 - y4, x8 - x4)
                )

                p2_p3_angle = math.degrees(math.atan2(y3 - y2, x3 - x2))
                p5_p6_angle = math.degrees(math.atan2(y6 - y5, x6 - x5))
                # Tính góc giữa đường thẳng đứng và đường thẳng nối giữa đầu gối và mắt cá chân
                left_knee_angle_line = abs(vertical_line_angle - p2_p3_angle)
                right_knee_angle_line = abs(vertical_line_angle - p5_p6_angle)

                left_hand_angle = int(
                    np.interp(left_hand_angle, [0, 95], [0, 100])
                )  # Ánh xạ sang từ 0-95 về 0-100 vì góc càng nhỏ thì càng tiến về trạng thái s3

                right_hand_angle = int(
                    np.interp(right_hand_angle, [0, 95], [0, 100])
                )  # Ánh xạ sang từ 0-95 về 0-100 vì góc càng nhỏ thì càng tiến về trạng thái s3

                left_back_angle = int(
                    np.interp(left_back_angle, [0, 95], [0, 100])
                )  # Ánh xạ sang từ 0-95 về 0-100 vì góc càng nhỏ thì càng tiến về trạng thái s3

                right_back_angle = int(
                    np.interp(right_back_angle, [0, 95], [0, 100])
                )  # Ánh xạ sang từ 0-95 về 0-100 vì góc càng nhỏ thì càng tiến về trạng thái s3

                # drawing circles and lines on selected points
                if self.draw_points:
                    cv2.circle(self.image, (x1, y1), 2, (0, 255, 0), 6)
                    cv2.circle(self.image, (x2, y2), 2, (0, 255, 0), 6)
                    cv2.circle(self.image, (x3, y3), 2, (0, 255, 0), 6)
                    cv2.circle(self.image, (x4, y4), 2, (0, 255, 0), 6)
                    cv2.circle(self.image, (x5, y5), 2, (0, 255, 0), 6)
                    cv2.circle(self.image, (x6, y6), 2, (0, 255, 0), 6)
                    cv2.circle(self.image, (x7, y7), 2, (0, 255, 0), 6)
                    cv2.circle(self.image, (x8, y8), 2, (0, 255, 0), 6)

                    cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.line(self.image, (x2, y2), (x3, y3), (0, 0, 255), 2)
                    cv2.line(self.image, (x4, y4), (x5, y5), (0, 0, 255), 2)
                    cv2.line(self.image, (x5, y5), (x6, y6), (0, 0, 255), 2)
                    cv2.line(self.image, (x1, y1), (x4, y4), (0, 0, 255), 2)
                    cv2.line(self.image, (x1, y1), (x7, y7), (0, 0, 255), 2)
                    cv2.line(self.image, (x4, y4), (x8, y8), (0, 0, 255), 2)

                return [
                    left_hand_angle,
                    right_hand_angle,
                    left_back_angle,
                    right_back_angle,
                    left_knee_angle_line,
                    right_knee_angle_line,
                ]


class SquatModel:
    def __init__(self):
        warnings.filterwarnings("ignore")
        self.mp_drawing = mp.solutions.drawing_utils
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.mp_pose = mp.solutions.pose
        self.svc_model = self.load_model(f"{self.current_dir}\SVC_model_side.pkl")
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
        return {0: "Down", 1: "Middle", 2: "Up"}.get(encode_label)

    def get_color_for_landmarks(self, errors):
        if errors == "None":
            return ((255, 165, 0), (255, 140, 0))
        else:
            return ((29, 62, 199), (1, 143, 241))

    def get_image_size(self, image):
        return image.shape[1], image.shape[0]

    def detection_offline(self, video_path, prediction_probability_threshold=0.5):
        cap = cv2.VideoCapture(video_path if video_path else 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS: ", fps)

        # Số frame được bỏ qua
        image_width, image_height = 0, 0
        frame_skip = 3
        frame_count = 0

        counter = 0
        direction = 0
        error1 = False
        error2 = False
        error3 = False
        error4 = False

        error1_start_time = None
        error2_start_time = None
        error3_start_time = None
        error4_start_time = None

        result_frames = []
        error_details = {}

        # Đặt thời gian bắt đầu của video để lát tính thời gian tại thời điểm của từng frame
        previous_error = {"name": "Unknown", "time": 0}
        detector = PoseDetector(detectionCon=0.7, trackCon=0.7)

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
                image = self.rescale_frame(image, percent=20)
                image_width, image_height = self.get_image_size(image)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if not results.pose_landmarks:
                    print("No human found")
                    continue

                image.flags.writeable = True

                # Cần khôi phục lại màu gốc của ảnh
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Get landmarks
                try:
                    key_points = self.extract_and_recalculate_landmarks(
                        results.pose_landmarks.landmark
                    )
                    X = pd.DataFrame([key_points], columns=self.HEADERS[1:])
                    X = self.input_scaler.transform(X)

                    predicted_stage = self.svc_model.predict(X)[0]
                    predicted_stage = self.get_class(predicted_stage)

                    if predicted_stage == "D":
                        predicted_stage = "Downn"
                    elif predicted_stage == "M":
                        predicted_stage = "Middle"
                    elif predicted_stage == "U":
                        predicted_stage = "Up"
                    print("Predicted stage: ", predicted_stage)

                    detector.findPose(image, draw=0)
                    lm_list, _ = detector.findPosition(
                        image, bboxWithHands=0, draw=False
                    )

                    angle1 = AngleFinder(
                        lm_list, 23, 25, 27, 24, 26, 28, 11, 12, image, draw_points=True
                    )
                    hands = angle1.angle()
                    (
                        left,
                        right,
                        left_back_angle,
                        right_back_angle,
                        left_knee_angle_line,
                        right_knee_angle_line,
                    ) = hands[0:]

                    error = []

                    # Set lại giá trị ban đầu
                    error1 = False
                    error2 = False
                    error3 = False
                    error4 = False

                    # Counting number of squat
                    if left > 75 and right > 75:
                        if direction == 0:
                            counter += 0.5
                            direction = 1
                    if left <= 70 and right <= 70 and direction == 1:
                        counter += 0.5
                        direction = 0

                    current_time = time.time()

                    if left_back_angle >= 20 and right_back_angle >= 20:
                        error1 = True
                        error.append("bend_forward")
                        if error1_start_time is None:
                            error1_start_time = current_time
                    elif error1_start_time is not None:
                        error1_start_time = None

                    if left >= 95 and right >= 95:
                        error4 = True
                        error.append("deep_squat")
                        if error4_start_time is None:
                            error4_start_time = current_time
                    elif error4_start_time is not None:
                        error4_start_time = None

                    if left_knee_angle_line >= 30 and right_knee_angle_line >= 30:
                        error3 = True
                        error.append("knees_straight")
                        if error3_start_time is None:
                            error3_start_time = current_time
                    elif error3_start_time is not None:
                        error3_start_time = None

                    error = ", ".join(error)

                    # putting scores on the screen
                    cv2.rectangle(image, (0, 0), (120, 120), (255, 0, 0), -1)
                    cv2.putText(
                        image,
                        str(int(counter)),
                        (25, 100),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        1.6,
                        (0, 0, 255),
                        3,
                    )

                    # putting predicted class on the screen
                    cv2.putText(
                        image,
                        predicted_stage,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # Draw errors
                    if (
                        error1
                        and error1_start_time
                        and (current_time - error1_start_time) < 1000
                    ):
                        cv2.rectangle(image, (430, 60), (680, 100), (64, 64, 204), -1)
                        cv2.putText(
                            image,
                            "Bend Forward",
                            (440, 80),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.7,
                            (255, 255, 230),
                            1,
                        )

                    if (
                        error3
                        and error3_start_time
                        and (current_time - error3_start_time) < 1000
                    ):
                        cv2.rectangle(image, (430, 160), (680, 200), (64, 64, 204), -1)
                        cv2.putText(
                            image,
                            "Knee falling over toes",
                            (440, 180),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.6,
                            (255, 255, 230),
                            1,
                        )

                    if (
                        error4
                        and error4_start_time
                        and (current_time - error4_start_time) < 1000
                    ):
                        cv2.rectangle(image, (430, 210), (680, 250), (204, 122, 0), -1)
                        cv2.putText(
                            image,
                            "Deep squats",
                            (440, 230),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.7,
                            (255, 255, 230),
                            1,
                        )

                    # Converting values for rectangles
                    leftval = np.interp(left, [0, 100], [400, 200])
                    rightval = np.interp(right, [0, 100], [400, 200])

                    # For color changing
                    value_left = np.interp(left, [0, 100], [0, 100])
                    value_right = np.interp(right, [0, 100], [0, 100])

                    # Drawing right rectangle and putting text
                    cv2.putText(
                        image,
                        "R",
                        (24, 195),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                    cv2.rectangle(image, (8, 200), (50, 400), (0, 255, 0), 5)
                    cv2.rectangle(image, (8, int(rightval)), (50, 400), (255, 0, 0), -1)

                    # Drawing right rectangle and putting text
                    cv2.putText(
                        image,
                        "L",
                        (710, 195),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                    cv2.rectangle(image, (690, 200), (732, 400), (0, 255, 0), 5)
                    cv2.rectangle(
                        image, (690, int(leftval)), (732, 400), (255, 0, 0), -1
                    )

                    # Tô màu đỏ khi góc đạt đến trạng thái s3
                    if value_left > 75:
                        cv2.rectangle(
                            image, (690, int(leftval)), (732, 400), (0, 0, 255), -1
                        )

                    if value_right > 75:
                        cv2.rectangle(
                            image, (8, int(rightval)), (50, 400), (0, 0, 255), -1
                        )

                    # Lưu frame vào để phục vụ cho việc xuất video
                    result_frames.append(image)
                    current_time = round(frame_count / fps, 1)
                    if (
                        error != ""
                        and (error != previous_error["name"]
                        or current_time - previous_error["time"] >= 1.5)
                    ):
                        if error == "Unknown":
                            continue

                        error_details.setdefault(error, []).append(
                            {
                                "frame": image,
                                "frame_in_seconds": current_time,
                            }
                        )
                        previous_error = {"name": error, "time": current_time}

                    cv2.imshow("Image", image)

                    # Nhấn q để thoát
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception as e:
                    print(f"Error: {e}")

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
