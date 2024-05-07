import cv2
from cvzone.PoseModule import PoseDetector
from models.squat.angleFinder import angleFinder
import numpy as np
import time

class SquatModel:
    def __init__(self):
        self.counter = 0
        self.direction = 0
        self.error1 = False
        self.error2 = False
        self.error3 = False
        self.error4 = False
        self.error5 = False

        self.error1_start_time = None
        self.error2_start_time = None
        self.error3_start_time = None
        self.error4_start_time = None
        self.error5_start_time = None

        self.detector = PoseDetector(detectionCon=0.7, trackCon=0.7)

    def squat_detection(self, frame, size_original = (640, 480)):
        img = cv2.resize(frame, (size_original[0], size_original[1]))

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

        current_time = time.time()

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
        if self.error2 and self.error2_start_time and (current_time - self.error2_start_time) < 1000:
            cv2.rectangle(img, (380, 60), (630, 100), (0, 215, 215), -1)
            cv2.putText(img, "Bend Backwards", (390, 80), cv2.FONT_HERSHEY_TRIPLEX , 0.7, (59, 59, 56), 3)
        if self.error3 and self.error3_start_time and (current_time - self.error3_start_time) < 1000:
            cv2.rectangle(img, (380, 110), (630, 150), (64, 64, 204), -1)
            cv2.putText(img, "Lower one's hips", (390, 130), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 230), 3)
        if self.error4 and self.error4_start_time and (current_time - self.error4_start_time) < 1000:
            cv2.rectangle(img, (380, 160), (630, 200), (64, 64, 204), -1)
            cv2.putText(img, "Knee falling over toes", (390, 180), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 230), 3)
        if self.error5 and self.error5_start_time and (current_time - self.error5_start_time) < 1000:
            cv2.rectangle(img, (380, 210), (630, 250), (204, 122, 0), -1)
            cv2.putText(img, "Deep squats", (390, 230), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 230), 3)

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

        return img