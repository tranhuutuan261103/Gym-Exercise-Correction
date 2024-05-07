import cv2
from cvzone.PoseModule import PoseDetector
import math
import numpy as np

# Creating Angle finder class
class angleFinder:
    def __init__(self, img, lmlist, p1, p2, p3, p4, p5, p6, p7, p8, drawPoints):
        self.lmlist = lmlist
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.drawPoints = drawPoints

        self.img = img

    # finding angles
    def angle(self):
        if len(self.lmlist) != 0:
            hipLeft = self.lmlist[self.p1][:2]
            kneeLeft = self.lmlist[self.p2][:2]
            ankleLeft = self.lmlist[self.p3][:2]
            hipRight = self.lmlist[self.p4][:2]
            kneeRight = self.lmlist[self.p5][:2]
            ankleRight = self.lmlist[self.p6][:2]
            leftShoulder = self.lmlist[self.p7][:2]
            rightShoulder = self.lmlist[self.p8][:2]

            if len(hipLeft) >= 2 and len(kneeLeft) >= 2 and len(ankleLeft) >= 2 and len(hipRight) >= 2 and len(kneeRight) >= 2 and len(
                    ankleRight) >= 2 and len(leftShoulder) >= 2 and len(rightShoulder) >= 2:
                x1, y1 = hipLeft[:2]
                x2, y2 = kneeLeft[:2]
                x3, y3 = ankleLeft[:2]
                x4, y4 = hipRight[:2]
                x5, y5 = kneeRight[:2]
                x6, y6 = ankleRight[:2]
                x7, y7 = leftShoulder[:2]
                x8, y8 = rightShoulder[:2]

                vertical_line_angle = 90  # Góc của đường thẳng đứng so với trục x

                # calculating angle for left and right hands
                leftHandAngle = math.degrees(math.atan2(y1 - y2, x1 - x2) - math.atan2(y1 - y2, 0))
                rightHandAngle = math.degrees(math.atan2(y4 - y5, x4 - x5) - math.atan2(y4 - y5, 0))

                leftBackAngle = math.degrees(math.atan2(y1 - y2, 0) - math.atan2(y7 - y1, x7 - x1))
                rightBackAngle = math.degrees(math.atan2(y4 - y5, 0) - math.atan2(y8 - y4, x8 - x4))

                p2_p3_angle = math.degrees(math.atan2(y3 - y2, x3 - x2))
                p5_p6_angle = math.degrees(math.atan2(y6 - y5, x6 - x5))
                # Tính góc giữa đường thẳng đứng và đường thẳng nối giữa đầu gối và mắt cá chân
                leftkneeAngleLineAngle = abs(vertical_line_angle - p2_p3_angle)
                rightkneeAngleLineAngle = abs(vertical_line_angle - p5_p6_angle)

                print (f"leftHandAngle: {leftHandAngle}")
                print (f"rightHandAngle: {rightHandAngle}")
                print (f"leftBackAngle: {leftBackAngle}")
                print (f"rightBackAngle: {rightBackAngle}")
                print (f"leftkneeAngleLineAngle: {leftkneeAngleLineAngle}")
                print (f"rightkneeAngleLineAngle: {rightkneeAngleLineAngle}")
                

                leftHandAngle = int(np.interp(leftHandAngle, [0, 95], [0, 100])) # Ánh xạ sang từ 0-95 về 0-100 vì góc càng nhỏ thì càng tiến về trạng thái s3
                rightHandAngle = int(np.interp(rightHandAngle, [0, 95], [0, 100])) # Ánh xạ sang từ 0-95 về 0-100 vì góc càng nhỏ thì càng tiến về trạng thái s3
                leftBackAngle = int(np.interp(leftBackAngle, [0, 95], [0, 100])) # Ánh xạ sang từ 0-95 về 0-100 vì góc càng nhỏ thì càng tiến về trạng thái s3
                rightBackAngle = int(np.interp(rightBackAngle, [0, 95], [0, 100])) # Ánh xạ sang từ 0-95 về 0-100 vì góc càng nhỏ thì càng tiến về trạng thái s3

                # drawing circles and lines on selected points
                if self.drawPoints:
                    cv2.circle(self.img, (x1, y1), 5, (0, 255, 255), 5)
                    cv2.circle(self.img, (x1, y1), 10, (0, 255, 0), 6)
                    cv2.circle(self.img, (x2, y2), 5, (0, 255, 255), 5)
                    cv2.circle(self.img, (x2, y2), 10, (0, 255, 0), 6)
                    cv2.circle(self.img, (x3, y3), 5, (0, 255, 255), 5)
                    cv2.circle(self.img, (x3, y3), 10, (0, 255, 0), 6)
                    cv2.circle(self.img, (x4, y4), 5, (0, 255, 255), 5)
                    cv2.circle(self.img, (x4, y4), 10, (0, 255, 0), 6)
                    cv2.circle(self.img, (x5, y5), 5, (0, 255, 255), 5)
                    cv2.circle(self.img, (x5, y5), 10, (0, 255, 0), 6)
                    cv2.circle(self.img, (x6, y6), 5, (0, 255, 255), 5)
                    cv2.circle(self.img, (x6, y6), 10, (0, 255, 0), 6)
                    cv2.circle(self.img, (x7, y7), 5, (0, 255, 255), 5)
                    cv2.circle(self.img, (x7, y7), 10, (0, 255, 0), 6)
                    cv2.circle(self.img, (x8, y8), 5, (0, 255, 255), 5)
                    cv2.circle(self.img, (x8, y8), 10, (0, 255, 0), 6)

                    cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.line(self.img, (x2, y2), (x3, y3), (0, 0, 255), 4)
                    cv2.line(self.img, (x4, y4), (x5, y5), (0, 0, 255), 4)
                    cv2.line(self.img, (x5, y5), (x6, y6), (0, 0, 255), 4)
                    cv2.line(self.img, (x1, y1), (x4, y4), (0, 0, 255), 4)
                    cv2.line(self.img, (x1, y1), (x7, y7), (0, 0, 255), 4)
                    cv2.line(self.img, (x4, y4), (x8, y8), (0, 0, 255), 4)

                return [leftHandAngle, rightHandAngle, leftBackAngle, rightBackAngle, leftkneeAngleLineAngle, rightkneeAngleLineAngle]
            
        return [0, 0, 0, 0, 0, 0]