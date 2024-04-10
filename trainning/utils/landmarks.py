def recalculate_landmarks(landmarks):
    """
    Tịnh tiến thân người vào giữa bức hình, đồng thời dời lại trục toạ độ
    """
    hip_center_x = float((landmarks[23].x + landmarks[24].x) / 2)
    hip_center_y = float((landmarks[23].y + landmarks[24].y) / 2)

    new_center = (0.5, 0.5)
    delta_x = new_center[0] - hip_center_x
    delta_y = new_center[1] - hip_center_y

    for landmark in landmarks:
        landmark.x += delta_x
        landmark.y += delta_y


def draw_landmarks(mp_drawing, mp_pose, image, pose_landmarks):
    """
    Vẽ landmarks lên ảnh
    """
    mp_drawing.draw_landmarks(
        image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # Màu sắc của các landmark
            thickness=5,  # Độ dày của các đường nối landmark
            circle_radius=5,  # Bán kính của các điểm landmark
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Màu sắc của các đường nối
            thickness=5,  # Độ dày của các đường nối
        ),
    )
