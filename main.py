from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import time

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

count_left = 0
count_right = 0
stage_left = None
stage_right = None
start_time = None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def angleco(a, b, c):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c

    A = (x2 - x1, y2 - y1)
    C = (x3 - x1, y3 - y1)

    dot_product = A[0] * C[0] + A[1] * C[1]

    magnitude_A = math.sqrt(A[0] ** 2 + A[1] ** 2)
    magnitude_C = math.sqrt(C[0] ** 2 + C[1] ** 2)

    if magnitude_A == 0 or magnitude_C == 0:
        return 0

    cos_theta = dot_product / (magnitude_A * magnitude_C)

    cos_theta = max(-1, min(1, cos_theta))

    theta_radians = math.acos(cos_theta)

    theta_degrees = math.degrees(theta_radians)

    return theta_degrees

def one_arm_row_tracking():
    global count_left, count_right, stage_left, stage_right, start_time

    start_time = time.time()
    cap = cv2.VideoCapture(0)

    with (mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False,
                       smooth_landmarks=True) as pose):
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks and len(results.pose_landmarks.landmark) >= 33:
                    landmarks = results.pose_landmarks.landmark
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        landmarks = results.pose_landmarks.landmark
                        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]


                        for joint in [right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist]:
                            cv2.circle(frame,
                                       (int(joint.x * frame.shape[1]), int(joint.y * frame.shape[0])),
                                       10, (255, 0, 0), -1)
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                    angle_right = calculate_angle(
                        (right_shoulder.x, right_shoulder.y),
                        (right_elbow.x, right_elbow.y),
                        (right_wrist.x, right_wrist.y)
                    )

                    angleco1 = angleco(
                        (right_shoulder.x, right_shoulder.y),
                        (right_elbow.x, right_elbow.y),
                        (right_wrist.x, right_wrist.y)
                    )

                    if angle_right >= 160:
                        stage_right = "down"
                    if angle_right <= 90 and stage_right == "down" :
                        stage_right = "up"
                        count_right += 1

                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

                    angle_left = calculate_angle(
                        (left_shoulder.x, left_shoulder.y),
                        (left_elbow.x, left_elbow.y),
                        (left_wrist.x, left_wrist.y)
                    )

                    angleco1 = angleco(
                        (left_shoulder.x, left_shoulder.y),
                        (left_elbow.x, left_elbow.y),
                        (left_wrist.x, left_wrist.y)
                    )

                    if angle_left >= 160:
                        stage_left = "down"
                    if angle_left <= 90 and stage_left == "down" :
                        stage_left = "up"
                        count_left += 1

                    if angleco1 < 3:
                        text = "Keep your elbow away from your body"
                    elif angleco1 > 10:
                        text = "Bring your elbows slightly closer to your body"
                    else:
                        text = None

                    if text:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.5
                        thickness = 2
                        color = (0, 0, 0)
                        background_color = (255, 255, 255)
                        border_color = (255, 255, 255)

                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        text_x = (frame.shape[1] - text_width) // 2
                        text_y = (frame.shape[0] + text_height) // 2

                        rect_x1 = text_x - 10
                        rect_y1 = text_y - text_height - 10
                        rect_x2 = text_x + text_width + 10
                        rect_y2 = text_y + baseline + 10

                        cv2.rectangle(frame, (rect_x1 - 2, rect_y1 - 2), (rect_x2 + 2, rect_y2 + 2), border_color, 2)
                        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)

                        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                timer_text = f"{minutes:02}:{seconds:02}"

                timer_background_x1 = frame.shape[1] - 200
                timer_background_y1 = 10
                timer_background_x2 = frame.shape[1] - 10
                timer_background_y2 = 60

                cv2.rectangle(frame, (timer_background_x1, timer_background_y1), (timer_background_x2, timer_background_y2),
                              (255, 255, 255), -1)

                cv2.putText(frame, timer_text, (frame.shape[1] - 190, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
                            cv2.LINE_AA)
                circle_center2 = (200, 200)
                circle_radius2 = 120
                circle_color2 = (255, 255, 255)

                circle_center = circle_center2
                circle_radius = 100
                circle_color = (200, 200, 200)

                text_color = (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                thickness = 4

                cv2.circle(frame, circle_center2, circle_radius2, circle_color2, -1)

                cv2.circle(frame, circle_center, circle_radius, circle_color, -1)

                text = str(count_left + count_right)
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (circle_center[0] - text_size[0] // 2)
                text_y = circle_center[1] + text_size[1] // 2
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

                total_count = count_left + count_right
                if total_count % 5 == 0 and total_count != 0:
                    message = f"Keep going you can Continue to {total_count + 5}"
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 2.5
                    thickness = 4
                    text_color = (18, 111, 249)

                    background_color = (249, 111, 128)

                    (text_width, text_height), _ = cv2.getTextSize(message, font, font_scale, thickness)
                    text_x = (frame.shape[1] - text_width) // 2
                    text_y = (frame.shape[0] + text_height) // 2

                    rect_x1 = text_x - 30
                    rect_y1 = text_y - text_height - 30
                    rect_x2 = text_x + text_width + 30
                    rect_y2 = text_y + 30

                    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)
                    cv2.putText(frame, message, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    frame_data = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')
        finally:
            cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(one_arm_row_tracking(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_one_arm_row', methods=['GET'])
def start_one_arm_row():
    threading.Thread(target=one_arm_row_tracking).start()
    return jsonify({"message": "One Arm Row Tracking started"})

@app.route('/stop_exercise', methods=['GET'])
def stop_exercise():
    global count_left, count_right
    total_count = count_left + count_right
    return jsonify({"message": "Exercise stopped", "total_count": total_count})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)