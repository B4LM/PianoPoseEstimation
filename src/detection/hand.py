import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
import time


class MediaPipeHandDetection:
    '''
    Class to handle hand detection and landmark extraction using MediaPipe.
    '''

    def __init__(self,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize the MediaPipe Hands solution
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.FINGER_TIP_INDICES = {
            'THUMB': 4,
            'INDEX': 8,
            'MIDDLE': 12,
            'RING': 16,
            'PINKY': 20
        }


        self.FINGER_COLORS = {
            'THUMB': (0, 165, 255),  # Orange
            'INDEX': (0, 255, 0),  # Green
            'MIDDLE': (255, 255, 0),  # Cyan
            'RING': (255, 0, 255),  # Magenta
            'PINKY': (0, 0, 255)  # Red
        }

        self.last_hand_position = None
        self.last_detection_time = time.time()
        self.detection_timeout = 0.5  # seconds

    def detect(self, image: np.ndarray) -> Optional[Dict]:
        '''
        Detect one hand in the given image and return the results.
        :param image: BGR image (OpenCV default format)
        :return: MediaPipe results object or None if processing fails
        '''
        if image is None or image.size == 0:
            return None

        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image
            results = self.hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                if time.time() - self.last_detection_time > self.detection_timeout:
                    self.last_hand_position = None
                return None

            hand_landmarks = results.multi_hand_landmarks[0]
            self.last_detection_time = time.time()
            height, width = image.shape[:2]
            landmarks_pixel = []
            landmarks_normalized = []

            world_landmarks = results.multi_hand_world_landmarks[0].landmark

            for landmark in hand_landmarks.landmark:
                x_pixel = int(landmark.x * width)
                y_pixel = int(landmark.y * height)

                landmarks_pixel.append([x_pixel, y_pixel, landmark.z])
                landmarks_normalized.append([landmark.x, landmark.y, landmark.z])

            landmarks_pixel = np.array(landmarks_pixel)
            landmarks_normalized = np.array(landmarks_normalized)

            finger_tips = {}
            for finger_name, tip_index in self.FINGER_TIP_INDICES.items():
                tip = landmarks_pixel[tip_index]
                finger_tips[finger_name] = {
                    'position': tip[:2],  # x, y
                    'depth': tip[2],  # z (relative depth)
                    'index': tip_index
                }

            wrist_pos = landmarks_pixel[0][:2]

            if self.last_hand_position is None:
                alpha = 1.0
                wrist = alpha * wrist_pos
            else:
                alpha = 0.3
                wrist = alpha * wrist_pos + (1 - alpha) * self.last_hand_position

            self.last_hand_position = wrist

            x_coords = landmarks_pixel[:, 0]
            y_coords = landmarks_pixel[:, 1]
            padding = 20

            bbox = {
                'x_min': max(0, int(x_coords.min() - padding)),
                'y_min': max(0, int(y_coords.min() - padding)),
                'x_max': min(width, int(x_coords.max() + padding)),
                'y_max': min(height, int(y_coords.max() + padding)),
                'width': int(x_coords.max() - x_coords.min() + 2 * padding),
                'height': int(y_coords.max() - y_coords.min() + 2 * padding)
            }

            wrist_landmark = landmarks_pixel[0][:2]
            middle_tip = landmarks_pixel[12][:2]
            hand_size = np.linalg.norm(middle_tip - wrist_landmark)


            hand_data = {
                'landmarks_pixel': landmarks_pixel,
                'landmarks_normalized': landmarks_normalized,
                'world_landmarks': world_landmarks,
                'finger_tips': finger_tips,
                'wrist_position': wrist_pos,
                'bbox': bbox,
                'hand_size': hand_size,
                'mp_landmarks': hand_landmarks,
                'detection_time': time.time(),
                'image_shape': (height, width)
            }

            return hand_data
        except Exception as e:
            print(f"Hand detection failed: {e}")
            return None

    def draw_hand(self, image: np.ndarray, hand_data: Dict,
                  draw_landmarks: bool = True,
                  draw_finger_tips: bool = True,
                  draw_bbox: bool = True) -> np.ndarray:
        '''
        Draw hand landmarks, finger tips, and bounding box on the image.
        :param image: BGR image
        :param hand_data: Dictionary with hand detection data
        :param draw_landmarks: Whether to draw landmarks and connections
        :param draw_finger_tips: Whether to draw finger tip circles
        :param draw_bbox: Whether to draw bounding box
        :return: Image with drawings
        '''
        if hand_data is None:
            return image

        output_image = image.copy()

        if draw_landmarks and 'mp_landmarks' in hand_data:
            self.mp_drawing.draw_landmarks(
                output_image,
                hand_data['mp_landmarks'],
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        if draw_finger_tips and 'finger_tips' in hand_data:
            for finger_name, tip_info in hand_data['finger_tips'].items():
                pos = tip_info['position'].astype(int)
                color = self.FINGER_COLORS.get(finger_name, (255, 255, 255))

                cv2.circle(output_image, tuple(pos), 5, color, -1)
                cv2.circle(output_image, tuple(pos), 5, (255, 255, 255), 2)

                label = finger_name[0].upper()
                cv2.putText(output_image, label,
                            (pos[0] - 5, pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if draw_bbox and 'bbox' in hand_data:
            bbox = hand_data['bbox']
            cv2.rectangle(
                output_image,
                (bbox['x_min'], bbox['y_min']),
                (bbox['x_max'], bbox['y_max']),
                (0, 255, 255),
                2
            )

        if 'wrist_position' in hand_data:
            wrist = hand_data['wrist_position'].astype(int)
            cv2.circle(output_image, tuple(wrist), 8, (255, 0, 255), -1)

        return output_image