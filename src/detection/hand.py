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

        # Initialize MediaPipe modules - IMPORTANT: Use direct imports
        # Don't store them as attributes, use them directly
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Adjustable offset (you can tweak these!)
        self.debug_offset_x = 0  # pixels
        self.debug_offset_y = -50  # Move 50 pixels UP from wrist

        # Initialize the MediaPipe Hands solution
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Define landmark names
        self.LANDMARK_NAMES = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]

        self.FINGER_TIP_INDICES = {
            'THUMB': 4,
            'INDEX': 8,
            'MIDDLE': 12,
            'RING': 16,
            'PINKY': 20
        }

        self.FINGER_NAMES = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']

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
                  draw_bbox: bool = True,
                  draw_debug_lines: bool = True,
                  tag_offset: np.ndarray = None) -> np.ndarray:
        '''
        Draw hand landmarks, finger tips, and bounding box on the image.
        :param image: BGR image
        :param hand_data: Dictionary with hand detection data
        :param draw_landmarks: Whether to draw landmarks and connections
        :param draw_finger_tips: Whether to draw finger tip circles
        :param draw_bbox: Whether to draw bounding box
        :param draw_debug_lines: Whether to draw debug lines
        :param tag_offset: Offset of the tag
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

                cv2.circle(output_image, tuple(pos), 10, color, -1)
                cv2.circle(output_image, tuple(pos), 10, (255, 255, 255), 2)

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

        ####
        '''
        if 'wrist_position' in hand_data:
            wrist = hand_data['wrist_position'].astype(int)
            cv2.circle(output_image, tuple(wrist), 8, (255, 0, 255), -1)
            cv2.putText(output_image, "WRIST",
                        (wrist[0] + 10, wrist[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Draw debug lines for tag offset
        if draw_debug_lines and 'wrist_position' in hand_data:
            wrist = hand_data['wrist_position'].astype(int)

            # Default offset if none provided
            if tag_offset is None:
                # These are IN HAND COORDINATES (relative to hand orientation)
                # [towards fingers, towards pinky side] in hand-width units
                tag_offset = np.array([0.3, 0.0])  # 0.3 hand-width towards fingers

            # Calculate hand orientation and scale
            # Get middle finger base and tip for orientation
            middle_mcp = hand_data['landmarks_pixel'][9][:2].astype(int)  # Middle finger MCP
            middle_tip = hand_data['landmarks_pixel'][12][:2].astype(int)  # Middle finger tip

            # Vector from wrist to middle finger MCP gives hand direction
            wrist_to_middle = middle_mcp - wrist

            # Calculate hand width (approximate)
            pinky_mcp = hand_data['landmarks_pixel'][17][:2]  # Pinky MCP
            index_mcp = hand_data['landmarks_pixel'][5][:2]  # Index MCP
            hand_width_pixels = np.linalg.norm(pinky_mcp - index_mcp)

            if hand_width_pixels > 0:
                # Normalize the direction vector
                hand_direction = wrist_to_middle / np.linalg.norm(wrist_to_middle)

                # Perpendicular vector (90 degree rotation for side direction)
                # In image coordinates, rotate 90 degrees clockwise for hand side
                hand_side = np.array([hand_direction[1], -hand_direction[0]])

                # Convert hand-relative offset to pixel offset
                offset_pixels = (
                        tag_offset[0] * hand_width_pixels * hand_direction +  # Towards fingers
                        tag_offset[1] * hand_width_pixels * hand_side  # Towards pinky/thumb
                )

                # Calculate tag position
                tag_pos = wrist + offset_pixels.astype(int)

                # Draw hand orientation vectors (for debugging)
                # Hand direction (red)
                dir_end = wrist + (hand_direction * 50).astype(int)
                cv2.arrowedLine(output_image, tuple(wrist), tuple(dir_end),
                                (0, 0, 255), 2, tipLength=0.2)
                cv2.putText(output_image, "Fingers",
                            tuple(dir_end + np.array([5, 5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Hand side direction (green)
                side_end = wrist + (hand_side * 50).astype(int)
                cv2.arrowedLine(output_image, tuple(wrist), tuple(side_end),
                                (0, 255, 0), 2, tipLength=0.2)
                cv2.putText(output_image, "Thumb side",
                            tuple(side_end + np.array([5, 5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Draw tag position
                cv2.circle(output_image, tuple(tag_pos), 12, (0, 255, 255), -1)  # Yellow
                cv2.circle(output_image, tuple(tag_pos), 12, (0, 0, 0), 2)  # Black border

                cv2.putText(output_image, "TAG HERE",
                            (tag_pos[0] + 15, tag_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw line from wrist to tag
                cv2.arrowedLine(output_image,
                                tuple(wrist),
                                tuple(tag_pos),
                                (255, 255, 0),  # Cyan
                                3,
                                tipLength=0.15)

                # Display offset values
                offset_text = f"Offset: ({tag_offset[0]:.2f}, {tag_offset[1]:.2f}) hw"
                cv2.putText(output_image, offset_text,
                            (wrist[0] + 10, wrist[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Display hand width
                cv2.putText(output_image, f"Hand width: {hand_width_pixels:.0f} px",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ####
        '''
        return output_image

    def get_landmarks_3d_relative(self, hand_data: Dict, wrist_depth: float = None, hand_width: float = None) -> np.ndarray:
        '''
        Convert 2D landmarks to 3D relative coordinates.
        :param hand_data: hand detection data
        :param hand_width: distance between index and pinky knuckle, used for depth estimation, eather this or wrist_depth must be given
        :param wrist_depth: distance between wrist and camera, eather this or hand_width must be given
        :return: Nx3 array of landmarks in relative 3D coordinates
        '''
        if hand_data is None:
            return None

        landmarks_2d = hand_data['landmarks_normalized']
        landmarks_3d = []

        if hand_width is not None and wrist_depth is None:
            #depth estimation through hand width
            hand_width



        landmarks_2d = hand_data['landmarks_normalized']
        landmarks_3d = []

        for landmark in landmarks_2d:
            # x, y in normalized coordinates, z from MediaPipe + depth
            x = landmark[0]  # Normalized [0, 1]
            y = landmark[1]  # Normalized [0, 1]
            z = depth + landmark[2] * 0.1  # Add MediaPipe's relative z scaled

            landmarks_3d.append([x, y, z])

        return np.array(landmarks_3d)

    def estimate_hand_depth(self, hand_data: Dict, focal_length: float = 500) -> float:
        '''
        Estimate hand depth using hand size.
        :param hand_data: hand detection data
        :param focal_length: camera focal length in pixels
        :return: estimated depth in meters
        '''
        if hand_data is None:
            return None

        # Average adult hand width ~ 0.08-0.1m
        avg_hand_width = 0.085  # meters
        bbox = hand_data['bbox']
        hand_width_pixels = bbox['width']

        # Depth = (focal_length * actual_size) / pixel_size
        depth = (focal_length * avg_hand_width) / hand_width_pixels

        return depth

