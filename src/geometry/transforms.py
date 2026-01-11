import numpy as np
import cv2
import math
from typing import Optional, Dict, Tuple

class CoordinateTransformer:
    def __init__(self, camera_matrix, dist_coeffs):

        # Camera intrinsic parameters
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    # Transform hand pose to piano coordinate frame
    def hand_to_piano_transform(self,
                                tags: Dict[int, Dict],
                                piano_tag_id: int,
                                hand_tag_id: int) -> Optional[Dict]:

        if piano_tag_id not in tags or hand_tag_id not in tags:
            return None

        # Get poses
        piano_tag = tags[piano_tag_id]
        hand_tag = tags[hand_tag_id]

        # Compute relative pose of hand with respect to piano
        relative_3d = None
        if piano_tag['pose'] is not None and hand_tag['pose'] is not None:
            R_piano = piano_tag['pose']['rotation']
            t_piano = piano_tag['pose']['translation']
            R_hand = hand_tag['pose']['rotation']
            t_hand = hand_tag['pose']['translation']

            R_piano_inv = R_piano.T #since rotational matrix is orthogonal

            relative_translation = R_piano_inv @ (t_hand - t_piano)
            relative_rotation = R_piano_inv @ R_hand

            relative_3d = {
                'translation': relative_translation,
                'rotation': relative_rotation,
                'distance': np.linalg.norm(relative_translation)
            }

        return relative_3d

    # Transform world landmark coordinates to piano frame
    def worldlandmark_to_piano_transform(
            self,
            t_lm_to_hand,
            R_lm_to_hand,
            LM_pos,
            hand_pose
    ):

        LM_pos_array = np.array(
            [LM_pos.x, LM_pos.y, LM_pos.z],
            dtype=np.float32
        )

        R_hand_to_piano = np.array(hand_pose['rotation'])
        t_hand_to_piano = np.array(hand_pose['translation'])

        lm_pos_piano = (
                R_hand_to_piano @ (R_lm_to_hand @ LM_pos_array + t_lm_to_hand)
                + t_hand_to_piano
        )

        return lm_pos_piano

    # Get 2D projected axes of AprilTag for visualization
    def get_apriltag_axes(self, tag_pose, axis_length: float = 0.05):
        axes_3d = np.array([
            [0, 0, 0],  # origin
            [axis_length, 0, 0],  # +X
            [0, axis_length, 0],  # +Y
            [0, 0, axis_length],  # +Z
        ], dtype=np.float32)

        R = np.array(tag_pose['rotation'])
        t = np.array(tag_pose['translation']).reshape(1, 3)
        axes_cam = (R @ axes_3d.T).T + t
        pts = (self.camera_matrix @ axes_cam.T).T
        pts = pts[:, :2] / pts[:, 2:3]
        axis_2d_pts = pts.astype(int)

        return axis_2d_pts











