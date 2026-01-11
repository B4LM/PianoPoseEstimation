import yaml
import numpy as np
from datetime import datetime

# CameraManager class to handle multiple camera configurations
class CameraManager:
    def __init__(self, cfg_path = "camera.yaml"):
        self.cfg_path = cfg_path
        self.cameras = {}
        self.current_camera = None
        self.load_cfg()

    # Load camera configuration from YAML file
    def load_cfg(self):

        with open(self.cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.cameras = self.cfg.get("cameras", {})
        default_name = self.cfg.get("default_camera", "laptop_webcam")

        if default_name in self.cameras:
            self.current_camera = default_name
        else:
            self.current_camera = list(self.cameras.keys())[0]

        print(f"Loaded {len(self.cameras)} cameras. Current camera is {self.current_camera}")

    # Save camera configuration to YAML file
    def save_cfg(self):

        self.cfg["cameras"] = self.cameras
        with open(self.cfg_path, "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)
        print(f"Saved camera config to {self.cfg_path}")

    # Switch to a different camera
    def switch_camera(self, camera_name):

        if camera_name in self.cameras:
            self.current_camera = camera_name
            print(f"Switching to camera {camera_name}")
            return True
        else:
            print(f"Camera {camera_name} not found.")
            return False

    # Get list of camera names
    def get_camera_names(self):

        return list(self.cameras.keys())

    # Get current camera configuration
    def get_current_camera_cfg(self):

        return self.cameras.get(self.current_camera, {})

    # Get camera parameters (camera matrix and distortion coefficients)
    def get_camera_params(self, camera_name = None):

        if camera_name is None:
            camera_name = self.current_camera

        cfg = self.cameras[camera_name]
        camera_matrix= np.array(cfg["camera_matrix"], dtype = np.float32)
        dist_coeffs = np.array(cfg["dist_coeffs"], dtype = np.float32)

        return camera_matrix, dist_coeffs

    # Update camera calibration data
    def update_calibration(self, camera_name, camera_matrix, dist_coeffs, mean_error):

        if camera_name not in self.cameras:
            print(f"Camera {camera_name} not found.")
            return False

        self.cameras[camera_name]["camera_matrix"] = camera_matrix.tolist()
        self.cameras[camera_name]["dist_coeffs"] = dist_coeffs.tolist()

        self.cameras[camera_name]["calibration"]["calibrated"] = True
        self.cameras[camera_name]["calibration"]["calibration_date"] = datetime.now().isoformat()

        self.cameras[camera_name]["calibration"]["mean_error"] = float(mean_error)
        #self.cameras[camera_name]["calibration"]["calibration_images"] = calibration_images

        self.save_cfg()
        print(f"Calibration updated for {camera_name}, mean_error: {mean_error:.4f}")
        return True
