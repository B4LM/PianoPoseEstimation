import yaml
import numpy as np
from datetime import datetime


class CameraManager:
    def __init__(self, cfg_path = "camera.yaml"):
        self.cfg_path = cfg_path
        self.cameras = {}
        self.current_camera = None
        self.load_cfg()

    def load_cfg(self):
        "load camera config from camera-yaml file"
        with open(self.cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.cameras = self.cfg.get("cameras", {})
        default_name = self.cfg.get("default_camera", "laptop_webcam")

        if default_name in self.cameras:
            self.current_camera = default_name
        else:
            self.current_camera = list(self.cameras.keys())[0]

        print(f"Loaded {len(self.cameras)} cameras. Current camera is {self.current_camera}")

    def save_cfg(self):
        "save camera config to camera-yaml file"
        self.cfg["cameras"] = self.cameras
        with open(self.cfg_path, "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)
        print(f"Saved camera config to {self.cfg_path}")

    def switch_camera(self, camera_name):
        "switch camera from current camera to new one"
        if camera_name in self.cameras:
            self.current_camera = camera_name
            print(f"Switching to camera {camera_name}")
            return True
        else:
            print(f"Camera {camera_name} not found.")
            return False

    def get_camera_names(self):
        "get camera names"
        return list(self.cameras.keys())

    def get_current_camera_cfg(self):
        "get current camera config"
        return self.cameras.get(self.current_camera, {})

    def get_camera_params(self, camera_name = None):
        "get camera params"
        if camera_name is None:
            camera_name = self.current_camera

        cfg = self.cameras[camera_name]
        camera_matrix= np.array(cfg["camera_matrix"], dtype = np.float32)
        dist_coeffs = np.array(cfg["dist_coeffs"], dtype = np.float32)

        return camera_matrix, dist_coeffs

    def update_calibration(self, camera_name, camera_matrix, dist_coeffs, mean_error):
        "update camera calibration"
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
