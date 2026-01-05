import yaml
import numpy as np

def load_camera_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    camera_matrix = np.array(cfg["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(cfg["dist_coeffs"], dtype=np.float32)

    return {
        "name": cfg.get("camera_name", "unknown"),
        "resolution": (cfg["image_width"], cfg["image_height"]),
        "fps": cfg.get("fps", 30),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "calibrated": cfg["calibration"]["calibrated"]
    }