import numpy as np
from pathlib import Path
from pose_format import Pose
from typing import Dict, Any, Union


class PoseExtract:
    def __init__(self, keep_face: bool = False) -> None:
        self.cache = {}
        self.keep_face = keep_face

    def __call__(self, inputs: Union[Dict[str, Any], str, Path]) -> Pose:
        if isinstance(inputs, (str, Path)):
            path_str = str(inputs)
            if path_str in self.cache:
                return self.cache[path_str]
            with open(inputs, "rb") as f:
                pose = Pose.read(f.read())
            
            # Prune components to save RAM
            to_remove = ["POSE_WORLD_LANDMARKS"]
            points_to_remove = None
            if not self.keep_face:
                if "FACE_LANDMARKS" in [c.name for c in pose.header.components]:
                    to_remove.append("FACE_LANDMARKS")
            else:
                # Remove face points we don't need
                all_face_points = [str(i) for i in range(468)]
                face_landmarks_to_keep = [
                    "70", "105", "336", "334", "33", "133", "159", "145", "362", "263",
                    "386", "374", "61", "291", "0", "17", "13", "14", "37", "267"
                ]
                points_to_remove = {"FACE_LANDMARKS": [p for p in all_face_points if p not in face_landmarks_to_keep]}
            
            pose = pose.remove_components(to_remove, points_to_remove)
            self.cache[path_str] = pose
            return pose
        else:
            from pose_format.utils.holistic import load_holistic
            pose = load_holistic(
                frames=inputs["frames"],
                fps=inputs["fps"],
                width=inputs["width"],
                height=inputs["height"],
                progress=False,
            )
        return pose


class PoseInterpolate:
    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

    def __call__(self, pose: Pose) -> Pose:
        data = pose.body.data  # shape (T, 1, V, C)
        conf = pose.body.confidence  # shape (T, 1, V)
        
        T, _, V, C = data.shape
        
        for v in range(V):
            # Check if all coordinate values are 0 or confidence score is below threshold
            is_zero = np.all(data[:, 0, v, :2] == 0, axis=1)
            is_low_conf = conf[:, 0, v] < self.confidence_threshold
            missing_mask = is_zero | is_low_conf
            
            valid_indices = np.where(~missing_mask)[0]
            if len(valid_indices) == 0:
                continue
                
            if len(valid_indices) < T:
                for c in range(C):
                    coords = data[:, 0, v, c]
                    data[:, 0, v, c] = np.interp(
                        np.arange(T),
                        valid_indices,
                        coords[valid_indices]
                    )
                # Mark confidence of interpolated values as threshold
                conf[missing_mask, 0, v] = self.confidence_threshold
                
        return pose
