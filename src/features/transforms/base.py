import numpy as np
from pathlib import Path
from pose_format import Pose
from pose_format.utils.holistic import load_holistic
from typing import Dict, Any, Union


class PoseExtract:
    def __call__(self, inputs: Union[Dict[str, Any], str, Path]) -> Pose:
        if isinstance(inputs, (str, Path)):
            with open(inputs, "rb") as f:
                pose = Pose.read(f.read())
        else:
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
