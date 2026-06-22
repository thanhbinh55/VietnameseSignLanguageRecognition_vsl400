import torch
import logging
import numpy as np
from pose_format import Pose
from utils import BODY_LANDMARKS, HAND_LANDMARKS, LANDMARKS, HANDS_LANDMARKS


def get_spoter_landmarks(num_landmarks: int) -> list:
    if num_landmarks == 54:
        return LANDMARKS
    elif num_landmarks == 74:
        face_names = [f"face_{i}" for i in range(20)]
        return BODY_LANDMARKS + face_names + HANDS_LANDMARKS
    else:
        return LANDMARKS


class SPOTERJointSelect:
    def __init__(self, include_face: bool = False) -> None:
        self.include_face = include_face
        self.pose_landmarks = [
            "NOSE",
            "NECK",
            "RIGHT_EYE",
            "LEFT_EYE",
            "RIGHT_EAR",
            "LEFT_EAR",
            "RIGHT_SHOULDER",
            "LEFT_SHOULDER",
            "RIGHT_ELBOW",
            "LEFT_ELBOW",
            "RIGHT_WRIST",
            "LEFT_WRIST",
        ]
        self.face_landmarks = [
            "70", "105", "336", "334",  # Eyebrows
            "33", "133", "159", "145", "362", "263", "386", "374",  # Eyes
            "61", "291", "0", "17", "13", "14", "37", "267"  # Mouth corners
        ]
        self.hand_landmarks = [
            "WRIST",
            "INDEX_FINGER_TIP",
            "INDEX_FINGER_DIP",
            "INDEX_FINGER_PIP",
            "INDEX_FINGER_MCP",
            "MIDDLE_FINGER_TIP",
            "MIDDLE_FINGER_DIP",
            "MIDDLE_FINGER_PIP",
            "MIDDLE_FINGER_MCP",
            "RING_FINGER_TIP",
            "RING_FINGER_DIP",
            "RING_FINGER_PIP",
            "RING_FINGER_MCP",
            "PINKY_DIP",
            "PINKY_TIP",
            "PINKY_PIP",
            "PINKY_MCP",
            "THUMB_TIP",
            "THUMB_IP",
            "THUMB_MCP",
            "THUMB_CMC",
        ]

    def __get_point(self, component: str, point: str, pose: Pose) -> np.ndarray:
        if point == "NECK":
            return np.zeros_like(pose.body.data[:, 0, 0, :2].data)
        idx = pose.header._get_point_index(component, point)
        return pose.body.data[:, 0, idx, :2].data

    def __call__(self, pose: Pose) -> torch.Tensor:
        data = []
        for point in self.pose_landmarks:
            data.append(self.__get_point("POSE_LANDMARKS", point, pose))
        if self.include_face:
            for point in self.face_landmarks:
                data.append(self.__get_point("FACE_LANDMARKS", point, pose))
        for side in ["LEFT", "RIGHT"]:
            for point in self.hand_landmarks:
                data.append(self.__get_point(f"{side}_HAND_LANDMARKS", point, pose))
        # (num_landmarks, num_frames, 2) -> (num_frames, num_landmarks, 2)
        data = np.array(data).transpose((1, 0, 2))
        return torch.from_numpy(data)


class SPOTERPad:
    def __init__(self, num_frames: int = 150) -> None:
        self.num_frames = num_frames

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        padded_data = torch.zeros((self.num_frames, data.shape[1], data.shape[2]))
        L = data.shape[0]
        if L < self.num_frames:
            padded_data[:L, :, :] = data
            rest = self.num_frames - L
            num = int(np.ceil(rest / L))
            pad = torch.concatenate([data for _ in range(num)], 0)[:rest]
            padded_data[L:, :, :] = pad
        else:
            padded_data[:, :, :] = data[:self.num_frames, :, :]
        return padded_data


class SPOTERShift:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Move the landmark position interval to improve performance
        """
        return data - 0.5


class SPOTERTensorToDict:
    def __call__(self, data: torch.Tensor) -> dict:
        """
        Converts the tensor representation of the pose data into a dictionary.

        :param data: np.ndarray/Tensor containing the pose data
        :return: Dictionary with the pose data
        """
        data_array = data.numpy() if isinstance(data, torch.Tensor) else data
        output = {}
        landmarks_list = get_spoter_landmarks(data_array.shape[1])
        for idx, landmark in enumerate(landmarks_list):
            output[landmark] = data_array[:, idx]
        return output


class SPOTERDictToTensor:
    def __call__(self, data: dict) -> torch.Tensor:
        """
        Converts the dictionary representation of the pose data into a tensor.

        :param data: Dictionary containing the pose data
        :return: Tensor with the pose data
        """
        if "face_0" in data:
            landmarks_list = get_spoter_landmarks(74)
        else:
            landmarks_list = get_spoter_landmarks(54)
            
        first_key = landmarks_list[0]
        output = np.empty((len(data[first_key]), len(landmarks_list), 2))
        for idx, landmark in enumerate(landmarks_list):
            output[:, idx, 0] = [frame[0] for frame in data[landmark]]
            output[:, idx, 1] = [frame[1] for frame in data[landmark]]
        return torch.from_numpy(output)


class SPOTERSingleBodyDictNormalize:
    def __init__(self, anchor: str = "box") -> None:
        self.anchor = anchor

    def __call__(self, row: dict) -> dict:
        """
        Normalizes the skeletal data for a given sequence of frames with signer's body pose data. The normalization follows
        the definition from our paper.

        :param row: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                    that particular joints coordinates
        :return: Dictionary with normalized skeletal data (following the same schema as input data)
        """
        sequence_size = len(row["leftEar"])
        valid_sequence = True
        original_row = row

        last_starting_point, last_ending_point = None, None
        last_anchor_point, last_head_metric = None, None

        # Treat each element of the sequence (analyzed frame) individually
        for sequence_index in range(sequence_size):
            has_shoulders = (
                row["leftShoulder"][sequence_index][0] != 0
                and row["rightShoulder"][sequence_index][0] != 0
            )
            has_neck_nose = (
                row["neck"][sequence_index][0] != 0
                and row["nose"][sequence_index][0] != 0
            )
            
            anchor_missing = False
            if self.anchor == "neck" and row["neck"][sequence_index][0] == 0:
                anchor_missing = True
            elif self.anchor == "nose" and row["nose"][sequence_index][0] == 0:
                anchor_missing = True

            # Prevent from even starting the analysis if some necessary elements are not present
            if (not has_shoulders and not has_neck_nose) or anchor_missing:
                if self.anchor == "box" and not last_starting_point:
                    valid_sequence = False
                    continue
                elif self.anchor != "box" and not last_anchor_point:
                    valid_sequence = False
                    continue
                else:
                    if self.anchor == "box":
                        starting_point, ending_point = last_starting_point, last_ending_point
                    else:
                        anchor_point, head_metric = last_anchor_point, last_head_metric

            else:
                if (
                    row["leftShoulder"][sequence_index][0] != 0
                    and row["rightShoulder"][sequence_index][0] != 0
                ):
                    left_shoulder = (
                        row["leftShoulder"][sequence_index][0],
                        row["leftShoulder"][sequence_index][1],
                    )
                    right_shoulder = (
                        row["rightShoulder"][sequence_index][0],
                        row["rightShoulder"][sequence_index][1],
                    )
                    shoulder_distance = (
                        ((left_shoulder[0] - right_shoulder[0]) ** 2)
                        + ((left_shoulder[1] - right_shoulder[1]) ** 2)
                    ) ** 0.5
                    head_metric = shoulder_distance
                else:
                    neck = (row["neck"][sequence_index][0], row["neck"][sequence_index][1])
                    nose = (row["nose"][sequence_index][0], row["nose"][sequence_index][1])
                    neck_nose_distance = (
                        ((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)
                    ) ** 0.5
                    head_metric = neck_nose_distance

                if self.anchor == "box":
                    # Set the starting and ending point of the normalization bounding box
                    starting_point = [
                        row["neck"][sequence_index][0] - 3 * head_metric,
                        row["leftEye"][sequence_index][1] + head_metric,
                    ]
                    ending_point = [
                        row["neck"][sequence_index][0] + 3 * head_metric,
                        starting_point[1] - 6 * head_metric,
                    ]
                    last_starting_point, last_ending_point = starting_point, ending_point
                else:
                    if self.anchor == "neck":
                        anchor_point = [row["neck"][sequence_index][0], row["neck"][sequence_index][1]]
                    elif self.anchor == "nose":
                        anchor_point = [row["nose"][sequence_index][0], row["nose"][sequence_index][1]]
                    last_anchor_point, last_head_metric = anchor_point, head_metric

            if self.anchor == "box":
                # Ensure that all of the bounding-box-defining coordinates are not out of the picture
                if starting_point[0] < 0:
                    starting_point[0] = 0
                if starting_point[1] < 0:
                    starting_point[1] = 0
                if ending_point[0] < 0:
                    ending_point[0] = 0
                if ending_point[1] < 0:
                    ending_point[1] = 0

            # Normalize individual landmarks and save the results
            keys_to_normalize = list(BODY_LANDMARKS)
            if "face_0" in row:
                keys_to_normalize.extend([f"face_{i}" for i in range(20)])

            for identifier in keys_to_normalize:
                key = identifier

                # Prevent from trying to normalize incorrectly captured points
                if row[key][sequence_index][0] == 0:
                    continue

                if self.anchor == "box":
                    if any(
                        [
                            (ending_point[0] - starting_point[0]) == 0,
                            (starting_point[1] - ending_point[1]) == 0,
                        ]
                    ):
                        logging.info(f"Problematic normalization with {key}")
                        valid_sequence = False
                        break

                    normalized_x = (row[key][sequence_index][0] - starting_point[0]) / (
                        ending_point[0] - starting_point[0]
                    )
                    normalized_y = (row[key][sequence_index][1] - ending_point[1]) / (
                        starting_point[1] - ending_point[1]
                    )
                else:
                    if head_metric == 0:
                        logging.info(f"Problematic normalization with {key}")
                        valid_sequence = False
                        break
                    normalized_x = (row[key][sequence_index][0] - anchor_point[0]) / head_metric
                    normalized_y = (row[key][sequence_index][1] - anchor_point[1]) / head_metric

                row[key][sequence_index] = list(row[key][sequence_index])

                row[key][sequence_index][0] = normalized_x
                row[key][sequence_index][1] = normalized_y

        if valid_sequence:
            return row
        return original_row


class SPOTERSingleHandDictNormalize:
    def __call__(self, row: dict) -> dict:
        """
        Normalizes the skeletal data for a given sequence of frames with signer's hand pose data. The normalization follows
        the definition from our paper.

        :param row: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                    that particular joints coordinates
        :return: Dictionary with normalized skeletal data (following the same schema as input data)
        """

        hand_landmarks = {0: [], 1: []}

        # Determine how many hands are present in the dataset
        range_hand_size = 1
        if "wrist_1" in row.keys():
            range_hand_size = 2

        # Construct the relevant identifiers
        for identifier in HAND_LANDMARKS:
            for hand_index in range(range_hand_size):
                hand_landmarks[hand_index].append(identifier + "_" + str(hand_index))

        # Treat each hand individually
        for hand_index in range(range_hand_size):

            sequence_size = len(row["wrist_" + str(hand_index)])

            wrist_key = "wrist_" + str(hand_index)

            # Treat each element of the sequence (analyzed frame) individually
            for sequence_index in range(sequence_size):

                # Prevent from trying to normalize if wrist is not captured
                if row[wrist_key][sequence_index][0] == 0:
                    continue

                wrist_x = row[wrist_key][sequence_index][0]
                wrist_y = row[wrist_key][sequence_index][1]

                # Normalize individual landmarks by shifting relative to wrist
                for identifier in HAND_LANDMARKS:
                    key = identifier + "_" + str(hand_index)

                    # Prevent from trying to normalize incorrectly captured points
                    if row[key][sequence_index][0] == 0:
                        continue

                    row[key][sequence_index] = list(row[key][sequence_index])
                    row[key][sequence_index][0] -= wrist_x
                    row[key][sequence_index][1] -= wrist_y

        return row
