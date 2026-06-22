import cv2
import math
import torch
import random
import logging
import numpy as np
from utils import BODY_LANDMARKS, HANDS_LANDMARKS, ARM_LANDMARKS_ORDER


class SPOTERGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        message = self.__class__.__name__ + '(mean={0}, std={1})'
        return message.format(self.mean, self.std)


class SPOTERAugment:
    def wrap_sign_into_row(self, body_landmarks: dict, handmarks: dict) -> dict:
        """
        Supplementary method for merging body and hand data into a single dictionary.
        """
        return {**body_landmarks, **handmarks}

    def rotate(self, origin: tuple, point: tuple, angle: float) -> tuple:
        """
        Rotates a point counterclockwise by a given angle around a given origin.

        :param origin: Landmark in the (X, Y) format of the origin from which to count angle of rotation
        :param point: Landmark in the (X, Y) format to be rotated
        :param angle: Angle under which the point shall be rotated
        :return: New landmarks (coordinates)
        """
        ox, oy = origin
        px, py = point
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def dict_to_numpy(self, landmarks_dict: dict) -> np.ndarray:
        """
        Supplementary method converting dictionaries of body landmark data into respective NumPy arrays. The resulting array
        will match the order of the BODY_LANDMARKS list.
        """
        output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_LANDMARKS), 2))
        for landmark_index, identifier in enumerate(BODY_LANDMARKS):
            output[:, landmark_index, 0] = np.array(landmarks_dict[identifier])[:, 0]
            output[:, landmark_index, 1] = np.array(landmarks_dict[identifier])[:, 1]
        return output

    def landmarks_to_numpy(self, landmarks_dict: dict, keys: list) -> np.ndarray:
        T = len(landmarks_dict[keys[0]])
        output = np.empty(shape=(T, len(keys), 2))
        for idx, key in enumerate(keys):
            output[:, idx, 0] = np.array(landmarks_dict[key])[:, 0]
            output[:, idx, 1] = np.array(landmarks_dict[key])[:, 1]
        return output

    def preprocess_row_sign(self, sign: dict) -> tuple:
        """
        Supplementary method splitting the single-dictionary skeletal data into two dictionaries of body and hand landmarks
        respectively.
        """
        if "nose_X" in sign:
            face_keys = [k[:-2] for k in sign.keys() if k.startswith("face_") and k.endswith("_X")]
            body_keys = list(BODY_LANDMARKS) + face_keys
            body_landmarks = {
                identifier: [
                    (x, y)
                    for x, y in zip(
                        sign[identifier + "_X"], sign[identifier + "_Y"]
                    )
                ]
                for identifier in body_keys
                if identifier + "_X" in sign and identifier + "_Y" in sign
            }
            hand_landmarks = {
                identifier: [
                    (x, y)
                    for x, y in zip(
                        sign[identifier + "_X"], sign[identifier + "_Y"]
                    )
                ]
                for identifier in HANDS_LANDMARKS
                if identifier + "_X" in sign and identifier + "_Y" in sign
            }
        else:
            face_keys = [k for k in sign.keys() if k.startswith("face_")]
            body_keys = list(BODY_LANDMARKS) + face_keys
            body_landmarks = {
                identifier: sign[identifier]
                for identifier in body_keys
                if identifier in sign
            }
            hand_landmarks = {
                identifier: sign[identifier]
                for identifier in HANDS_LANDMARKS
                if identifier in sign
            }
        return body_landmarks, hand_landmarks

    def numpy_to_dict(self, data: np.ndarray) -> dict:
        """
        Supplementary method converting a NumPy array of body landmark data into dictionaries. The array data must match the
        order of the BODY_LANDMARKS list.
        """
        output = {}
        for landmark_index, identifier in enumerate(BODY_LANDMARKS):
            output[identifier] = data[:, landmark_index].tolist()
        return output

    def numpy_to_landmarks_dict(self, data: np.ndarray, keys: list) -> dict:
        output = {}
        for idx, key in enumerate(keys):
            output[key] = data[:, idx].tolist()
        return output


class SPOTERRandomAugment:
    def __init__(self, p: float, active_augs: list = None) -> None:
        self.p = p
        all_augs = {
            0: SPOTERRotate((-13, 13)),
            1: SPOTERShear("squeeze", (0, 0.15)),
            2: SPOTERArmJointRotate(0.3, (-4, 4)),
            3: SPOTERShear("perspective", (0, 0.1)),
        }
        if active_augs is not None:
            self.augs = {i: all_augs[i] for i in active_augs if i in all_augs}
        else:
            self.augs = all_augs

    def __call__(self, data: dict) -> dict:
        if random.random() < self.p and len(self.augs) > 0:
            selected_key = random.choice(list(self.augs.keys()))
            data = self.augs[selected_key](data)
        return data


class SPOTERRotate(SPOTERAugment):
    def __init__(self, angle_range: tuple) -> None:
        self.angle_range = angle_range

    def __call__(self, sign: dict) -> dict:
        """
        AUGMENTATION TECHNIQUE. All the joint coordinates in each frame are rotated by a random angle up to 13 degrees with
        the center of rotation lying in the center of the frame, which is equal to [0.5; 0.5].

        :param sign: Dictionary with sequential skeletal data of the signing person
        :param angle_range: Tuple containing the angle range (minimal and maximal angle in degrees) to randomly choose the
                            angle by which the landmarks will be rotated from

        :return: Dictionary with augmented (by rotation) sequential skeletal data of the signing person
        """

        body_landmarks, hand_landmarks = self.preprocess_row_sign(sign)
        angle = math.radians(random.uniform(*self.angle_range))

        body_landmarks = {
            key: [self.rotate((0.5, 0.5), frame, angle) for frame in value]
            for key, value in body_landmarks.items()
        }
        hand_landmarks = {
            key: [self.rotate((0.5, 0.5), frame, angle) for frame in value]
            for key, value in hand_landmarks.items()
        }

        return self.wrap_sign_into_row(body_landmarks, hand_landmarks)


class SPOTERShear(SPOTERAugment):
    def __init__(self, shear_type: str, squeeze_ratio: tuple) -> None:
        self.shear_type = shear_type
        self.squeeze_ratio = squeeze_ratio

    def __call__(self, sign: dict) -> dict:
        """
        AUGMENTATION TECHNIQUE.

            - Squeeze. All the frames are squeezed from both horizontal sides. Two different random proportions up to 15% of
            the original frame's width for both left and right side are cut.

            - Perspective transformation. The joint coordinates are projected onto a new plane with a spatially defined
            center of projection, which simulates recording the sign video with a slight tilt. Each time, the right or left
            side, as well as the proportion by which both the width and height will be reduced, are chosen randomly. This
            proportion is selected from a uniform distribution on the [0; 1) interval. Subsequently, the new plane is
            delineated by reducing the width at the desired side and the respective vertical edge (height) at both of its
            adjacent corners.

        :param sign: Dictionary with sequential skeletal data of the signing person
        :param type: Type of shear augmentation to perform (either 'squeeze' or 'perspective')
        :param squeeze_ratio: Tuple containing the relative range from what the proportion of the original width will be
                            randomly chosen. These proportions will either be cut from both sides or used to construct the
                            new projection

        :return: Dictionary with augmented (by squeezing or perspective transformation) sequential skeletal data of the
                signing person
        """

        body_landmarks, hand_landmarks = self.preprocess_row_sign(sign)

        if self.shear_type == "squeeze":
            move_left = random.uniform(*self.squeeze_ratio)
            move_right = random.uniform(*self.squeeze_ratio)

            src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)
            dest = np.array(((0 + move_left, 1), (1 - move_right, 1), (0 + move_left, 0), (1 - move_right, 0)),
                            dtype=np.float32)
            mtx = cv2.getPerspectiveTransform(src, dest)
        elif self.shear_type == "perspective":
            move_ratio = random.uniform(*self.squeeze_ratio)
            src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)

            if random.random() < 0.5:
                dest = np.array(((0 + move_ratio, 1 - move_ratio), (1, 1), (0 + move_ratio, 0 + move_ratio), (1, 0)),
                                dtype=np.float32)
            else:
                dest = np.array(((0, 1), (1 - move_ratio, 1 - move_ratio), (0, 0), (1 - move_ratio, 0 + move_ratio)),
                                dtype=np.float32)

            mtx = cv2.getPerspectiveTransform(src, dest)
        else:
            logging.error("Unsupported shear type provided.")
            return {}

        body_keys = list(body_landmarks.keys())
        body_array = self.landmarks_to_numpy(body_landmarks, body_keys)
        
        hand_keys = list(hand_landmarks.keys())
        hand_array = self.landmarks_to_numpy(hand_landmarks, hand_keys)

        augmented_zero_landmark = cv2.perspectiveTransform(
            np.array([[[0, 0]]], dtype=np.float32),
            mtx
        )[0][0]
        
        augmented_body = cv2.perspectiveTransform(np.array(body_array, dtype=np.float32), mtx)
        augmented_hand = cv2.perspectiveTransform(np.array(hand_array, dtype=np.float32), mtx)

        for t in range(augmented_body.shape[0]):
            is_zero = np.all(augmented_body[t] == augmented_zero_landmark, axis=-1, keepdims=True)
            augmented_body[t] = np.where(is_zero, [0.0, 0.0], augmented_body[t])
            
        for t in range(augmented_hand.shape[0]):
            is_zero = np.all(augmented_hand[t] == augmented_zero_landmark, axis=-1, keepdims=True)
            augmented_hand[t] = np.where(is_zero, [0.0, 0.0], augmented_hand[t])

        body_landmarks = self.numpy_to_landmarks_dict(augmented_body, body_keys)
        hand_landmarks = self.numpy_to_landmarks_dict(augmented_hand, hand_keys)

        return self.wrap_sign_into_row(body_landmarks, hand_landmarks)


class SPOTERArmJointRotate(SPOTERAugment):
    def __init__(self, probability: float, angle_range: tuple) -> None:
        self.probability = probability
        self.angle_range = angle_range

    def __call__(self, sign: dict) -> dict:
        """
        AUGMENTATION TECHNIQUE. The joint coordinates of both arms are passed successively, and the impending landmark is
        slightly rotated with respect to the current one. The chance of each joint to be rotated is 3:10 and the angle of
        alternation is a uniform random angle up to +-4 degrees. This simulates slight, negligible variances in each
        execution of a sign, which do not change its semantic meaning.

        :param sign: Dictionary with sequential skeletal data of the signing person
        :param probability: Probability of each joint to be rotated (float from the range [0, 1])
        :param angle_range: Tuple containing the angle range (minimal and maximal angle in degrees) to randomly choose the
                            angle by which the landmarks will be rotated from

        :return: Dictionary with augmented (by arm joint rotation) sequential skeletal data of the signing person
        """

        body_landmarks, hand_landmarks = self.preprocess_row_sign(sign)

        # Iterate over both directions (both hands)
        for side in ["left", "right"]:
            # Iterate gradually over the landmarks on arm
            for landmark_index, landmark_origin in enumerate(ARM_LANDMARKS_ORDER):
                landmark_origin = landmark_origin.replace("$side$", side)

                # End the process on the current hand if the landmark is not present
                if landmark_origin not in body_landmarks:
                    break

                # Perform rotation by provided probability
                if random.random() < self.probability:
                    angle = math.radians(random.uniform(*self.angle_range))

                    for to_be_rotated in ARM_LANDMARKS_ORDER[landmark_index + 1:]:
                        to_be_rotated = to_be_rotated.replace("$side$", side)

                        # Skip if the landmark is not present
                        if to_be_rotated not in body_landmarks:
                            continue

                        body_landmarks[to_be_rotated] = [
                            self.rotate(
                                body_landmarks[landmark_origin][frame_index], frame, angle
                            )
                            for frame_index, frame in enumerate(body_landmarks[to_be_rotated])
                        ]

        return self.wrap_sign_into_row(body_landmarks, hand_landmarks)
