import cv2
import numpy as np


def draw_text_on_image(
    image: np.ndarray,
    text: str,
    position: tuple[int, int] = (20, 20),
    color: tuple[int, int, int] = (0, 0, 255),
    font_size: int = 20,
) -> np.ndarray:
    """
    Draws text on a numpy image array using OpenCV.
    """
    # Create a copy to avoid in-place modification of the original frame
    image_copy = image.copy()

    # Map font_size to cv2 fontScale (approximate 20px font_size is scale 0.7)
    font_scale = font_size / 28.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Draw text
    cv2.putText(
        image_copy,
        text,
        position,
        font,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA
    )
    return image_copy
