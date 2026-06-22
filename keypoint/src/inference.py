import cv2
import shutil
import logging
import numpy as np
import pandas as pd
from time import time
from traceback import format_exc
from argparse import Namespace
from transformers import Pipeline
from simple_parsing import ArgumentParser
from visualization import draw_text_on_image
from configs import ModelConfig, InferenceConfig
from utils import config_logger, POSE_BASED_MODELS
from data import Arm, get_sample_timestamp, ok_to_get_frame
from mediapipe.python.solutions import pose, drawing_utils, holistic
from tools import load_pipeline, Predictions


def get_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description="Train a model on VSL",
        add_config_path_arg=True,
    )
    parser.add_arguments(ModelConfig, "model")
    parser.add_arguments(InferenceConfig, "inference")
    return parser.parse_args()


def process_frame(frame: np.ndarray, keypoints_detector) -> tuple[np.ndarray, any]:
    """Process a single frame to extract Mediapipe pose landmarks."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    detection_results = keypoints_detector.process(rgb_frame)
    rgb_frame.flags.writeable = True
    return rgb_frame, detection_results


def run_prediction(
    data: list, 
    pipeline: Pipeline, 
    config: InferenceConfig, 
    source_fps: float, 
    frame_shape: tuple
) -> Predictions:
    """Run model inference on the accumulated frames/pose data."""
    start_inference_time = time()
    if config.use_pose_model:
        sample = {
            "frames": data,
            "fps": source_fps,
            "width": frame_shape[1],
            "height": frame_shape[0],
        }
    else:
        sample = np.array(data)
        
    predictions = Predictions(predictions=pipeline(sample, top_k=config.top_k))
    predictions.inference_time = time() - start_inference_time
    return predictions


def render_visuals(
    frame: np.ndarray,
    detection_results,
    left_arm: Arm,
    right_arm: Arm,
    predictions: Predictions,
    config: InferenceConfig
) -> np.ndarray:
    """Render arms angle, prediction results, and skeleton on the frame."""
    frame = left_arm.visualize(frame, (20, 10), "Left arm angle")
    frame = right_arm.visualize(frame, (20, 40), "Right arm angle")
    frame = predictions.visualize(frame, (20, 70))
    
    if config.show_skeleton and detection_results.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            detection_results.pose_landmarks,
            pose.POSE_CONNECTIONS
        )
    return frame


def inference(config: InferenceConfig, pipeline: Pipeline) -> None:
    """Main inference loop for processing video/webcam feed."""
    source = str(config.source) if config.source.is_file() else 0
    cap = cv2.VideoCapture(source)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    writer = None
    if config.output_dir is not None:
        writer = cv2.VideoWriter(
            str(config.output_dir / "output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (int(cap.get(3)), int(cap.get(4))),
        )

    # Init Mediapipe holistic model for body and hands keypoints
    keypoints_detector = holistic.Holistic(
        model_complexity=0,
        min_detection_confidence=0.9,
    )

    right_arm = Arm("right", config.visibility)
    left_arm = Arm("left", config.visibility)
    data = []
    results = None
    predictions = Predictions()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame, detection_results = process_frame(frame, keypoints_detector)

        # Update arm poses if landmarks are found
        if detection_results.pose_landmarks:
            landmarks = detection_results.pose_landmarks.landmark
            left_arm.set_pose(landmarks)
            right_arm.set_pose(landmarks)
        else:
            continue

        # Check if arms are raised indicating a sign is being performed
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        left_ok = ok_to_get_frame(
            arm=left_arm, angle_threshold=config.angle_threshold,
            min_num_up_frames=config.min_num_up_frames, min_num_down_frames=config.min_num_down_frames,
            current_time=current_time_ms, delay=config.delay,
        )
        right_ok = ok_to_get_frame(
            arm=right_arm, angle_threshold=config.angle_threshold,
            min_num_up_frames=config.min_num_up_frames, min_num_down_frames=config.min_num_down_frames,
            current_time=current_time_ms, delay=config.delay,
        )

        # Accumulate frames while arms are up
        if left_ok or right_ok:
            predictions = Predictions() # Clear old predictions
            data.append(rgb_frame.copy() if config.use_pose_model else frame.copy())

        # Determine start and end time of the sign based on arm movements
        start_time, end_time = get_sample_timestamp(left_arm, right_arm)

        if start_time != 0 and end_time != 0:
            if config.visualize:
                wait_frame = draw_text_on_image(
                    np.zeros_like(frame), text="Please wait for the prediction...",
                    position=(20, 20), color=(255, 255, 255), font_size=20,
                )
                cv2.imshow("Video Visualization", wait_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Execute pipeline
            predictions = run_prediction(data, pipeline, config, source_fps, frame.shape)
            predictions.start_time = start_time
            predictions.end_time = end_time
            
            logging.info(str(predictions))
            results = predictions.merge_results(results)

            # Reset variables for the next sign
            start_time, end_time = 0, 0
            left_arm.reset_state()
            right_arm.reset_state()
            data = []

        frame = render_visuals(frame, detection_results, left_arm, right_arm, predictions, config)

        if writer is not None:
            writer.write(frame)

        if config.visualize:
            cv2.imshow("Video Visualization", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if writer is not None:
        writer.release()
        logging.info(f"Video is recorded and saved to {config.output_dir / 'output.mp4'}")
        pd.DataFrame(results).to_csv(config.output_dir / "results.csv", index=False)
        logging.info(f"Results saved to {config.output_dir / 'results.csv'}")


def main(args: Namespace) -> None:
    """Load model pipeline and start inference."""
    model_config = args.model
    inference_config = args.inference
    logging.info(model_config)
    logging.info(inference_config)

    inference_config.use_pose_model = model_config.arch in POSE_BASED_MODELS

    pipeline = load_pipeline(model_config, inference_config)
    logging.info("Pipeline loaded")

    inference(inference_config, pipeline)
    logging.info("Inference completed")


if __name__ == "__main__":
    try:
        args = get_args()
        config_logger(args.inference.output_dir / "inference.log")
        logging.info(f"Config file loaded from {args.config_path[0]}")
        shutil.copy(args.config_path[0], args.inference.output_dir / "inference.yaml")
        logging.info(f"Config file saved to {args.inference.output_dir}")
        main(args=args)
    except Exception:
        print(format_exc())
