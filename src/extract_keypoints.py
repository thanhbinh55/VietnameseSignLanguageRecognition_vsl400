import os
import sys
import logging
import subprocess
from tqdm import tqdm
from glob import glob
from pathlib import Path
from argparse import Namespace, ArgumentParser
from utils import config_logger, VIDEO_EXTENSIONS
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_args() -> Namespace:
    """Parse command line arguments for keypoint extraction."""
    parser = ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=str,
        help="Directory containing videos to extract keypoints from",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing keypoints",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the order of the keypoints",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing",
    )
    return parser.parse_args()


def process_video(video_path: str, overwrite: bool) -> str:
    """Run the video_to_pose executable to extract Mediapipe keypoints from a video."""
    video = Path(video_path)
    pose = video.with_suffix(".pose")

    if not overwrite and pose.exists():
        return "skipped"

    # Resolve local video_to_pose executable in virtual environment
    local_video_to_pose = Path(sys.executable).parent / "video_to_pose"
    video_to_pose_cmd = str(local_video_to_pose) if local_video_to_pose.exists() else "video_to_pose"

    # Prevent CPU oversubscription by limiting internal libraries to 1 thread per process
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    try:
        subprocess.run(
            [
                video_to_pose_cmd,
                "--format",
                "mediapipe",
                "-i",
                str(video),
                "-o",
                str(pose),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            check=True,
        )
        return "processed"
    except Exception as e:
        return f"failed: {str(e)}"


def main(args: Namespace) -> None:
    """Main entrypoint for extracting keypoints from all videos in a directory."""
    logging.info(f"Extracting keypoints from videos in {args.video_dir}")
    logging.info(f"Overwrite existing keypoints: {args.overwrite}")
    logging.info(f"Number of workers: {args.num_workers}")

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob(f"{args.video_dir}/*{ext}"))
    videos = sorted(videos, reverse=args.reverse)
    num_videos = len(videos)
    logging.info(f"Found {num_videos} videos")

    failures = []
    skipped_count = 0
    processed_count = 0

    if args.num_workers <= 1:
        # Sequential execution
        for video in tqdm(videos):
            res = process_video(video, args.overwrite)
            if "failed" in res:
                failures.append((video, res))
            elif res == "skipped":
                skipped_count += 1
            else:
                processed_count += 1
    else:
        # Parallel execution using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_video, video, args.overwrite): video
                for video in videos
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                video = futures[future]
                try:
                    res = future.result()
                    if "failed" in res:
                        failures.append((video, res))
                    elif res == "skipped":
                        skipped_count += 1
                    else:
                        processed_count += 1
                except Exception as e:
                    failures.append((video, f"raised exception: {str(e)}"))

    logging.info(f"Finished extracting keypoints. Processed: {processed_count}, Skipped: {skipped_count}, Failed: {len(failures)}")
    if failures:
        logging.error(f"Failed to process {len(failures)} videos:")
        for video, err in failures[:10]:
            logging.error(f"  - {Path(video).name}: {err}")
        if len(failures) > 10:
            logging.error(f"  - ... and {len(failures) - 10} more failures.")


if __name__ == "__main__":
    args = get_args()
    config_logger()
    main(args=args)
