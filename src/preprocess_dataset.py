import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from simple_parsing import ArgumentParser
from configs import DataConfig, ModelConfig
from tools import load_model, load_dataset


def get_args() -> Namespace:
    """Parse command line arguments for dataset preprocessing."""
    parser = ArgumentParser(description="Preprocess and export VSL-400 keypoint dataset")
    parser.add_arguments(DataConfig, "data")
    parser.add_arguments(ModelConfig, "model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocessed_vsl_400",
        help="Directory to save preprocessed keypoints",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    """Main entrypoint for preprocessing the VSL-400 dataset."""
    data_config = args.data
    model_config = args.model
    output_root = Path(args.output_dir)

    print("Loading dataset metadata...")
    dataset = load_dataset(data_config)

    # For preprocessing we only need the processor (transform config), not the full model.
    # Directly instantiate the processor from ModelConfig to avoid needing pretrained
    # weights or a label2id mapping (which is only required for the classifier head).
    from models import (
        SPOTERConfig, SPOTERFeatureExtractor,
        SLGCNConfig, SLGCNFeatureExtractor,
    )
    if model_config.arch == "spoter":
        proc_config = SPOTERConfig(**vars(model_config))
        processor = SPOTERFeatureExtractor(config=proc_config)
    elif model_config.arch == "sl_gcn":
        proc_config = SLGCNConfig(**vars(model_config))
        processor = SLGCNFeatureExtractor(config=proc_config)
    else:
        raise ValueError(f"Unsupported arch for preprocessing: {model_config.arch}")

    # Get the clean pose transform pipeline (no training augmentations like shear/noise)
    # This uses split="test" to ensure we only get: Interpolate -> Extract -> Select -> Normalize -> Tensor
    transforms = dataset.get_split("test", processor).transforms

    # Determine camera/views structure from subset if available
    cams = []
    if data_config.subset and "_" in data_config.subset:
        cams = data_config.subset.split("_")[1:]
    
    # We use subdirectories only if:
    # 1. cams is not empty.
    # 2. output_root's name is NOT already a camera/view name (like cam_side, side, cam_front, front),
    #    which indicates the user specified a camera-specific directory directly.
    use_subdirs = False
    if cams:
        use_subdirs = True
        for cam in cams:
            if output_root.name == f"cam_{cam}" or output_root.name == cam:
                use_subdirs = False
                break

    if use_subdirs:
        for cam in cams:
            (output_root / f"cam_{cam}").mkdir(parents=True, exist_ok=True)

    # Process each split
    all_processed_records = []
    for split in ["train", "validation", "test"]:
        print(f"Preprocessing {split} split...")
        samples = dataset.dataset[split]

        for sample in tqdm(samples):
            video_id = sample["video_id"]
            pose_path = sample["pose"]
            
            # Run the preprocessing pipeline
            try:
                # Apply the transform sequence
                preprocessed_tensor = transforms(pose_path)
                
                # Convert PyTorch Tensor to NumPy array
                preprocessed_np = preprocessed_tensor.cpu().numpy()
                
                # Determine subdirectory
                cam_dir = ""
                if use_subdirs:
                    for cam in cams:
                        if f"cam_{cam}" in pose_path or f"cam_{cam}" in video_id:
                            cam_dir = f"cam_{cam}"
                            break
                
                # Save as .npy
                out_file_name = f"{video_id}_preprocessed.npy"
                out_path = output_root / cam_dir / out_file_name
                np.save(out_path, preprocessed_np)
                
                # Record metadata mapping
                record = sample.copy()
                record["pose"] = str(out_path)
                all_processed_records.append(record)
                
            except Exception as e:
                print(f"Error processing {video_id}: {e}")

    # Save preprocessed metadata JSON
    if use_subdirs:
        for cam in cams:
            cam_records = [r for r in all_processed_records if f"cam_{cam}" in r["pose"] or f"cam_{cam}" in r["video_id"]]
            if cam_records:
                meta_out_path = output_root / f"cam_{cam}.json"
                with open(meta_out_path, "w", encoding="utf-8") as f:
                    json.dump(cam_records, f, indent=4, ensure_ascii=False)
                print(f"Saved preprocessed metadata to {meta_out_path}")
    else:
        if cams:
            meta_name = f"cam_{cams[0]}.json"
        elif data_config.subset:
            meta_name = f"{data_config.subset}.json"
        else:
            meta_name = "metadata.json"
            
        meta_out_path = output_root / meta_name
        with open(meta_out_path, "w", encoding="utf-8") as f:
            json.dump(all_processed_records, f, indent=4, ensure_ascii=False)
        print(f"Saved preprocessed metadata to {meta_out_path}")

    # Copy gloss.csv if exists
    gloss_csv = Path(data_config.data_dir) / "gloss.csv"
    if gloss_csv.exists():
        import shutil
        shutil.copy(gloss_csv, output_root / "gloss.csv")

    print(f"\nPreprocessing completed! Preprocessed dataset saved at: {output_root.resolve()}")


if __name__ == "__main__":
    main(get_args())
