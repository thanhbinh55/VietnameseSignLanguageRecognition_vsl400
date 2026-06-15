import os
import json
from pathlib import Path

results_dir = Path("experiments/ablation_results")
report_path = Path("docs/ablation_study_report.md")

run_descriptions = {
    0: "Raw Baseline (No TBL, no interpolation, no augmentations)",
    1: "TBL Preprocessing (θ = 140°, τb = 400 ms)",
    2: "TBL Preprocessing (θ = 150°, τb = 400 ms)",
    3: "TBL Preprocessing (θ = 160°, τb = 400 ms - Default Base)",
    4: "TBL Preprocessing (θ = 170°, τb = 400 ms)",
    5: "TBL Preprocessing (θ = 160°, τb = 200 ms)",
    6: "TBL Preprocessing (θ = 160°, τb = 600 ms)",
    7: "Keypoint Interpolation (using best TBL θ = 160°, τb = 400 ms)",
    8: "Neck Anchor Normalization",
    9: "Nose Anchor Normalization",
    10: "Spatial Augmentations only (Rotate / Squeeze)",
    11: "Perspective Skew Augmentation only",
    12: "Kinematic Augmentation only (ArmJointRotate)",
    13: "Gaussian Noise Augmentation only",
    14: "Combined Augmentations (Spatial + Perspective + Kinematic + Noise)",
    15: "Facial Landmarks Integration (Eyebrows, Eyes, Mouth + Combined Augs)",
    16: "SL-GCN Baseline",
    17: "SL-GCN Optimized (Interpolation + Best TBL)",
    18: "SL-GCN Optimized + Face Landmarks"
}

def load_run_results():
    results = {}
    for run_id in range(19):
        file_path = results_dir / f"run_{run_id:02d}.json"
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    results[run_id] = data
            except Exception as e:
                results[run_id] = {"status": "error", "error": str(e)}
        else:
            results[run_id] = {"status": "missing"}
    return results

def make_markdown_row(run_id, info):
    desc = run_descriptions.get(run_id, "Unknown run")
    if info.get("status") == "missing":
        return f"| Run {run_id:02d} | {desc} | *N/A* | *N/A* | *N/A* | *N/A* | Missing |"
    elif info.get("status") == "failed" or info.get("status") == "error":
        return f"| Run {run_id:02d} | {desc} | *Err* | *Err* | *Err* | *Err* | Failed |"
    
    val_m = info.get("val_metrics", {})
    test_m = info.get("test_metrics", {})
    
    val_acc = val_m.get("val_accuracy", val_m.get("accuracy", -1))
    val_f1 = val_m.get("val_f1", val_m.get("f1", -1))
    test_acc = test_m.get("test_accuracy", test_m.get("accuracy", -1))
    test_f1 = test_m.get("test_f1", test_m.get("f1", -1))
    
    val_acc_str = f"{val_acc * 100:.2f}%" if val_acc >= 0 else "N/A"
    val_f1_str = f"{val_f1 * 100:.2f}%" if val_f1 >= 0 else "N/A"
    test_acc_str = f"{test_acc * 100:.2f}%" if test_acc >= 0 else "N/A"
    test_f1_str = f"{test_f1 * 100:.2f}%" if test_f1 >= 0 else "N/A"
    
    return f"| Run {run_id:02d} | {desc} | {val_acc_str} | {val_f1_str} | {test_acc_str} | {test_f1_str} | Completed |"

def main():
    results = load_run_results()
    
    os.makedirs(report_path.parent, exist_ok=True)
    
    markdown = []
    markdown.append("# VSL-400 Ablation Study and Research Report")
    markdown.append("\nThis report aggregates and analyzes the results of the 19 runs conducted for the Vietnamese Sign Language isolated word recognition ablation study.")
    
    # Category 1: TBL Preprocessing Sweep
    markdown.append("\n## Phase 1: TBL Preprocessing Sweep (Góc Ngưỡng & Trễ Đệm)")
    markdown.append("Optimizing the temporal boundary localization parameters for segmenting gesture boundaries.")
    markdown.append("\n| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |")
    markdown.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in range(1, 7):
        markdown.append(make_markdown_row(r, results[r]))
        
    # Category 2: Interpolation & Normalization
    markdown.append("\n## Phase 2: Keypoint Interpolation & Anchor Normalization")
    markdown.append("Comparing linear joint interpolation and centering strategies (Neck vs. Nose vs. Box).")
    markdown.append("\n| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |")
    markdown.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in [3, 7, 8, 9]:
        markdown.append(make_markdown_row(r, results[r]))
        
    # Category 3: Augmentations
    markdown.append("\n## Phase 3: Augmentation Ablation Study")
    markdown.append("Evaluating rotation, squeezing, perspective transforms, joint kinematics, and noise additions.")
    markdown.append("\n| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |")
    markdown.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in [7, 10, 11, 12, 13, 14, 15]:
        markdown.append(make_markdown_row(r, results[r]))
        
    # Category 4: Cross-Model on SL-GCN
    markdown.append("\n## Phase 4: Cross-Model Validation (SL-GCN)")
    markdown.append("Transferring the best preprocessing, interpolation, and face selections to the local SL-GCN model.")
    markdown.append("\n| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |")
    markdown.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in [16, 17, 18]:
        markdown.append(make_markdown_row(r, results[r]))

    # Raw baseline reference
    markdown.append("\n## Reference: Raw Baseline")
    markdown.append("\n| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |")
    markdown.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    markdown.append(make_markdown_row(0, results[0]))
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown))
        
    print(f"Aggregated ablation report successfully generated at {report_path}")

if __name__ == "__main__":
    main()
