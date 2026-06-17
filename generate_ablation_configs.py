import os

configs_dir = "src/configs/ablation"
os.makedirs(configs_dir, exist_ok=True)

def generate_spoter_yaml(run_id, data_dir="data/processed/vsl_400", aug_prob=0.3, add_gaussian_noise="true", interpolate="false", anchor="box", include_face="false", active_augs="[0, 1, 2, 3]"):
    is_face_included = str(include_face).lower() == "true"
    hidden_dim = 148 if is_face_included else 108
    content = f"""data:
  dataset: visl_400
  modality: pose
  subset: cam_1
  data_dir: {data_dir}
  transform:
    aug_prob: {aug_prob}
    add_gaussian_noise: {str(add_gaussian_noise).lower()}
    gaussian_noise_mean: 0.0
    gaussian_noise_std: 0.001
    interpolate: {str(interpolate).lower()}
    anchor: {anchor}
    include_face: {str(include_face).lower()}
    active_augs: {active_augs}
  fps: 25
  debug: false
model:
  arch: spoter
  num_frozen_layers: 0
  num_frames: 96
  hidden_dim: {hidden_dim}
training:
  output_dir: experiments/ablation/run_{run_id:02d}
  run_name: run_{run_id:02d}
  learning_rate: 0.0005
  weight_decay: 0.01
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  num_train_epochs: 100
  per_device_train_batch_size: 64
  per_device_eval_batch_size: 128
  gradient_accumulation_steps: 1
  dataloader_num_workers: 0
  dataloader_pin_memory: false
  dataloader_persistent_workers: false
  save_total_limit: 1
  save_safetensors: true
  fp16: false
  bf16: false
  use_cpu: false
  use_mps_device: true
  metric_for_best_model: accuracy
  report_to: null
"""
    return content

def generate_sl_gcn_yaml(run_id, interpolate="false", include_face="false"):
    content = f"""data:
  dataset: visl_400
  modality: pose
  subset: cam_1
  data_dir: data/processed/vsl_400
  transform:
    aug_prob: 0.5
    rotation_std: 0.2
    shear_std: 0.2
    scale_std: 0.2
    interpolate: {str(interpolate).lower()}
    anchor: box
    include_face: {str(include_face).lower()}
  fps: 25
  debug: false
model:
  arch: sl_gcn
  num_points: 27
  groups: 8
  block_size: 41
  in_channels: 3
  num_frozen_layers: 0
  num_frames: 150
training:
  output_dir: experiments/ablation/run_{run_id:02d}
  run_name: run_{run_id:02d}
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  num_train_epochs: 100
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 64
  gradient_accumulation_steps: 1
  dataloader_num_workers: 0
  dataloader_pin_memory: false
  dataloader_persistent_workers: false
  save_total_limit: 1
  save_safetensors: true
  fp16: false
  bf16: false
  use_cpu: false
  use_mps_device: true
  metric_for_best_model: accuracy
  report_to: null
"""
    return content

# Run definitions
runs = {}

# SPOTER runs
runs[0] = generate_spoter_yaml(0, aug_prob=0.0, add_gaussian_noise=False)
runs[1] = generate_spoter_yaml(1, data_dir="data/processed/vsl_400_tbl_theta140_taub400")
runs[2] = generate_spoter_yaml(2, data_dir="data/processed/vsl_400_tbl_theta150_taub400")
runs[3] = generate_spoter_yaml(3, data_dir="data/processed/vsl_400")
runs[4] = generate_spoter_yaml(4, data_dir="data/processed/vsl_400_tbl_theta170_taub400")
runs[5] = generate_spoter_yaml(5, data_dir="data/processed/vsl_400_tbl_theta160_taub200")
runs[6] = generate_spoter_yaml(6, data_dir="data/processed/vsl_400_tbl_theta160_taub600")
runs[7] = generate_spoter_yaml(7, interpolate=True)
runs[8] = generate_spoter_yaml(8, interpolate=True, anchor="neck")
runs[9] = generate_spoter_yaml(9, interpolate=True, anchor="nose")
runs[10] = generate_spoter_yaml(10, interpolate=True, aug_prob=0.3, add_gaussian_noise=False, active_augs="[0, 1]")
runs[11] = generate_spoter_yaml(11, interpolate=True, aug_prob=0.3, add_gaussian_noise=False, active_augs="[3]")
runs[12] = generate_spoter_yaml(12, interpolate=True, aug_prob=0.3, add_gaussian_noise=False, active_augs="[2]")
runs[13] = generate_spoter_yaml(13, interpolate=True, aug_prob=0.0, add_gaussian_noise=True)
runs[14] = generate_spoter_yaml(14, interpolate=True, aug_prob=0.3, add_gaussian_noise=True)
runs[15] = generate_spoter_yaml(15, interpolate=True, aug_prob=0.3, add_gaussian_noise=True, include_face=True)

# SL-GCN runs
runs[16] = generate_sl_gcn_yaml(16)
runs[17] = generate_sl_gcn_yaml(17, interpolate=True)
runs[18] = generate_sl_gcn_yaml(18, interpolate=True, include_face=True)

# Write out files
for run_id, yaml_content in runs.items():
    file_path = os.path.join(configs_dir, f"run_{run_id:02d}.yaml")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

print(f"Generated {len(runs)} configurations under {configs_dir}/")
