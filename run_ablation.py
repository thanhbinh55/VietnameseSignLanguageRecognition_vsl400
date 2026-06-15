import os
import sys
import json
import argparse
from pathlib import Path

# Add src to python path so we can import modules correctly
sys.path.insert(0, os.path.abspath("src"))

def main():
    parser = argparse.ArgumentParser(description="Run a specific VSL ablation study trial")
    parser.add_argument("--run_id", type=int, required=True, help="Run ID from 0 to 18")
    parser.add_argument("--dry_run", action="store_true", help="Perform a dry-run check (verifies model & dataset loading)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs (e.g., set to 1 for quick tests)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory path (e.g., /kaggle/input/data-vsl400-front-view)")
    
    args = parser.parse_args()
    
    config_path = f"src/configs/ablation/run_{args.run_id:02d}.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}. Did you run generate_ablation_configs.py first?")
        sys.exit(1)
        
    print(f"==================================================")
    print(f"Starting VSL Ablation Study - Run {args.run_id:02d}")
    print(f"Config path: {config_path}")
    print(f"==================================================")
    
    # Pre-configure sys.argv so simple_parsing inside train.py reads the correct config file
    original_argv = sys.argv
    sys.argv = [sys.argv[0], "--config_path", config_path]
    
    # Overwrite environment variables for offline mode
    os.environ["WANDB_MODE"] = "disabled"
    
    try:
        import train
        from configs import DataConfig, ModelConfig, TrainingConfig
        from tools import load_dataset, load_model
        
        # Parse arguments using simple_parsing to get namespace
        train_args = train.get_args()
        
        # Override parameters if requested
        if args.epochs is not None:
            train_args.training.num_train_epochs = args.epochs
        if args.per_device_train_batch_size is not None:
            train_args.training.per_device_train_batch_size = args.per_device_train_batch_size
        if args.per_device_eval_batch_size is not None:
            train_args.training.per_device_eval_batch_size = args.per_device_eval_batch_size
        if args.data_dir is not None:
            train_args.data.data_dir = args.data_dir
            
        if args.dry_run:
            print("Running dry-run verification...")
            # Load dataset
            dataset = load_dataset(train_args.data)
            print(f"Dataset successfully loaded. Labels: {len(dataset.gloss2id)}")
            
            # Load model
            config, processor, model = load_model(
                train_args.model,
                label2id=dataset.gloss2id,
                id2label=dataset.id2gloss,
                do_train=True
            )
            print(f"Model ({train_args.model.arch}) successfully loaded and compiled.")
            
            # Create splits
            train_dataset = dataset.get_split("train", processor)
            print(f"Data transforms tested successfully. Train samples: {train_dataset.num_videos}")
            print("Dry-run check: PASSED!")
            sys.exit(0)
            
        # Run standard training main
        train.main(train_args)
        
        # Collect results
        output_dir = Path(train_args.training.output_dir)
        val_results_path = output_dir / "validation" / train_args.data.dataset / "results.json"
        test_results_path = output_dir / "test" / train_args.data.dataset / "results.json"
        
        summary = {
            "run_id": args.run_id,
            "arch": train_args.model.arch,
            "epochs": train_args.training.num_train_epochs,
            "val_metrics": {},
            "test_metrics": {},
            "status": "success"
        }
        
        if val_results_path.exists():
            with open(val_results_path, "r") as f:
                val_data = json.load(f)
                summary["val_metrics"] = val_data.get("metrics", {})
                
        if test_results_path.exists():
            with open(test_results_path, "r") as f:
                test_data = json.load(f)
                summary["test_metrics"] = test_data.get("metrics", {})
                
        # Write ablation study summary JSON file
        summary_dir = Path("experiments/ablation_results")
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"run_{args.run_id:02d}.json"
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        print(f"Ablation results for Run {args.run_id:02d} written successfully to {summary_path}")
        
    except Exception as e:
        print(f"Error during Run {args.run_id:02d} execution:")
        import traceback
        traceback.print_exc()
        
        # Save failure info
        summary_dir = Path("experiments/ablation_results")
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / f"run_{args.run_id:02d}.json", "w") as f:
            json.dump({
                "run_id": args.run_id,
                "status": "failed",
                "error": str(e)
            }, f, indent=4)
        sys.exit(1)

if __name__ == "__main__":
    main()
