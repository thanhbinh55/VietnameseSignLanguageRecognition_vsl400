import os
import shutil
import logging
import torch
from argparse import Namespace
from simple_parsing import ArgumentParser
from transformers import TrainingArguments, Trainer
from configs import DataConfig, ModelConfig, TrainingConfig
from utils import (
    compute_metrics,
    config_logger,
    TrainingCallback,
    save_evaluation_results,
    compute_flops_and_params,
)
from tools import (
    load_model,
    load_dataset,
    pose_collate_fn
)


def train_with_checkpoint_compat(trainer: Trainer, resume_from_checkpoint: str = None):
    if resume_from_checkpoint is None:
        return trainer.train()

    original_torch_load = torch.load

    def torch_load_with_pickle_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_with_pickle_compat
    try:
        return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        torch.load = original_torch_load


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Train a SLR model",
        add_config_path_arg=True,
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="VieSignLang",
        help="Weights and Biases entity",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="sign-language-recognition",
        help="Weights and Biases project",
    )
    parser.add_arguments(DataConfig, "data")
    parser.add_arguments(ModelConfig, "model")
    parser.add_arguments(TrainingConfig, "training")
    return parser.parse_args()


def main(args: Namespace) -> None:
    data_config = args.data
    logging.info(data_config)
    model_config = args.model
    logging.info(model_config)
    training_config = args.training
    logging.info(training_config)

    dataset = load_dataset(data_config)
    logging.info(f"{data_config.dataset.upper()} dataset loaded")

    config, processor, model = load_model(
        model_config,
        label2id=dataset.gloss2id,
        id2label=dataset.id2gloss,
        do_train=True,
    )
    logging.info(f"{model_config.arch} model loaded from {model_config.pretrained}")

    train_dataset = dataset.get_split("train", processor)
    val_dataset = dataset.get_split("validation", processor)
    test_dataset = dataset.get_split("test", processor)
    logging.info("Splits created for training and evaluation")
    logging.info(f"Number of samples in training set: {train_dataset.num_videos}")
    logging.info(f"Number of samples in validation set: {val_dataset.num_videos}")
    logging.info(f"Number of samples in test set: {test_dataset.num_videos}")

    data_collator = pose_collate_fn

    callbacks = [TrainingCallback()]
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**vars(training_config)),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks,
        tokenizer=processor,
    )
    logging.info("Trainer created")

    try:
        flops, params = compute_flops_and_params(model, next(iter(train_dataset)))
        logging.info(f"FLOPs: {flops:,}")
        logging.info(f"Number of parameters: {params:,}")
    except Exception as exc:
        logging.warning(f"Skipping FLOPs/profile calculation: {exc}")

    logging.info("Training started")
    train_with_checkpoint_compat(
        trainer,
        resume_from_checkpoint=training_config.resume_from_checkpoint,
    )
    logging.info("Training completed")

    trainer.save_model(training_config.output_dir)
    logging.info(f"Model saved to {training_config.output_dir}")

    logging.info("Evaluation started")

    val_output_dir = training_config.output_dir / "validation" / data_config.dataset
    val_results = trainer.predict(val_dataset, metric_key_prefix="val")
    logging.info(f"Validation results: {val_results.metrics}")
    save_evaluation_results(
        results=val_results,
        classes=dataset.gloss2id.keys(),
        output_dir=val_output_dir,
    )
    logging.info(f"Validation results saved to {val_output_dir}")

    test_output_dir = training_config.output_dir / "test" / data_config.dataset
    test_results = trainer.predict(test_dataset, metric_key_prefix="test")
    logging.info(f"Test results: {test_results.metrics}")
    save_evaluation_results(
        results=test_results,
        classes=dataset.gloss2id.keys(),
        output_dir=test_output_dir,
    )
    logging.info(f"Test results saved to {test_output_dir}")

    logging.info("Evaluation completed")


if __name__ == "__main__":
    args = get_args()
    config_logger(log_file=args.training.output_dir / "train.log")
    logging.info(f"Config file loaded from {args.config_path[0]}")

    shutil.copy(args.config_path[0], args.training.output_dir / "train.yaml")
    logging.info(f"Config file saved to {args.training.output_dir}")

    os.environ["WANDB_MODE"] = "disabled"

    main(args=args)
