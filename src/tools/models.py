import logging
import torch
import onnxruntime as ort
from time import time
from pathlib import Path
from typing import Union
from configs import ModelConfig, InferenceConfig, EvaluationConfig
from utils import (
    POSE_BASED_MODELS,
)
from transformers import (
    FeatureExtractionMixin,
    AutoModel,
    Pipeline,
    pipeline,
)
from transformers.pipelines import PIPELINE_REGISTRY
from visualization import draw_text_on_image
from models import (
    SPOTERConfig, SPOTERFeatureExtractor, SPOTERForGraphClassification,
    SLGCNConfig, SLGCNFeatureExtractor, SLGCNForGraphClassification,
)


def _is_local_checkpoint(path: str) -> bool:
    """Return True if path points to an existing local directory."""
    if path is None:
        return False
    return Path(str(path)).exists()


def load_model(
    model_config: Union[ModelConfig, EvaluationConfig],
    label2id: dict = None,
    id2label: dict = None,
    do_train: bool = False,
) -> tuple:
    """
    Load a model from a local checkpoint or initialise from scratch.

    Parameters
    ----------
    model_config : ModelConfig | EvaluationConfig
    label2id : dict, optional
    id2label : dict, optional
    do_train : bool
        If True, prepare the model for training (gradient updates enabled).

    Returns
    -------
    tuple
        (config, processor, model)
    """
    if isinstance(model_config, EvaluationConfig):
        do_train = False

    if do_train:
        return load_pose_model_for_training(model_config, label2id, id2label)

    # Inference / evaluation: load from local checkpoint
    if not _is_local_checkpoint(model_config.pretrained):
        logging.error(
            f"Checkpoint not found at '{model_config.pretrained}'. "
            "Train the model first and point 'pretrained' to the local experiments/ directory."
        )
        exit(1)

    if model_config.arch in POSE_BASED_MODELS:
        if model_config.arch == "spoter":
            config_class = SPOTERConfig
            processor_class = SPOTERFeatureExtractor
            model_class = SPOTERForGraphClassification
        elif model_config.arch == "sl_gcn":
            config_class = SLGCNConfig
            processor_class = SLGCNFeatureExtractor
            model_class = SLGCNForGraphClassification
        else:
            config_class = None

        if config_class is not None:
            config = config_class.from_pretrained(model_config.pretrained)
            processor = processor_class.from_pretrained(model_config.pretrained)
            model = model_class.from_pretrained(model_config.pretrained)
        else:
            processor = FeatureExtractionMixin.from_pretrained(
                model_config.pretrained,
                trust_remote_code=True,
            )
            model = AutoModel.from_pretrained(
                model_config.pretrained,
                trust_remote_code=True,
            )
    else:
        logging.error(f"Model {model_config.arch} is not supported")
        exit(1)
    model.eval()
    return model.config if hasattr(model, 'config') else config, processor, model


def load_pose_model_for_training(
    model_config: ModelConfig,
    label2id: dict = None,
    id2label: dict = None,
) -> tuple:
    """
    Load or initialise a pose-based model for training.
    If 'pretrained' is a local checkpoint directory, resume from it.
    Otherwise initialise from scratch.
    """
    if _is_local_checkpoint(model_config.pretrained):
        # Resume fine-tuning from a local checkpoint
        if model_config.arch == "spoter":
            config_class = SPOTERConfig
            processor_class = SPOTERFeatureExtractor
            model_class = SPOTERForGraphClassification
        elif model_config.arch == "sl_gcn":
            config_class = SLGCNConfig
            processor_class = SLGCNFeatureExtractor
            model_class = SLGCNForGraphClassification
        else:
            config_class = None

        if config_class is not None:
            config = config_class.from_pretrained(model_config.pretrained)
            processor = processor_class.from_pretrained(model_config.pretrained)
            model = model_class.from_pretrained(
                model_config.pretrained,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            )
            return config, processor, model
        else:
            processor = FeatureExtractionMixin.from_pretrained(
                model_config.pretrained,
                trust_remote_code=True,
            )
            model = AutoModel.from_pretrained(
                model_config.pretrained,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
            )
            return model.config, processor, model

    if model_config.arch in POSE_BASED_MODELS:
        if model_config.arch == "spoter":
            config_class = SPOTERConfig
            processor_class = SPOTERFeatureExtractor
            model_class = SPOTERForGraphClassification
        elif model_config.arch == "sl_gcn":
            config_class = SLGCNConfig
            processor_class = SLGCNFeatureExtractor
            model_class = SLGCNForGraphClassification
        elif model_config.arch == "dsta_slr":
            config_class = DSTASLRConfig
            processor_class = DSTASLRFeatureExtractor
            model_class = DSTASLRForGraphClassification
        else:
            logging.error(f"Model {model_config.arch} is not supported")
            exit(1)
    else:
        logging.error(f"Model {model_config.arch} is not supported")
        exit(1)

    config_class.register_for_auto_class()
    processor_class.register_for_auto_class("AutoFeatureExtractor")
    model_class.register_for_auto_class("AutoModel")
    logging.info(f"Registering {model_config.arch} classes")

    config = config_class(**vars(model_config))
    processor = processor_class(config=config)
    model = model_class(config=config, label2id=label2id, id2label=id2label)

    return config, processor, model


class Predictions:
    def __init__(
        self,
        predictions: list[dict] = None,
        inference_time: float = 0,
        start_time: float = 0,
        end_time: float = 0,
    ) -> None:
        self.predictions = predictions
        self.inference_time = inference_time
        self.start_time = start_time
        self.end_time = end_time

    def visualize(
        self,
        frame: torch.Tensor,
        position: tuple = (20, 100),
        prefix: str = "Predictions",
        color: tuple = (0, 0, 255),
    ) -> None:
        text = prefix + ": " + self.get_pred_message()
        return draw_text_on_image(
            image=frame,
            text=text,
            position=position,
            color=color,
            font_size=20,
        )

    def get_pred_message(self) -> str:
        if not any((
            self.start_time,
            self.end_time,
            self.inference_time,
            self.predictions
        )):
            return ""

        return ', '.join(
            [
                f"{pred['gloss']} ({pred['score']*100:.2f}%)"
                for pred in self.predictions
            ]
        )

    def __str__(self) -> str:
        if not any((
            self.start_time,
            self.end_time,
            self.inference_time,
            self.predictions
        )):
            return ""

        predictions = self.get_pred_message()
        message = "Sample start: {:.2f}s - end: {:.2f}s | Runtime: {:.2f}s | Predictions: {}"
        return message.format(self.start_time, self.end_time, self.inference_time, predictions)

    def merge_results(self, results: dict = None) -> dict:
        if results is None:
            results = {
                "start_time": [],
                "end_time": [],
                "inference_time": [],
                "prediction": [],
            }
        results["start_time"].append(self.start_time)
        results["end_time"].append(self.end_time)
        results["inference_time"].append(self.inference_time)
        results["prediction"].append(self.predictions)
        return results


def get_predictions(
    inputs: torch.Tensor,
    model: Union[ort.InferenceSession, AutoModel],
    id2gloss: dict,
    k: int = 3,
) -> Predictions:
    """
    Get the top-k predictions.

    Parameters
    ----------
    inputs : torch.Tensor
        Model inputs (Time, Height, Width, Channels).
    model : Union[ort.InferenceSession, AutoModel]
        Model to get predictions from.
    id2gloss : dict
        Mapping of class indices to glosses.
    k : int, optional
        Number of predictions to return, by default 3.

    Returns
    -------
    Predictions
    """
    if inputs is None:
        return Predictions()

    start_time = time()
    if isinstance(model, ort.InferenceSession):
        inputs = inputs.cpu().numpy()
        logits = torch.from_numpy(model.run(None, {"pixel_values": inputs})[0])
    else:
        logits = model(inputs.to(model.device)).logits
    inference_time = time() - start_time

    topk_scores, topk_indices = torch.topk(logits, k, dim=1)
    topk_scores = torch.nn.functional.softmax(topk_scores, dim=1).squeeze().detach().numpy()
    topk_indices = topk_indices.squeeze().detach().numpy()
    predictions = [
        {
            'gloss': id2gloss[str(topk_indices[i])],
            'score': topk_scores[i],
        }
        for i in range(k)
    ]

    return Predictions(predictions=predictions, inference_time=inference_time)


def load_pipeline(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
) -> Pipeline:
    """
    Load an inference pipeline from a local checkpoint.

    Parameters
    ----------
    model_config : ModelConfig
    inference_config : InferenceConfig

    Returns
    -------
    Pipeline
    """
    from pipelines import (
        SLGCNGraphClassificationPipeline,
        SPOTERGraphClassificationPipeline,
    )
    if model_config.arch in POSE_BASED_MODELS:
        _, processor, model = load_model(model_config)

        if model_config.arch == "spoter":
            return SPOTERGraphClassificationPipeline(
                model=model,
                feature_extractor=processor,
                device=inference_config.device,
            )

        if model_config.arch in ["sl_gcn", "dsta_slr"]:
            return SLGCNGraphClassificationPipeline(
                model=model,
                feature_extractor=processor,
                device=inference_config.device,
            )

    logging.error(f"Pipeline loading not supported for model {model_config.arch}")
    exit(1)


def get_input_shape(
    arch: str,
    processor: FeatureExtractionMixin,
    batch_size: int = 1,
) -> tuple:
    """
    Get the input shape for a given model architecture.

    Parameters
    ----------
    arch : str
    processor : FeatureExtractionMixin
    batch_size : int, optional

    Returns
    -------
    tuple
    """
    if arch in POSE_BASED_MODELS:
        if arch == "spoter":
            return (
                batch_size,
                processor.num_frames,
                processor.num_points,
                processor.in_channels,
            )
        elif arch in ["sl_gcn", "dsta_slr"]:
            return (
                batch_size,
                processor.in_channels,
                processor.window_size,
                processor.num_points,
                processor.num_people,
            )
        else:
            logging.error(f"Model {arch} is not supported")
            exit(1)
    else:
        logging.error(f"Model {arch} is not supported")
        exit(1)
