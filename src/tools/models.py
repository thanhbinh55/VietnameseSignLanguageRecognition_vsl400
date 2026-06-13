import logging
import torch
import onnxruntime as ort
from time import time
from pathlib import Path
from typing import Union
from configs import ModelConfig, InferenceConfig, EvaluationConfig
from utils import (
    POSE_BASED_MODELS,
    RGB_BASED_MODELS,
    HUGGINGFACE_RGB_BASED_MODELS,
    TORCHHUB_RGB_BASED_MODELS,
)
from transformers import (
    ImageProcessingMixin,
    FeatureExtractionMixin,
    AutoModelForVideoClassification,
    AutoModel,
    Pipeline,
    pipeline,
)
from transformers.pipelines import PIPELINE_REGISTRY
from visualization import draw_text_on_image
from models import (
    SPOTERConfig, SPOTERFeatureExtractor, SPOTERForGraphClassification,
)
from pipelines import (
    SLGCNGraphClassificationPipeline,
    SPOTERGraphClassificationPipeline,
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
        if model_config.arch in POSE_BASED_MODELS:
            return load_pose_model_for_training(model_config, label2id, id2label)
        return load_rgb_model_for_training(model_config, label2id, id2label)

    # Inference / evaluation: load from local checkpoint
    if not _is_local_checkpoint(model_config.pretrained):
        logging.error(
            f"Checkpoint not found at '{model_config.pretrained}'. "
            "Train the model first and point 'pretrained' to the local experiments/ directory."
        )
        exit(1)

    if model_config.arch in POSE_BASED_MODELS:
        processor = FeatureExtractionMixin.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
        )
    else:
        processor = ImageProcessingMixin.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
        )
        model = AutoModelForVideoClassification.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
        )
    model.eval()
    return model.config, processor, model


def load_rgb_model_for_training(
    model_config: ModelConfig,
    label2id: dict = None,
    id2label: dict = None,
) -> tuple:
    """
    Load or initialise an RGB-based model for training.
    If 'pretrained' is a local directory with a saved checkpoint, resume from it.
    Otherwise initialise weights from scratch (or from ImageNet pretrained weights).
    """
    if _is_local_checkpoint(model_config.pretrained):
        # Resume fine-tuning from a local checkpoint
        processor = ImageProcessingMixin.from_pretrained(
            model_config.pretrained,
            trust_remote_code=True,
        )
        model = AutoModelForVideoClassification.from_pretrained(
            model_config.pretrained,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
        return model.config, processor, model

    if model_config.arch in HUGGINGFACE_RGB_BASED_MODELS:
        if model_config.arch == "videomae":
            from models.videomae import (
                VideoMAEConfig,
                VideoMAEImageProcessor,
                VideoMAEForVideoClassification,
            )
            config_class = VideoMAEConfig
            processor_class = VideoMAEImageProcessor
            model_class = VideoMAEForVideoClassification
        else:
            logging.error(f"Model {model_config.arch} is not supported")
            exit(1)
    elif model_config.arch in TORCHHUB_RGB_BASED_MODELS:
        if model_config.arch in ['swin3d_t', 'swin3d_s', 'swin3d_b']:
            config_class = Swin3DConfig
            processor_class = Swin3DImageProcessor
            model_class = Swin3DForVideoClassification
        elif model_config.arch in ['r3d_18', 'mc3_18', 'r2plus1d_18']:
            config_class = VideoResNetConfig
            processor_class = VideoResNetImageProcessor
            model_class = VideoResNetForVideoClassification
        elif model_config.arch in ['s3d']:
            config_class = S3DConfig
            processor_class = S3DImageProcessor
            model_class = S3DForVideoClassification
        elif model_config.arch in ['mvit_v1_b', 'mvit_v2_s']:
            config_class = MViTConfig
            processor_class = MViTImageProcessor
            model_class = MViTForVideoClassification
        else:
            logging.error(f"Model {model_config.arch} is not supported")
            exit(1)
    else:
        logging.error(f"Model {model_config.arch} is not supported")
        exit(1)

    config_class.register_for_auto_class()
    processor_class.register_for_auto_class("AutoImageProcessor")
    model_class.register_for_auto_class("AutoModel")
    model_class.register_for_auto_class("AutoModelForVideoClassification")
    logging.info(f"{model_config.arch} classes registered")

    config = config_class(**vars(model_config))
    processor = processor_class(config=config)
    model = model_class(config=config, label2id=label2id, id2label=id2label)

    return config, processor, model


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

    if not _is_local_checkpoint(model_config.pretrained):
        logging.error(
            f"Checkpoint not found at '{model_config.pretrained}'. "
            "Train the model first."
        )
        exit(1)

    return pipeline(
        "video-classification",
        model=model_config.pretrained,
        image_processor=model_config.pretrained,
        device=inference_config.device,
        trust_remote_code=True,
        use_onnx=inference_config.use_onnx,
        top_k=inference_config.top_k,
    )


def get_input_shape(
    arch: str,
    processor: Union[ImageProcessingMixin, FeatureExtractionMixin],
    batch_size: int = 1,
) -> tuple:
    """
    Get the input shape for a given model architecture.

    Parameters
    ----------
    arch : str
    processor : Union[ImageProcessingMixin, FeatureExtractionMixin]
    batch_size : int, optional

    Returns
    -------
    tuple
    """
    if arch in RGB_BASED_MODELS:
        return (
            batch_size,
            processor.num_frames,
            3,
            processor.size["height"],
            processor.size["width"]
        )
    elif arch in POSE_BASED_MODELS:
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
