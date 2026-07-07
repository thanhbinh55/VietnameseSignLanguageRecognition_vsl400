import torch
import numpy as np
import onnxruntime as ort
from typing import Union
from transformers import Pipeline, AutoConfig
from torchvision.transforms.v2 import Compose
from pathlib import Path

from features.transforms.base import PoseExtract, PoseInterpolate
from features.transforms.spoter import (
    SPOTERJointSelect as JointSelect,
    SPOTERTensorToDict as TensorToDict,
    SPOTERSingleBodyDictNormalize as SingleBodyDictNormalize,
    SPOTERSingleHandDictNormalize as SingleHandDictNormalize,
    SPOTERDictToTensor as DictToTensor,
    SPOTERShift as Shift,
)

class SPOTERGraphClassificationPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.pop("use_onnx", False):
            checkpoint_dir = Path(self.model.config._name_or_path)
            model_name = checkpoint_dir.name
            model_file = checkpoint_dir / f"{model_name}.onnx"
            if not model_file.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at '{model_file}'. "
                    "Export the model first using convert_model_to_onnx.py."
                )
            self.config = AutoConfig.from_pretrained(
                str(checkpoint_dir),
                trust_remote_code=True,
            )
            self.id2label = self.config.id2label
            self.model = ort.InferenceSession(str(model_file))
        else:
            self.id2label = self.model.config.id2label

        self.transforms = Compose(
            [
                PoseExtract(keep_face=False),
                PoseInterpolate(confidence_threshold=0.5),
                JointSelect(include_face=False),
                TensorToDict(),
                SingleBodyDictNormalize(anchor="neck"),
                SingleHandDictNormalize(),
                DictToTensor(),
                Shift()
            ]
        )

    def _sanitize_parameters(self, **kwargs):
        # Sanitize the parameters for preprocessing
        preprocess_kwargs = {}
        # Sanitize the parameters for the forward pass
        forward_kwargs = {}
        # Sanitize the parameters for postprocessing
        postprocess_kwargs = {}
        postprocess_kwargs["top_k"] = kwargs.get("top_k", 3)

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocesses the inputs to the model.

        Parameters
        ----------
        inputs : Union[np.ndarray, torch.Tensor]
            The inputs to the model (time, height, width, channels).

        Returns
        -------
        torch.Tensor
            The preprocessed inputs (batch, channels, time, height, width).
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs)
        return self.transforms(inputs).unsqueeze(0)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, ort.InferenceSession):
            inputs = inputs.cpu().numpy()
            return torch.from_numpy(self.model.run(None, {"poses": inputs})[0])
        return self.model(inputs.to(self.device)).logits

    def postprocess(self, logits: torch.Tensor, top_k: int = 3) -> list:
        logits = logits.cpu()

        topk_scores, topk_indices = torch.topk(logits, top_k, dim=1)
        topk_scores = torch.nn.functional.softmax(topk_scores, dim=1)
        topk_scores = topk_scores.squeeze().detach().numpy()
        topk_indices = topk_indices.squeeze().detach().numpy()
        
        topk_scores = np.atleast_1d(topk_scores)
        topk_indices = np.atleast_1d(topk_indices)

        return [
            {
                'gloss': self.id2label[str(topk_indices[i])],
                'score': topk_scores[i],
            }
            for i in range(top_k)
        ]
