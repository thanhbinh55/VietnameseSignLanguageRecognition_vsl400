from configs import TransformConfig
from transformers import FeatureExtractionMixin
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    CenterCrop,
)
from .augmentations import (
    SPOTERRandomAugment,
    SPOTERGaussianNoise,
    SLGCNAugment,
)
from .transforms import (
    SPOTERShift,
    SPOTERJointSelect,
    SPOTERPad,
    SPOTERTensorToDict,
    SPOTERSingleBodyDictNormalize,
    SPOTERSingleHandDictNormalize,
    SPOTERDictToTensor,
    SLGCNJointSelect,
    SLGCNPad,
    SLGCNNormalize,
    SLGCNBoneStream,
    SLGCNMotionStream,
    NumPyToTensor,
    PoseExtract,
    PoseInterpolate,
)




def get_pose_transforms(
    split: str,
    processor: FeatureExtractionMixin,
    transform_config: TransformConfig,
) -> Compose:
    if processor.arch == "spoter":
        return _get_spoter_transforms(split, processor, transform_config)
    if processor.arch in ["sl_gcn", "dsta_slr"]:
        return _get_sl_gcn_transforms(split, processor, transform_config)
    return Compose([])


def _get_spoter_transforms(
    split: str,
    processor: FeatureExtractionMixin,
    transform_config: TransformConfig,
) -> Compose:
    transforms = [PoseExtract()]
    
    if transform_config.interpolate:
        transforms.append(PoseInterpolate())
        
    transforms.extend([
        SPOTERJointSelect(include_face=transform_config.include_face),
        SPOTERTensorToDict(),
    ])

    if split == "train" and transform_config.aug_prob > 0:
        transforms.append(
            SPOTERRandomAugment(
                transform_config.aug_prob,
                active_augs=transform_config.active_augs
            )
        )

    transforms.extend([
        SPOTERSingleBodyDictNormalize(anchor=transform_config.anchor),
        SPOTERSingleHandDictNormalize(),
        SPOTERDictToTensor(),
        SPOTERPad(processor.num_frames),
        SPOTERShift()
    ])

    if split == "train" and transform_config.add_gaussian_noise:
        transforms.append(
            SPOTERGaussianNoise(
                transform_config.gaussian_noise_mean,
                transform_config.gaussian_noise_std,
            )
        )

    return Compose(transforms)


def _get_sl_gcn_transforms(
    split: str,
    processor: FeatureExtractionMixin,
    transform_config: TransformConfig,
) -> Compose:
    transforms = [PoseExtract()]

    if transform_config.interpolate:
        transforms.append(PoseInterpolate())

    if split == "train":
        transforms.append(
            SLGCNAugment(
                aug_prob=transform_config.aug_prob,
                rotation_std=transform_config.rotation_std,
                shear_std=transform_config.shear_std,
                scale_std=transform_config.scale_std,
            )
        )

    transforms.extend(
        [
            SLGCNJointSelect(processor.num_points),
            SLGCNPad(processor.num_frames),
        ]
    )

    if processor.bone_stream:
        transforms.append(SLGCNBoneStream())
    if processor.motion_stream:
        transforms.append(SLGCNMotionStream())

    transforms.extend(
        [
            SLGCNNormalize(processor.is_vector),
            NumPyToTensor(),
        ]
    )
    return Compose(transforms)
