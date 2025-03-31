import spatial_transforms
import temporal_transforms
from datasets.videodataset import generate_dataset
from datasets.videodataset_hdf5 import generate_dataset_hdf5

from torchvision.transforms.functional import InterpolationMode

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset_folder(is_train, args):
    spatial_transform, temporal_transform = build_transform(is_train, args)

    dataset = generate_dataset(
        args.video_root_path,
        args.annotation_path,
        "train" if is_train else "test",
        spatial_transform,
        temporal_transform,
    )

    return dataset, len(dataset)


def build_dataset_hdf5(is_train, args):
    spatial_transform, temporal_transform = build_transform(is_train, args)

    dataset = generate_dataset_hdf5(
        args.video_root_path,
        args.annotation_path,
        "train" if is_train else "test",
        spatial_transform,
        temporal_transform,
    )

    return dataset, len(dataset)


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        spatial_transform = spatial_transforms.Compose(
            [
                spatial_transforms.RandomResizedCrop(
                    args.input_size,
                    scale=(args.min_scale, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                spatial_transforms.RandomHorizontalFlip(),
                spatial_transforms.ToTensor(),
                spatial_transforms.Normalize(mean=mean, std=std),
            ]
        )
        temporal_transform = []
        if args.input_frame_stride > 1:
            temporal_transform.append(
                temporal_transforms.TemporalSubsampling(args.input_frame_stride)
            )
        temporal_transform.append(
            temporal_transforms.TemporalRandomCrop(args.input_frame_size)
        )
        temporal_transform = temporal_transforms.Compose(temporal_transform)
        return spatial_transform, temporal_transform

    # eval transform
    spatial_transform = spatial_transforms.Compose(
        [
            spatial_transforms.Resize(
                args.input_size, interpolation=InterpolationMode.BICUBIC
            ),
            spatial_transforms.CenterCrop(args.input_size),
            spatial_transforms.ToTensor(),
            spatial_transforms.Normalize(mean=mean, std=std),
        ]
    )

    temporal_transform = []
    if args.input_frame_stride > 1:
        temporal_transform.append(
            temporal_transforms.TemporalSubsampling(args.input_frame_stride)
        )
    if args.eval:
        if args.shuffle_eval:
            temporal_transform.append(temporal_transforms.Shuffle(block_size=1))
        temporal_transform.append(
            temporal_transforms.TemporalUniformKCrops(
                args.input_frame_size, args.eval_clip_size
            )
        )
    else:
        temporal_transform.append(
            temporal_transforms.TemporalCenterCrop(args.input_frame_size)
        )
    temporal_transform = temporal_transforms.Compose(temporal_transform)
    return spatial_transform, temporal_transform
