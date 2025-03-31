from pathlib import Path

import torch
import torch.utils.data as data
from PIL import Image
from tqdm.auto import tqdm


class ImageLoaderPIL(object):

    def __call__(self, path):
        with path.open("rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB")


class VideoLoader(object):
    def __init__(
        self, image_name_formatter=lambda x: f"{x:05d}.jpg", image_loader=None
    ):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            video.append(self.image_loader(image_path))

        return video


class VideoDataset(data.Dataset):

    def __init__(
        self,
        video_root_path,
        annotation_path,
        subset,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None,
    ):
        self.data = self.__make_dataset(video_root_path, annotation_path, subset)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.loader = VideoLoader()

    def __make_dataset(self, video_root_path, annotation_path, subset):
        if isinstance(video_root_path, str):
            video_root_path = Path(video_root_path)
        if isinstance(annotation_path, str):
            annotation_path = Path(annotation_path)

        with open(annotation_path / f"{subset}.csv", "r") as f:
            data = [line.strip().split(" ") for line in f]

        dataset = []
        for i, (video_id, n_frames, label) in enumerate(tqdm(data)):
            video_path = video_root_path / video_id
            segment = [1, int(n_frames) + 1]
            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                "video": video_path,
                "segment": segment,
                "frame_indices": frame_indices,
                "video_id": video_id,
                "label": int(label),
            }
            dataset.append(sample)

        return dataset

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __loading_multi_clips(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clips.append(self.__loading(path, clip_frame_indices))
            segments.append([min(clip_frame_indices), max(clip_frame_indices) + 1])

        return torch.stack(clips), segments

    def __getitem__(self, index):
        path = self.data[index]["video"]
        target = self.data[index]["label"]

        frame_indices = self.data[index]["frame_indices"]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        if isinstance(frame_indices[0], list):
            clip, _ = self.__loading_multi_clips(path, frame_indices)
        else:
            clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return path.name, str(path), clip, target

    def __len__(self):
        return len(self.data)


def generate_dataset(
    video_root_path,
    annotation_path,
    subset,
    spatial_transform,
    temporal_transform,
    target_transform=None,
):
    return VideoDataset(
        video_root_path,
        annotation_path,
        subset,
        spatial_transform,
        temporal_transform,
        target_transform,
    )
