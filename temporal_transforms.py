import random
import math


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = Compose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]

                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices


class LoopPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[: self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):
    def __init__(self, size):
        self.size = size
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        if len(out) < self.size:
            out = self.loop(out)

        return out


class TemporalUniformKCrops(object):
    def __init__(self, size, k=1):
        self.size = size
        self.k = k
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        n_frames = len(frame_indices)

        if n_frames <= self.size:
            for _ in range(self.k):
                out = [self.loop(frame_indices) for _ in range(self.k)]
            return out

        if self.k > 1:
            stride = math.ceil((n_frames - self.size) / (self.k - 1)) - 1
        else:
            stride = 1
        out = []
        for clip_index in range(self.k):
            s = min(n_frames - 1, clip_index * stride)
            e = min(n_frames, s + self.size)
            sample = [frame_indices[i] for i in range(s, e)]
            if len(sample) < self.size:
                out.append(self.loop(sample))
            else:
                out.append(sample)

        return out


class TemporalSubsampling(object):
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, frame_indices):
        return frame_indices[:: self.stride]


class Shuffle(object):
    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, frame_indices):
        frame_indices = [
            frame_indices[i : (i + self.block_size)]
            for i in range(0, len(frame_indices), self.block_size)
        ]
        random.shuffle(frame_indices)
        frame_indices = [t for block in frame_indices for t in block]
        return frame_indices
