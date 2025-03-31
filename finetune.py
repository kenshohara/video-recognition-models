import argparse
import datetime
import json
import numpy as np
import os
import sys
import math
import pickle
import random
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler

from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import accuracy

import util.misc as misc
from util.datasets import build_dataset_folder, build_dataset_hdf5

import models_video_vit
import models_3d_resnet


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--val_batch_size",
        default=-1,
        type=int,
    )
    parser.add_argument("--epochs", default=200, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--input_frame_size", default=1, type=int, help="number of input video frames"
    )

    parser.add_argument(
        "--input_frame_stride", default=1, type=int, help="stride of input video frames"
    )

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        "--adamw_b2",
        type=float,
        default=0.999,
        help="B2 of AdamW",
    )

    # Augmentation parameters
    parser.add_argument(
        "--min_scale",
        type=float,
        default=0.08,
        metavar="N",
        help="min scale of RandomResizedCrop",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--input_frame_size_pretrain", default=-1, type=int)
    parser.add_argument("--global_pool", type=str, default="avg")
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Dataset parameters
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )

    parser.add_argument(
        "--video_root_path",
        default="",
        type=Path,
        help="frames dir path",
    )
    parser.add_argument(
        "--annotation_path",
        default="",
        type=Path,
        help="annotations dir path",
    )
    parser.add_argument(
        "--dataset_impl_type",
        default="folder",
        type=str,
        help="folder | hdf5",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--eval_clip_size", default=10, type=int, help="num of clips for inference"
    )
    parser.add_argument(
        "--save_inference_results",
        action="store_true",
        help="",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--val_num_workers", default=-1, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--distributed", action="store_true")

    parser.add_argument(
        "--checkpoint",
        default=10,
        type=int,
        metavar="N",
        help="save and eval on every checkpoint epoch",
    )

    return parser


def main(args):
    args.local_rank = 0
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=args.world_size,
            rank=args.rank,
        )
        print(args.rank, args.world_size, args.local_rank)
        assert args.rank >= 0

        args.is_master = args.rank == 0
        misc.setup_for_distributed(args.is_master)

    seed = args.seed + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    if args.val_batch_size < 0:
        args.val_batch_size = args.batch_size
    if args.val_num_workers < 0:
        args.val_num_workers = args.num_workers

    if args.dataset_impl_type == "folder":
        dataset_train, dataset_size_train = build_dataset_folder(
            is_train=True, args=args
        )
        dataset_val, dataset_size_val = build_dataset_folder(is_train=False, args=args)
    else:
        dataset_train, dataset_size_train = build_dataset_hdf5(is_train=True, args=args)
        dataset_val, dataset_size_val = build_dataset_hdf5(is_train=False, args=args)

    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )
    else:
        sampler_train = None
        sampler_val = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler_train,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=True,
        sampler=sampler_val,
    )

    if "vit" in args.model:
        model = models_video_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            img_size=args.input_size,
            num_frames=args.input_frame_size,
        )
    elif "resnet" in args.model:
        model = models_3d_resnet.__dict__[args.model](n_classes=args.nb_classes)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu", weights_only=False)

        print(f"Load pre-trained checkpoint from: {args.finetune}")
        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "fc.weight", "fc.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        if "vit" in args.model:
            if args.input_frame_size_pretrain < 0:
                original_num_frames = args.input_frame_size
            misc.interpolate_pos_embed(
                model,
                checkpoint_model,
                args.input_frame_size,
                original_num_frames,
            )

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if "resnet" in args.model:
            trunc_normal_(model.fc.weight, std=2e-5)
        else:
            trunc_normal_(model.head.weight, std=2e-5)

    model.to(args.device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

    eff_batch_size = args.batch_size * args.world_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
        print(f"base lr: {args.blr:.2e}")
    print(f"actual lr: {args.lr:.2e}")

    print(f"effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()]
        )
    else:
        model = torch.nn.parallel.DataParallel(model)
    model_without_ddp = model.module

    if "resnet" not in args.model:
        param_groups = misc.param_groups(
            model_without_ddp,
            args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
        )
        optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, args.adamw_b2)
        )
    else:
        param_groups = model_without_ddp.parameters()
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, args.adamw_b2),
            weight_decay=args.weight_decay,
        )
    loss_scaler = GradScaler()

    if args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = {criterion}")

    if args.resume:
        misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

    if args.eval:
        print("Start evaluation")
        test_stats, inference_results = evaluate(
            data_loader_val,
            model,
            torch.cuda.current_device(),
            args.save_inference_results,
        )
        print(
            f"Accuracy of the network on the {dataset_size_val} test videos: {test_stats['acc1']:.1f}%"
        )

        log_stats = {f"test_{k}": v for k, v in test_stats.items()}

        if args.output_dir and misc.is_main_process():
            with open(
                os.path.join(args.output_dir, "log_eval.txt"),
                mode="a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.save_inference_results:
                with open(os.path.join(args.output_dir, f"inference.pkl"), "wb") as f:
                    pickle.dump(inference_results, f)

        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            torch.cuda.current_device(),
            epoch,
            loss_scaler,
            args.clip_grad,
            args=args,
        )
        test_stats = {}
        if args.output_dir and (
            (epoch + 1) % args.checkpoint == 0 or epoch + 1 == args.epochs
        ):
            if misc.is_main_process():
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

            test_stats, _ = evaluate(
                data_loader_val,
                model,
                torch.cuda.current_device(),
                args.save_inference_results,
            )
            print(
                f"Accuracy of the network on the {dataset_size_val} test images: {test_stats['acc1']:.1f}%"
            )
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    args=None,
):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    for data_iter_step, (_, _, samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        misc.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.detach().cpu().item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss_scaler.scale(loss).backward()
        if max_norm is not None:
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        loss_scaler.step(optimizer)
        loss_scaler.update()

        optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, save_inference_results):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", misc.SmoothedValue())
    metric_logger.add_meter("acc1", misc.SmoothedValue())
    metric_logger.add_meter("acc5", misc.SmoothedValue())
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    results = {}
    for keys, _, images, target in metric_logger.log_every(data_loader, 10, header):
        batch_size = images.shape[0]

        is_reshape = False
        if len(images.shape) > 5:
            n_clips = images.shape[1]
            images = images.reshape([batch_size * n_clips, *images.shape[2:]])
            is_reshape = True

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(images)

        if is_reshape:
            output = output.reshape([batch_size, n_clips, *output.shape[1:]])
            output = output.mean(dim=1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        if save_inference_results:
            output = output.cpu().float()
            for i, key in enumerate(keys):
                results[key] = output[i]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, results


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
