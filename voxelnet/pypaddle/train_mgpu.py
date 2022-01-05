import os
import pathlib
import pickle
import shutil
import time
from functools import partial

import argparse
import numpy as np
import paddle
from visualdl import LogWriter as SummaryWriter

import paddleplus
from voxelnet.configs import cfg_from_config_py_file
from voxelnet.builder import target_assigner_builder, voxel_builder
from voxelnet.data.preprocess import merge_voxelnet_batch
from voxelnet.pypaddle.builder import (box_coder_builder, input_reader_builder,
                                       lr_scheduler_builder, optimizer_builder,
                                       voxelnet_builder)

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).astype(cls_loss.dtype) * cls_loss.reshape((
            batch_size, -1))
        cls_neg_loss = (labels == 0).astype(cls_loss.dtype) * cls_loss.reshape((
            batch_size, -1))
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted


def calculate_eta(remaining_step, speed):
    if remaining_step < 0:
        remaining_step = 0
    remaining_time = int(remaining_step * speed)
    result = "{:0>2}:{:0>2}:{:0>2}"
    arr = []
    for i in range(2, -1, -1):
        arr.append(int(remaining_time / 60 ** i))
        remaining_time %= 60 ** i
    return result.format(*arr)


def example_convert_to_paddle(example, dtype=paddle.float32,
                              ) -> dict:
    example_paddle = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_paddle[k] = paddle.to_tensor(v, dtype=dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_paddle[k] = paddle.to_tensor(
                v, dtype=paddle.int32)
        elif k in ["anchors_mask"]:
            example_paddle[k] = paddle.to_tensor(
                v, dtype=paddle.uint8)
        else:
            example_paddle[k] = v
    return example_paddle


def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          accum_step=1,
          pickle_result=True):
    """train a VoxelNet model specified by a config file.
    """

    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    if nranks > 1:
        # 1. initialize parallel environment
        paddle.distributed.init_parallel_env()

    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = paddleplus.train.create_folder(model_dir)

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.py"
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))
    config = cfg_from_config_py_file(config_path)
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.voxelnet
    train_cfg = config.train_config

    class_names = list(input_cfg.class_names)
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    ######################
    # BUILD NET
    ######################
    center_limit_range = model_cfg.post_center_limit_range
    net = voxelnet_builder.build(model_cfg, voxel_generator, target_assigner)
    if nranks>1:
        dp_net = paddle.DataParallel(net)
    # net_train = paddle.nn.DataParallel(net).to('gpu')
    print("num_trainable parameters:", len(list(net.parameters())))
    # for n, p in net.named_parameters():
    #     print(n, p.shape)
    ######################
    # BUILD OPTIMIZER
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    paddleplus.train.try_restore_latest_checkpoints(model_dir, [net])
    gstep = net.get_global_step() - 1
    optimizer_cfg = train_cfg.optimizer

    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, gstep)
    if nranks > 1:
        optimizer = optimizer_builder.build(optimizer_cfg, lr_scheduler, dp_net.parameters())
    else:
        optimizer = optimizer_builder.build(optimizer_cfg, lr_scheduler, net.parameters())
    # must restore optimizer AFTER using MixedPrecisionWrapper
    paddleplus.train.try_restore_latest_checkpoints(model_dir,
                                                    [optimizer])

    float_dtype = paddle.float32
    ######################
    # PREPARE INPUT
    ######################

    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

    batch_sampler = paddle.io.DistributedBatchSampler(
        dataset, batch_size=input_cfg.batch_size, shuffle=True, drop_last=False)

    dataloader = paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=input_cfg.num_workers,
        collate_fn=merge_voxelnet_batch,
        worker_init_fn=_worker_init_fn,
        use_shared_memory=False,
        persistent_workers=True,
    )
    data_iter = iter(dataloader)

    ######################
    # TRAINING
    ######################
    log_path = model_dir / 'log.txt'
    logf = open(log_path, 'a')
    logf.write("\n")
    summary_dir = model_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(summary_dir))

    total_step_elapsed = 0
    remain_steps = train_cfg.steps - net.get_global_step()
    t = time.time()
    ckpt_start_time = t

    # total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    total_loop = remain_steps // train_cfg.steps_per_eval + 1
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    if train_cfg.steps % train_cfg.steps_per_eval == 0:
        total_loop -= 1
    optimizer.clear_grad()
    try:
        for _ in range(total_loop):
            if total_step_elapsed + train_cfg.steps_per_eval > train_cfg.steps:
                steps = train_cfg.steps % train_cfg.steps_per_eval
            else:
                steps = train_cfg.steps_per_eval
            for step in range(steps):
                try:
                    example = next(data_iter)
                except StopIteration:
                    print("end epoch")
                    if clear_metrics_every_epoch:
                        net.clear_metrics()
                    data_iter = iter(dataloader)
                    example = next(data_iter)
                example_paddle = example_convert_to_paddle(example, float_dtype)

                batch_size = example["anchors"].shape[0]

                if nranks > 1:
                    ret_dict = dp_net(example_paddle)
                else:
                    ret_dict = net(example_paddle)

                # box_preds = ret_dict["box_preds"]
                cls_preds = ret_dict["cls_preds"]
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                cls_pos_loss = ret_dict["cls_pos_loss"]
                cls_neg_loss = ret_dict["cls_neg_loss"]
                loc_loss = ret_dict["loc_loss"]
                cls_loss = ret_dict["cls_loss"]
                if model_cfg.use_direction_classifier:
                    dir_loss_reduced = ret_dict["dir_loss_reduced"]
                cared = ret_dict["cared"]
                labels = example_paddle["labels"]
                loss = loss / accum_step
                loss.backward()
                if step % accum_step ==0:
                    optimizer.step()
                    optimizer.clear_grad()
                    # update lr
                    if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                        lr_sche = optimizer.user_defined_optimizer._learning_rate
                    else:
                        lr_sche = optimizer._learning_rate
                    if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                        lr_sche.step()
                net.update_global_step()
                net_metrics = net.update_metrics(cls_loss_reduced.detach(),
                                                 loc_loss_reduced.detach(), cls_preds.detach(),
                                                 labels, cared)

                step_time = (time.time() - t)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].astype(paddle.float32).sum().cpu().numpy())
                num_neg = int((labels == 0)[0].astype(paddle.float32).sum().cpu().numpy())
                if 'anchors_mask' not in example_paddle:
                    num_anchors = example_paddle['anchors'].shape[1]
                else:
                    num_anchors = int(example_paddle['anchors_mask'].astype(paddle.int32)[0].sum())
                global_step = net.get_global_step()
                remain_steps = remain_steps - 1
                eta = calculate_eta(remain_steps, step_time)
                if global_step % display_step == 0 and local_rank == 0:
                    loc_loss_elem = [
                        float(loc_loss.detach().cpu()[:, :, i].sum().numpy() /
                              batch_size) for i in range(loc_loss.detach().cpu().numpy().shape[-1])
                    ]
                    metrics["step"] = global_step
                    metrics["steptime"] = step_time
                    metrics.update(net_metrics)
                    metrics["loss"] = {}
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())
                    metrics["num_vox"] = int(example_paddle["voxels"].shape[0])
                    metrics["num_pos"] = int(num_pos)
                    metrics["num_neg"] = int(num_neg)
                    metrics["num_anchors"] = int(num_anchors)
                    metrics["lr"] = float(
                        optimizer.get_lr())
                    metrics["image_idx"] = example['image_idx'][0].cpu().numpy()
                    flatted_metrics = flat_nested_json_dict(metrics)
                    flatted_summarys = flat_nested_json_dict(metrics, "/")
                    metrics["eta"] = eta
                    for k, v in flatted_summarys.items():
                        if isinstance(v, (list, tuple)):
                            v = {str(i): e for i, e in enumerate(v)}
                            for i in v:
                                writer.add_scalar(k + '/' + i, v[i], global_step)
                        else:
                            writer.add_scalar(k, v, global_step)
                    metrics_str_list = []
                    flatted_metrics = flat_nested_json_dict(metrics)
                    for k, v in flatted_metrics.items():
                        if isinstance(v, float):
                            metrics_str_list.append(f"{k}={v:.3}")
                        elif isinstance(v, (list, tuple)):
                            if v and isinstance(v[0], float):
                                v_str = ', '.join([f"{e:.3}" for e in v])
                                metrics_str_list.append(f"{k}=[{v_str}]")
                            else:
                                metrics_str_list.append(f"{k}={v}")
                        else:
                            metrics_str_list.append(f"{k}={v}")
                    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    log_str = ', '.join(metrics_str_list)
                    log_str = timestr + ', ' + log_str
                    print(log_str, file=logf)
                    print(log_str)
                ckpt_elasped_time = time.time() - ckpt_start_time
                if ckpt_elasped_time > train_cfg.save_checkpoints_secs and local_rank == 0:
                    paddleplus.train.save_models(model_dir, [net, optimizer],
                                                 net.get_global_step())
                    ckpt_start_time = time.time()

            # paddle.device.cuda.empty_cache() # 清空显存 防止溢出
            total_step_elapsed += steps
            paddleplus.train.save_models(model_dir, [net, optimizer],
                                             net.get_global_step())
    except Exception as e:
        paddleplus.train.save_models(model_dir, [net, optimizer],
                                     net.get_global_step())
        logf.close()
        raise e
    # save model before exit
    paddleplus.train.save_models(model_dir, [net, optimizer],
                                 net.get_global_step())
    logf.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        '--config_path', dest="config_path", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--result_path',
        dest='result_path',
        help='The directory for saving the model snapshot',
        type=str,
        default=None)
    parser.add_argument(
        '--create_folder',
        dest='create_folder',
        help='Whether to create_folder in save_dir/.No use now.',
        type=bool,
        default=False)
    parser.add_argument(
        '--display_step',
        dest='display_step',
        help='Display logging information at every log_iters',
        default=50,
        type=int)
    parser.add_argument(
        '--summary_step',
        dest='summary_step',
        help='do summary. No use now.',
        default=10,
        type=int)
    parser.add_argument(
        '--accum_step',
        dest='accum_step',
        help='do grad accum..',
        default=1,
        type=int)
    return parser.parse_args()


def main(args):
    train(config_path=args.config_path,
          model_dir=args.model_dir,
          result_path=args.result_path,
          create_folder=args.create_folder,
          display_step=args.display_step,
          summary_step=args.summary_step,
          accum_step=args.accum_step,
          )

if __name__ == '__main__':
    args = parse_args()
    main(args)