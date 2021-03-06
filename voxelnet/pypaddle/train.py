import os
import pathlib
import pickle
import shutil
import time
from functools import partial

import fire
import numpy as np
import paddle
from visualdl import LogWriter as SummaryWriter

import paddleplus
import voxelnet.data.kitti_common as kitti
from voxelnet.configs import cfg_from_config_py_file
from voxelnet.builder import target_assigner_builder, voxel_builder
from voxelnet.data.preprocess import merge_voxelnet_batch
from voxelnet.pypaddle.builder import (box_coder_builder, input_reader_builder,
                                     lr_scheduler_builder, optimizer_builder,
                                     voxelnet_builder)
from voxelnet.utils.eval import get_coco_eval_result, get_official_eval_result
from voxelnet.utils.progress_bar import ProgressBar


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
        arr.append(int(remaining_time / 60**i))
        remaining_time %= 60**i
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
    paddle.device.set_device("gpu")
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

    net.train()
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
    optimizer = optimizer_builder.build(optimizer_cfg,lr_scheduler,net.parameters())
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
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

    dataloader = paddle.io.DataLoader(
        dataset=dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.num_workers,
        collate_fn=merge_voxelnet_batch,
        worker_init_fn=_worker_init_fn,
        use_shared_memory=False,
        persistent_workers=True,
        )
    eval_dataloader = paddle.io.DataLoader(
        dataset=eval_dataset,
        batch_size=eval_input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.num_workers,
        collate_fn=merge_voxelnet_batch,
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

    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    # total_loop = remain_steps // train_cfg.steps_per_eval + 1
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
                if step % accum_step == 0:
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
                if global_step % display_step == 0:
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
                    # if unlabeled_training:
                    #     metrics["loss"]["diff_rt"] = float(
                    #         diff_loc_loss_reduced.detach().cpu().numpy())
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
                                writer.add_scalar(k+'/'+i, v[i], global_step)
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
                if ckpt_elasped_time > train_cfg.save_checkpoints_secs:
                    paddleplus.train.save_models(model_dir, [net, optimizer],
                                                 net.get_global_step())
                    ckpt_start_time = time.time()
                
                # paddle.device.cuda.empty_cache() # ???????????? ????????????

            total_step_elapsed += steps
            paddleplus.train.save_models(model_dir, [net, optimizer],
                                         net.get_global_step())
            net.eval()
            result_path_step = result_path / f"step_{net.get_global_step()}"
            result_path_step.mkdir(parents=True, exist_ok=True)
            print("#################################")
            print("#################################", file=logf)
            print("# EVAL")
            print("# EVAL", file=logf)
            print("#################################")
            print("#################################", file=logf)
            print("Generate output labels...")
            print("Generate output labels...", file=logf)
            t = time.time()
            dt_annos = []
            prog_bar = ProgressBar()
            prog_bar.start(len(eval_dataset) // eval_input_cfg.batch_size + 1)
            for example in iter(eval_dataloader):
                example = example_convert_to_paddle(example, float_dtype)
                if pickle_result:
                    dt_annos += predict_kitti_to_anno(
                        net, example, class_names, center_limit_range,
                        model_cfg.lidar_input)
                else:
                    _predict_kitti_to_file(net, example, result_path_step,
                                           class_names, center_limit_range,
                                           model_cfg.lidar_input)

                prog_bar.print_bar()

            sec_per_ex = len(eval_dataset) / (time.time() - t)
            print(f"avg forward time per example: {net.avg_forward_time:.3f}")
            print(
                f"avg postprocess time per example: {net.avg_postprocess_time:.3f}"
            )

            net.clear_time_metrics()
            print(f'generate label finished({sec_per_ex:.2f}/s). start eval:')
            print(
                f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                file=logf)
            gt_annos = [
                info["annos"] for info in eval_dataset.dataset.kitti_infos
            ]
            if not pickle_result:
                dt_annos = kitti.get_label_annos(result_path_step)
            if paddle.device.cuda.device_count() == 0:
                print("Evaluation only support gpu.")
            else:
                result = get_official_eval_result(gt_annos, dt_annos, class_names)
                print(result, file=logf)
                print(result)
                writer.add_text('eval_result', result, global_step)
                result = get_coco_eval_result(gt_annos, dt_annos, class_names)
                print(result, file=logf)
                print(result)
                writer.add_text('eval_result', result, global_step)
            if pickle_result:
                with open(result_path_step / "result.pkl", 'wb') as f:
                    pickle.dump(dt_annos, f)
            
            net.train()
    except Exception as e:
        paddleplus.train.save_models(model_dir, [net, optimizer],
                                     net.get_global_step())
        logf.close()
        raise e
    # save model before exit
    paddleplus.train.save_models(model_dir, [net, optimizer],
                                 net.get_global_step())
    logf.close()


def _predict_kitti_to_file(net,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].cpu().numpy()
            box_preds = preds_dict["box3d_camera"].cpu().numpy()
            scores = preds_dict["scores"].cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)


def predict_kitti_to_anno(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i].cpu().numpy()
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
    return annos


def evaluate(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True):
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = cfg_from_config_py_file(config_path)
    # with open(config_path, "r") as f:
    #     proto_str = f.read()
    #     text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.voxelnet
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    net = voxelnet_builder.build(model_cfg, voxel_generator, target_assigner)

    if ckpt_path is None:
        paddleplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        paddleplus.train.restore(ckpt_path, net)

    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = paddle.io.DataLoader(
        dataset=eval_dataset,
        batch_size=input_cfg.batch_size,
        shuffle=False,
        num_workers=input_cfg.num_workers,
        collate_fn=merge_voxelnet_batch,
        use_shared_memory=False,
        persistent_workers=True)


    float_dtype = paddle.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    global_set = None
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start(len(eval_dataset) // input_cfg.batch_size + 1)

    for example in iter(eval_dataloader):
        example = example_convert_to_paddle(example, float_dtype)
        if pickle_result:
            dt_annos += predict_kitti_to_anno(
                net, example, class_names, center_limit_range,
                model_cfg.lidar_input, global_set)
        else:
            _predict_kitti_to_file(net, example, result_path_step, class_names,
                                   center_limit_range, model_cfg.lidar_input)
        bar.print_bar()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')

    print(f"avg forward time per example: {net.avg_forward_time:.3f}")
    print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")
    if paddle.device.cuda.device_count() == 0:
        print("Evaluation only support gpu. exit.")
    if not predict_test:
        gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)
        result = get_official_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        result = get_coco_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)


if __name__ == '__main__':
    fire.Fire()
