# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import pdb
import sys
from typing import Iterable

import torch
import util.misc as utils
# from datasets.coco_eval import CocoEvaluator
# from datasets.panoptic_eval import PanopticEvaluator
from datasets.progress_amount_eval import CocoEvaluator, ProgressEvaluator, AmountEvaluator
from datasets.data_prefetcher import data_prefetcher
from my_debug import save_tensor_as_image, convert_sample_and_boxes, tensor_to_pil, draw_bboxes_on_pil
import torch.nn.functional as F
# import xml.etree.ElementTree as ET
import numpy as np
import util.misc as utils
from util.misc import NestedTensor
from my_debug import get_data_from_string, get_duration_key_time, get_duration, get_duration_norm

def train_one_epoch_on_saclassifier(modelG: torch.nn.Module,
                                    modelH: torch.nn.Module,
                                    criterion: torch.nn.Module,
                                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                                    device: torch.device, epoch: int, max_norm: float = 0,
                                    path_to_input_debug: str = None,
                                    freeze_modelG: bool = True,
                                    logger: bool = None):
    if freeze_modelG:
        modelG.eval()
    else:
        modelG.train()
    modelH.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    # targets[0].keys(), 'boxes', 'labels', ...

    max_iter = len(data_loader)

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for i_batch in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        # check mask has all False values. samples.mask.shape, samples.tensors.shape
        outputG = modelG(samples)
        outputs = modelH(outputG)

        if (path_to_input_debug is not None) and (i_batch < 10):
            path_to_file = os.path.join(path_to_input_debug,
                                        'e%d_i%07d_id%d.jpg' % (epoch, i_batch, targets[0]['image_id'].item()))
            # samples.tensors[0]: [3, h, w]
            draw_image = samples.tensors[0].detach().cpu()
            draw_mask = samples.mask[0].detach().cpu()
            scaled_boxes = targets[0]['boxes'].detach().cpu()  # n_box, 4

            masked_image, unscaled_boxes = convert_sample_and_boxes(draw_image, draw_mask, scaled_boxes)
            save_tensor_as_image(path_to_file, masked_image, unscaled_boxes,
                                 targets[0]['labels'].detach().cpu())

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(modelH.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(modelH.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        if logger is not None:
            info = {
                'loss': loss_value,
                'lr': optimizer.param_groups[0]["lr"],
            }
            logger.add_scalars("iter", info, epoch * max_iter + i_batch)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_on_saclassifier(modelG, modelH, criterion, postprocessors, data_loader, base_ds, device, output_dir,
                         save_result_image=False, num_saved_results=0, vis_th=0.5, num_classes=5):
    # get all evaluation results
    modelG.eval()
    modelH.eval()
    criterion.eval()

    # postprocessors convert bbox type and scale
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    sac_evaluator = SACEvaluator(num_classes)

    coco_evaluator = None
    if 'bbox' in postprocessors.keys():
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        coco_evaluator = CocoEvaluator(base_ds, iou_types)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    if save_result_image:
        path_to_result_images = os.path.join(output_dir, 'vis')
        if not os.path.exists(path_to_result_images):
            os.makedirs(path_to_result_images)

    count_samples = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # this can be multiple.

        # keys: (['pred_logits', 'pred_boxes', 'amount_score', 'progress_score', 'enc_outputs', 'aux_outputs', 'features'])
        # outputG['features'] # n_batch x 1024
        outputG = modelG(samples)
        outputs = modelH(outputG)

        loss_dict = criterion(outputs, targets) # just loss for training
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        if coco_evaluator is not None:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            res = {target['image_id'].item(): output for target, output in
                   zip(targets, results)}
            # res: dict ('image_id') of dict ('scores', 'labels', 'boxes', )

            coco_evaluator.update(res)

        if sac_evaluator is not None:
            sac_evaluator.update(outputs, targets)

        pdb.set_trace()
        if save_result_image:
            for i, target in enumerate(targets):    # per batch
                count_samples += 1
                if count_samples <= num_saved_results:
                    image_id = target["image_id"].item()
                    file_name = f"{image_id:012d}.jpg"
                    save_file_name = os.path.join(path_to_result_images, file_name)

                    draw_image = samples.tensors[i].detach().cpu()
                    draw_mask = samples.mask[i].detach().cpu()

                    if coco_evaluator is not None:
                        scaled_boxes = res[image_id]['boxes'].detach().cpu()  # n_box, 4
                        res_labels = res[image_id]['labels'].detach().cpu()
                        res_scores = res[image_id]['scores'].detach().cpu()
                    else:
                        scaled_boxes = res_labels = res_scores = None

                    masked_image, unscaled_boxes = convert_sample_and_boxes(draw_image, draw_mask,
                                                                            scaled_boxes)

                    pil_image = tensor_to_pil(masked_image,
                                              orig_size=(targets[i]['orig_size'][1].item(),
                                                         targets[i]['orig_size'][0].item()))
                    if coco_evaluator is not None:
                        # draw bboxes
                        pil_image = draw_bboxes_on_pil(pil_image, unscaled_boxes, res_labels,
                                                       scores=res_scores,
                                                       vis_th=vis_th)

                    # draw saclass predictions
                    unscaled_boxes = torch.tensor([[1, 1, 10, 10], [1, 21, 10, 30],
                                    [1, 41, 10, 50], [1, 61, 10, 70]])
                    res_labels = ['refill', 'trash', 'dessert', 'lost']
                    res_scores = torch.sigmoid(outputs)[0, 1:].detach().cpu()
                    vis_th = 0.0
                    pil_image = draw_bboxes_on_pil(pil_image, unscaled_boxes, res_labels,
                                                   scores=res_scores,
                                                   vis_th=vis_th)

                    pil_image.save(save_file_name, 'JPEG')


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    sac_res = None
    if sac_evaluator is not None:
        sac_res = sac_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if sac_res is not None:
        stats.update(sac_res)

    return stats, coco_evaluator


class SACEvaluator(object):
    def __init__(self, num_classes, thresh=0.5):
        self.predictions = np.array([], dtype=np.float32).reshape(0, num_classes)
        self.groundtruths = np.array([], dtype=np.float32).reshape(0, num_classes)
        self.thresh = thresh

    def update(self, predictions, groundtruths):
        # torch to numpy
        self.predictions = np.concatenate([self.predictions, predictions.detach().cpu().numpy()], axis=0)

        target_classes_onehot = np.zeros([predictions.shape[0], predictions.shape[1]])
        for i_t, tgt in enumerate(groundtruths):
            for i_l in tgt['labels']:
                target_classes_onehot[i_t, i_l] = 1
        self.groundtruths = np.concatenate([self.groundtruths, target_classes_onehot], axis=0)

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        all_groundtruths = utils.all_gather(self.groundtruths)

        merged_predictions = np.array([], dtype=np.float32).reshape(0, 3)
        merged_groundtruths = np.array([], dtype=np.float32).reshape(0, 3)
        for p, gt in zip(all_predictions, all_groundtruths):
            #merged_predictions += p
            merged_predictions = np.concatenate([merged_predictions, p], axis=0)
            merged_groundtruths = np.concatenate([merged_groundtruths, gt], axis=0)
        self.predictions = merged_predictions
        self.groundtruths = merged_groundtruths

    def summarize(self, ignore_auc=False):
        if utils.is_main_process():
            from sklearn.metrics import (accuracy_score,
                                         precision_score, recall_score, f1_score,
                                         roc_auc_score, roc_curve,
                                         average_precision_score)

            ys_prob = torch.sigmoid(torch.Tensor(self.predictions))
            ys_pred = ys_prob > self.thresh
            ys_true = self.groundtruths

            list_acc = []   # accuracy
            list_prec = []
            list_recl = []
            list_f1 = []    # f1 score
            list_auc = []   # roc auc
            list_prauc= []  # pr auc
            list_eer_fpr = []
            list_eer_fnr = []
            list_eer_th = []

            num_classes = self.predictions.shape[1]
            for i_c in range(1, num_classes):
                y_prob = ys_prob[:, i_c]
                y_pred = ys_pred[:, i_c]
                y_true = ys_true[:, i_c]

                list_acc.append(accuracy_score(y_true, y_pred))     # accuracy
                list_prec.append(precision_score(y_true, y_pred))
                list_recl.append(recall_score(y_true, y_pred))
                list_f1.append(f1_score(y_true, y_pred))            # f1 score

                if ignore_auc:
                    auc_score = 0
                    prauc_score = 0
                    eer_score_fpr = 0
                    eer_score_fnr = 0
                    eer_threshold = -1
                else:
                    try:
                        auc_score = roc_auc_score(y_true, y_prob)   # roc auc
                        prauc_score = average_precision_score(y_true, y_prob)   # pr auc

                        fpr, tpr, threshold = roc_curve(y_true, y_prob, pos_label=1)
                        fnr = 1 - tpr
                        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
                        eer_score_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
                        eer_score_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
                    except ValueError:
                        auc_score = 0
                        prauc_score = 0
                        eer_score_fpr = 0
                        eer_score_fnr = 0
                        eer_threshold = -1
                list_auc.append(auc_score)
                list_prauc.append(prauc_score)
                list_eer_fpr.append(float(eer_score_fpr))
                list_eer_fnr.append(float(eer_score_fnr))
                list_eer_th.append(float(eer_threshold))

            dic_res = {
                'sacs_acc': list_acc,
                'sacs_f1score': list_f1,
                'sacs_auc': list_auc,
                'sacs_prauc': list_prauc,
                'sacs_eer_fpr': list_eer_fpr,
                'sacs_eer_fnr': list_eer_fnr,
                'sacs_eer_th': list_eer_th,
                'sacs_precision': list_prec,
                'sacs_recall': list_recl
            }
            return dic_res
        return None

    # def summarize(self):
    #     if utils.is_main_process():
    #         num_samples = self.groundtruths.shape[0]
    #
    #         src_prob = torch.sigmoid(torch.Tensor(self.predictions))
    #         prob_to_class = src_prob > self.thresh
    #
    #         acc = torch.sum(prob_to_class == torch.Tensor(self.groundtruths), dim=0).float() / float(num_samples)
    #         acc = acc[1:]
    #         # pred_tensor = torch.Tensor(self.predictions)
    #         # gt_tensor = torch.Tensor(self.groundtruths)
    #         # gt_tensor = gt_tensor.unsqueeze(1)
    #         # acc_progress = accuracy(pred_tensor, gt_tensor, topk=(1,))
    #
    #         return acc.tolist()
    #     return None

def train_one_epoch_on_saclassifier_wDETR(
        modelG: torch.nn.Module, modelH: torch.nn.Module,
        criterionG: torch.nn.Module, criterionH: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, max_norm: float = 0,
        path_to_input_debug: str = None, logger=None, frozen_modelG=False,
        dataset_id_to_filename=None):

    modelG.train()
    modelH.train()
    criterionG.train()
    criterionH.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    max_iter = len(data_loader)

    # len(data_loader) => how many times of batches (num_sample / num_batch)
    # len(data_loader.dataset) => how many samples
    for i_batch in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        if isinstance(samples, list):
            service_pred_logits = []
            for i_s, item in enumerate(samples):

                if frozen_modelG:
                    with torch.no_grad():
                        outputsG = modelG(item)
                else:
                    outputsG = modelG(item)

                # # get the duration from filename
                # x_duration = []
                # filename = dataset_id_to_filename(targets[i_s]['image_id'].item())[0]['file_name']
                # str_date = filename.split('.')[0].split('_')[1]
                # key, cur_time = get_data_from_string(str_date)
                # num_prev = item.tensors.shape[0]
                # for i_p in range(-num_prev+1, 1):
                #     duration_norm, _ = get_duration_key_time(key, cur_time+i_p)     # cur_time (sec.msec) + (-sec)
                #     x_duration.append(duration_norm)
                #
                # x_duration = torch.stack(x_duration, dim=0)
                # x_duration = x_duration.unsqueeze(dim=1)
                # x_duration = x_duration.cuda()
                # print('x_duration: ', x_duration)

                # get the duration from target
                # not normalized, just a second
                x_duration_t = targets[i_s]['duration_sec']  # n_b, n_T
                x_duration_t = x_duration_t.squeeze(dim=0)  # n_T
                x_duration_t = x_duration_t.unsqueeze(dim=1)
                x_duration_t = get_duration_norm(x_duration_t)

                # print('x_duration_t: ', x_duration_t)
                # pdb.set_trace()

                outputsG['duration'] = x_duration_t

                outputsH = modelH(outputsG)
                service_pred_logits.append(outputsH['service_pred_logits'])
            outputsH['service_pred_logits'] = torch.cat(service_pred_logits, dim=0)
        else:
            outputsG = modelG(samples)
            if dataset_id_to_filename is not None:
                # # get the duration from filename
                # x_duration = []
                # for item in targets:
                #     filename = dataset_id_to_filename(item['image_id'].item())[0]['file_name']
                #     str_date = filename.split('.')[0].split('_')[1]
                #     duration_norm = get_duration(str_date)
                #     x_duration.append(duration_norm)
                # x_duration = torch.stack(x_duration, dim=0)
                # x_duration = x_duration.unsqueeze(dim=1)
                # x_duration = x_duration.cuda()
                # print('x_duration: ', x_duration)

                # get the duration from target
                # not normalized, just a second
                x_duration_t = []
                for item in targets:
                    x_duration_t.append(item['duration_sec'])  # 1, 1
                x_duration_t = torch.cat(x_duration_t, dim=0)  # n_b, 1
                x_duration_t = get_duration_norm(x_duration_t)

                # print('x_duration_t: ', x_duration_t)
                # pdb.set_trace()

                outputsG['duration'] = x_duration_t
                # add also in evaluate
            outputsH = modelH(outputsG)

        # if i_batch > 30:
        #     pdb.set_trace()
        #     x1 = 0.2686
        #     y1 = 0.1563
        #     x2 = 0.6305
        #     y2 = 0.5233
        #     t_img = samples.tensors[-5, :, :, :]
        #     _, www, hhh = t_img.shape       # www: 1022, hhh: 975
        #
        #     xx1 = int(x1 * www)
        #     xx2 = int(x2 * www)
        #     yy1 = int(y1 * hhh)
        #     yy2 = int(y2 * hhh)
        #
        #     cropped_image = t_img[:, xx1:xx2, yy1:yy2]
        #     path_to_file = './crop.jpg'
        #     pil_image = tensor_to_pil(cropped_image.cpu().detach(), None)
        #     pil_image.save(path_to_file, 'JPEG')
        #
        #     path_to_file = './timg.jpg'
        #     pil_image = tensor_to_pil(t_img.cpu().detach(), None)
        #     pil_image.save(path_to_file, 'JPEG')


        if (path_to_input_debug is not None) and (i_batch < 2):
            path_to_file = os.path.join(path_to_input_debug,
                                        'e%d_i%d_id%d.jpg' % (epoch, i_batch,
                                                                     targets[0]['image_id'].item()))
            # samples.tensors[0]: [3, h, w]
            if isinstance(samples, list):
                draw_image = samples[-1].tensors[0].detach().cpu()
                draw_mask = samples[-1].mask[0].detach().cpu()
            else:
                draw_image = samples.tensors[0].detach().cpu()
                draw_mask = samples.mask[0].detach().cpu()
            scaled_boxes = targets[0]['boxes'].detach().cpu()  # n_box, 4

            masked_image, unscaled_boxes = convert_sample_and_boxes(draw_image, draw_mask, scaled_boxes)
            save_tensor_as_image(path_to_file, masked_image, unscaled_boxes,
                                 targets[0]['labels'].detach().cpu())

        if frozen_modelG:
            loss_dictH = criterionH(outputsH, targets)
            loss_dict = loss_dictH    # */iterable, **/dict unpacking operator
            weight_dict = criterionH.weight_dict
        else:
            loss_dictG = criterionG(outputsG, targets)
            loss_dictH = criterionH(outputsH, targets)
            loss_dict = {**loss_dictG, **loss_dictH}  # */iterable, **/dict unpacking operator
            weight_dict = {**criterionG.weight_dict, **criterionH.weight_dict}

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm_on_G = torch.nn.utils.clip_grad_norm_(modelG.parameters(), max_norm)
            grad_total_norm_on_H = torch.nn.utils.clip_grad_norm_(modelH.parameters(), max_norm)
        else:
            grad_total_norm_on_G = utils.get_total_grad_norm(modelG.parameters(), max_norm)
            grad_total_norm_on_H = utils.get_total_grad_norm(modelH.parameters(), max_norm)
        grad_total_norm = max(grad_total_norm_on_G, grad_total_norm_on_H)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        if logger is not None:
            info = {
                'loss': loss_value,
                'lr': optimizer.param_groups[0]["lr"],
            }
            logger.add_scalars("iter", info, epoch * max_iter + i_batch)

        samples, targets = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_on_saclassifier_wDETR(modelG, modelH, criterionG, criterionH, postprocessors,
                                   data_loader, base_ds, device, output_dir,
                                   save_result_image=False, num_saved_results=0, vis_th=0.5,
                                   num_classes=5, dataset_id_to_filename=None):
    with torch.no_grad():
        # get all evaluation results
        modelG.eval()
        modelH.eval()
        criterionG.eval()
        criterionH.eval()

        # postprocessors convert bbox type and scale
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Test:'

        sac_evaluator = SACEvaluator(num_classes)

        coco_evaluator = None
        if 'bbox' in postprocessors.keys():
            iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
            coco_evaluator = CocoEvaluator(base_ds, iou_types)
            # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

        if save_result_image:
            path_to_result_images = os.path.join(output_dir, 'vis')
            if not os.path.exists(path_to_result_images):
                os.makedirs(path_to_result_images)

        count_samples = 0
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # this can be multiple.
            targets = [{k: v.to(device) for k, v in t.items() if k != 'file_name'} for t in
                       targets]  # this can be multiple.

            # targets_filename = [t['file_name'] for t in targets]
            if isinstance(samples, list):
                service_pred_logits = []
                outputsG_pred_logits = []
                outputsG_pred_boxes = []
                outputsG_amount_score = []
                outputsG_progress_score = []
                for i_s, item in enumerate(samples):
                    item = item.to(device)
                    outputsG = modelG(item)

                    # # get the duration from filename
                    # x_duration = []
                    # filename = dataset_id_to_filename(targets[i_s]['image_id'].item())[0]['file_name']
                    # str_date = filename.split('.')[0].split('_')[1]
                    # key, cur_time = get_data_from_string(str_date)
                    # num_prev = item.tensors.shape[0]
                    #
                    # for i_p in range(-num_prev + 1, 1):
                    #     duration_norm, _ = get_duration_key_time(key, cur_time + i_p)
                    #     x_duration.append(duration_norm)
                    #
                    # x_duration = torch.stack(x_duration, dim=0) # n_T
                    # x_duration = x_duration.unsqueeze(dim=1)    # n_T, 1
                    # x_duration = x_duration.cuda()
                    # print('x_duration: ', x_duration)

                    # get the duration from target
                    # not normalized, just a second
                    x_duration_t = targets[i_s]['duration_sec']     # n_b, n_T
                    x_duration_t = x_duration_t.squeeze(dim=0)      # n_T
                    x_duration_t = x_duration_t.unsqueeze(dim=1)
                    x_duration_t = get_duration_norm(x_duration_t)
                    # print('x_duration_t: ', x_duration_t)

                    outputsG['duration'] = x_duration_t

                    outputsH = modelH(outputsG)

                    service_pred_logits.append(outputsH['service_pred_logits'])

                    outputsG_pred_logits.append(outputsG['pred_logits'][-1:, :, :])
                    outputsG_pred_boxes.append(outputsG['pred_boxes'][-1:, :, :])
                    outputsG_amount_score.append(outputsG['amount_score'][-1:, :, :])
                    outputsG_progress_score.append(outputsG['progress_score'][-1:, :])

                outputsH['service_pred_logits'] = torch.cat(service_pred_logits, dim=0)

                outputsG['pred_logits'] = torch.cat(outputsG_pred_logits, dim=0)
                outputsG['pred_boxes'] = torch.cat(outputsG_pred_boxes, dim=0)
                outputsG['amount_score'] = torch.cat(outputsG_amount_score, dim=0)
                outputsG['progress_score'] = torch.cat(outputsG_progress_score, dim=0)

                loss_dictH = criterionH(outputsH, targets)  # just loss for training
                loss_dict = loss_dictH  # */iterable, **/dict unpacking operator
                weight_dict = criterionH.weight_dict
            else:
                samples = samples.to(device)
                outputsG = modelG(samples)
                if dataset_id_to_filename is not None:
                    # # get the duration from filename
                    # x_duration = []
                    # for item in targets:
                    #     filename = dataset_id_to_filename(item['image_id'].item())[0]['file_name']
                    #     str_date = filename.split('.')[0].split('_')[1]
                    #     duration_norm = get_duration(str_date)
                    #     x_duration.append(duration_norm)
                    # x_duration = torch.stack(x_duration, dim=0) # n_b
                    # x_duration = x_duration.unsqueeze(dim=1)    # n_b, 1
                    # x_duration = x_duration.cuda()
                    # print('x_duration: ', x_duration)

                    # get the duration from target
                    # not normalized, just a second
                    x_duration_t = []
                    for item in targets:
                        x_duration_t.append(item['duration_sec']) # 1, 1
                    x_duration_t = torch.cat(x_duration_t, dim=0)   # n_b, 1
                    x_duration_t = get_duration_norm(x_duration_t)
                    # print('x_duration_t: ', x_duration_t)
                    # pdb.set_trace()

                    outputsG['duration'] = x_duration_t
                    # add also in evaluate
                outputsH = modelH(outputsG)

                loss_dictG = criterionG(outputsG, targets)
                loss_dictH = criterionH(outputsH, targets) # just loss for training

                loss_dict = {**loss_dictH, **loss_dictG}
                weight_dict = {**criterionH.weight_dict, **criterionG.weight_dict}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                 **loss_dict_reduced_scaled,
                                 **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

            if coco_evaluator is not None:
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = postprocessors['bbox'](outputsG, orig_target_sizes)

                res = {target['image_id'].item(): output for target, output in
                       zip(targets, results)}
                # res: dict ('image_id') of dict ('scores', 'labels', 'boxes', )

                coco_evaluator.update(res)

            if sac_evaluator is not None:
                outputs_for_saclass = outputsH['service_pred_logits']     # n_batch, n_class
                if len(outputs_for_saclass.shape) == 3:
                    outputs_for_saclass = torch.squeeze(outputs_for_saclass, dim=2)

                sac_evaluator.update(outputs_for_saclass, targets)  # insert score(logit), not prob

                # sac_evaluator.predictions
                # sac_evaluator.groundtruths
                # sac_evaluator.summarize()

            if save_result_image:
                for i, target in enumerate(targets):    # per batch
                    count_samples += 1
                    if count_samples <= num_saved_results:
                        image_id = target["image_id"].item()
                        file_name = f"{image_id:012d}.jpg"
                        # file_name = targets_filename[i]
                        save_file_name = os.path.join(path_to_result_images, file_name)

                        if isinstance(samples, list):
                            pdb.set_trace() # check -1 or 0, the key_frame
                            draw_image = samples[-1].tensors[i].detach().cpu()
                            draw_mask = samples[-1].mask[i].detach().cpu()

                            for key_weights in ['pca_weights_bbox', 'pca_weights_amount',
                                                'backbone_weights',
                                                'hs_output_weights', 'enc_output_weights']:
                                if outputsH[key_weights] is not None:
                                    outputsH[key_weights] = outputsH[key_weights][0:1, :, :]
                        else:
                            draw_image = samples.tensors[i].detach().cpu()
                            draw_mask = samples.mask[i].detach().cpu()

                        if coco_evaluator is not None:
                            scaled_boxes = res[image_id]['boxes'].detach().cpu()  # n_box, 4
                            res_labels = res[image_id]['labels'].detach().cpu()
                            res_scores = res[image_id]['scores'].detach().cpu()
                        else:
                            scaled_boxes = res_labels = res_scores = None

                        masked_image, unscaled_boxes = convert_sample_and_boxes(draw_image, draw_mask,
                                                                                scaled_boxes)

                        pil_image = tensor_to_pil(masked_image,
                                                  orig_size=(targets[i]['orig_size'][1].item(),
                                                             targets[i]['orig_size'][0].item()))
                        if coco_evaluator is not None:
                            # draw bboxes
                            pil_image = draw_bboxes_on_pil(pil_image, unscaled_boxes, res_labels,
                                                           scores=res_scores,
                                                           vis_th=vis_th)

                        if 'bbox_attn' in postprocessors.keys():
                            resultsG, resultsH = postprocessors['bbox_attn'](outputsG, outputsH, orig_target_sizes)

                            # resultsH['hs_attn_values']  # n_b or n_T, n_class-1, topk(3)
                            # resultsH['hs_attn_bbox']    # n_b or n_T, n_class-1, topk(3), 4

                            res_hs = {target['image_id'].item(): {'boxes': hs_bbox, 'scores': hs_scores} for target, hs_scores, hs_bbox in zip(targets, resultsH['hs_attn_values'], resultsH['hs_attn_bbox'])}
                            res_boxes_hs = res_hs[image_id]['boxes']
                            res_scores_hs = res_hs[image_id]['scores']

                            for i_c in range(res_boxes_hs.shape[0]):
                                pil_image = draw_bboxes_on_pil(pil_image, res_boxes_hs[i_c, :, :], [i_c] * 100,
                                                               scores=res_scores_hs[i_c, :],
                                                               vis_th=0.01)

                        # draw saclass predictions
                        unscaled_boxes = torch.tensor([[1, 1, 10, 10], [1, 21, 10, 30],
                                                       [1, 41, 10, 50], [1, 61, 10, 70]])
                        res_labels = ['refill', 'trash', 'dessert', 'lost']
                        saclass_logits = outputsH['service_pred_logits']
                        res_scores = torch.sigmoid(saclass_logits)[0, 1:].detach().cpu()
                        vis_th = 0.0
                        pil_image = draw_bboxes_on_pil(pil_image, unscaled_boxes, res_labels,
                                                       scores=res_scores,
                                                       vis_th=vis_th)

                        pil_image.save(save_file_name, 'JPEG')


        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

        sac_res = None
        if sac_evaluator is not None:
            sac_evaluator.synchronize_between_processes()   # necessary for multi GPUs
            sac_res = sac_evaluator.summarize()
            print(sac_res)
            # sac_evaluator.predictions

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if coco_evaluator is not None:
            if 'bbox' in postprocessors.keys():
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in postprocessors.keys():
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if sac_res is not None:
            stats.update(sac_res)

    return stats, coco_evaluator



@torch.no_grad()
def evaluate_on_saclassifier_wDETR_multiInput(modelG, modelH, criterionG, criterionH, postprocessors,
                                   data_loader, data_loader2, base_ds, device, output_dir,
                                   save_result_image=False, num_saved_results=0, vis_th=0.5,
                                   num_classes=5, dataset_id_to_filename=None, dataset_id_to_filename2=None):
    with torch.no_grad():
        # get all evaluation results
        modelG.eval()
        modelH.eval()
        criterionG.eval()
        criterionH.eval()

        # postprocessors convert bbox type and scale
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Test:'

        sac_evaluator = SACEvaluator(num_classes)

        coco_evaluator = None
        if 'bbox' in postprocessors.keys():
            iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
            coco_evaluator = CocoEvaluator(base_ds, iou_types)
            # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

        if save_result_image:
            path_to_result_images = os.path.join(output_dir, 'vis')
            if not os.path.exists(path_to_result_images):
                os.makedirs(path_to_result_images)




        prefetcher = data_prefetcher(data_loader, device, prefetch=True)
        samples, targets = prefetcher.next()

        prefetcher2 = data_prefetcher(data_loader2, device, prefetch=True)
        samples2, targets2 = prefetcher2.next()

        count_samples = 0
        for i_batch in metric_logger.log_every(range(len(data_loader)), 10, header):
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # this can be multiple.
            targets = [{k: v.to(device) for k, v in t.items() if k != 'file_name'} for t in targets]  # this can be multiple.
            targets2 = [{k: v.to(device) for k, v in t.items() if k != 'file_name'} for t in targets2]  # this can be multiple.

            filename = dataset_id_to_filename(targets[0]['image_id'].item())[0]['file_name']
            filename2 = dataset_id_to_filename2(targets2[0]['image_id'].item())[0]['file_name']

            assert filename.replace('captured1', 'captured2') == filename2

            # merge netested tensor
            #samples = samples1 + samples2
            # .tensors, .mask
            if isinstance(samples, list):
                for i_s in range(len(samples)):
                    samples[i_s].tensors = torch.cat([samples[i_s].tensors, samples2[i_s].tensors], dim=3)
                    samples[i_s].mask = torch.cat([samples[i_s].mask, samples2[i_s].mask], dim=2)
            else:
                samples.tensors = torch.cat([samples.tensors, samples2.tensors], dim=3)
                samples.mask = torch.cat([samples.mask, samples2.mask], dim=2)

            if isinstance(samples, list):
                service_pred_logits = []
                outputsG_pred_logits = []
                outputsG_pred_boxes = []
                outputsG_amount_score = []
                outputsG_progress_score = []
                for i_s, item in enumerate(samples):
                    item = item.to(device)
                    outputsG = modelG(item)
                    
                    # # get the duration from filename
                    # x_duration = []
                    # filename = dataset_id_to_filename(targets[i_s]['image_id'].item())[0]['file_name']
                    # str_date = filename.split('.')[0].split('_')[1]
                    # key, cur_time = get_data_from_string(str_date)
                    # num_prev = item.tensors.shape[0]
                    #
                    # for i_p in range(-num_prev + 1, 1):
                    #     duration_norm, _ = get_duration_key_time(key, cur_time + i_p)
                    #     x_duration.append(duration_norm)
                    #
                    # x_duration = torch.stack(x_duration, dim=0) # n_T
                    # x_duration = x_duration.unsqueeze(dim=1)    # n_T, 1
                    # x_duration = x_duration.cuda()
                    # print('x_duration: ', x_duration)

                    # get the duration from target
                    # not normalized, just a second
                    x_duration_t = targets[i_s]['duration_sec']     # n_b, n_T
                    x_duration_t = x_duration_t.squeeze(dim=0)      # n_T
                    x_duration_t = x_duration_t.unsqueeze(dim=1)    # n_T, 1
                    x_duration_t = get_duration_norm(x_duration_t)
                    # print('x_duration_t: ', x_duration_t)

                    outputsG['duration'] = x_duration_t

                    outputsH = modelH(outputsG)

                    service_pred_logits.append(outputsH['service_pred_logits'])

                    outputsG_pred_logits.append(outputsG['pred_logits'][-1:, :, :])
                    outputsG_pred_boxes.append(outputsG['pred_boxes'][-1:, :, :])
                    outputsG_amount_score.append(outputsG['amount_score'][-1:, :, :])
                    outputsG_progress_score.append(outputsG['progress_score'][-1:, :])

                outputsH['service_pred_logits'] = torch.cat(service_pred_logits, dim=0)

                outputsG['pred_logits'] = torch.cat(outputsG_pred_logits, dim=0)
                outputsG['pred_boxes'] = torch.cat(outputsG_pred_boxes, dim=0)
                outputsG['amount_score'] = torch.cat(outputsG_amount_score, dim=0)
                outputsG['progress_score'] = torch.cat(outputsG_progress_score, dim=0)

                loss_dictH = criterionH(outputsH, targets)  # just loss for training
                loss_dict = loss_dictH  # */iterable, **/dict unpacking operator
                weight_dict = criterionH.weight_dict
            else:
                samples = samples.to(device)
                outputsG = modelG(samples)
                if dataset_id_to_filename is not None:
                    # # get the duration from filename
                    # x_duration = []
                    # for item in targets:
                    #     filename = dataset_id_to_filename(item['image_id'].item())[0]['file_name']
                    #     str_date = filename.split('.')[0].split('_')[1]
                    #     duration_norm = get_duration(str_date)
                    #     x_duration.append(duration_norm)
                    # x_duration = torch.stack(x_duration, dim=0) # n_b
                    # x_duration = x_duration.unsqueeze(dim=1)    # n_b, 1
                    # x_duration = x_duration.cuda()
                    # print('x_duration: ', x_duration)

                    # get the duration from target
                    # not normalized, just a second
                    x_duration_t = []
                    for item in targets:
                        x_duration_t.append(item['duration_sec']) # 1, 1
                    x_duration_t = torch.cat(x_duration_t, dim=0)   # n_b, 1
                    x_duration_t = get_duration_norm(x_duration_t)
                    # print('x_duration_t: ', x_duration_t)
                    # pdb.set_trace()

                    outputsG['duration'] = x_duration_t
                    # add also in evaluate
                outputsH = modelH(outputsG)

                loss_dictG = criterionG(outputsG, targets)
                loss_dictH = criterionH(outputsH, targets) # just loss for training

                loss_dict = {**loss_dictH, **loss_dictG}
                weight_dict = {**criterionH.weight_dict, **criterionG.weight_dict}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                 **loss_dict_reduced_scaled,
                                 **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

            if coco_evaluator is not None:
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = postprocessors['bbox'](outputsG, orig_target_sizes)

                res = {target['image_id'].item(): output for target, output in
                       zip(targets, results)}
                # res: dict ('image_id') of dict ('scores', 'labels', 'boxes', )

                coco_evaluator.update(res)

            if sac_evaluator is not None:
                outputs_for_saclass = outputsH['service_pred_logits']     # n_batch, n_class
                if len(outputs_for_saclass.shape) == 3:
                    outputs_for_saclass = torch.squeeze(outputs_for_saclass, dim=2)

                sac_evaluator.update(outputs_for_saclass, targets)  # insert score(logit), not prob

                # sac_evaluator.predictions
                # sac_evaluator.groundtruths
                # sac_evaluator.summarize()

            if save_result_image:
                for i, target in enumerate(targets):    # per batch
                    count_samples += 1
                    if count_samples <= num_saved_results:
                        image_id = target["image_id"].item()
                        file_name = f"{image_id:012d}.jpg"
                        # file_name = targets_filename[i]
                        save_file_name = os.path.join(path_to_result_images, file_name)

                        if isinstance(samples, list):
                            pdb.set_trace() # check -1 or 0, the key_frame
                            draw_image = samples[-1].tensors[i].detach().cpu()
                            draw_mask = samples[-1].mask[i].detach().cpu()

                            for key_weights in ['pca_weights_bbox', 'pca_weights_amount',
                                                'backbone_weights',
                                                'hs_output_weights', 'enc_output_weights']:
                                if outputsH[key_weights] is not None:
                                    outputsH[key_weights] = outputsH[key_weights][0:1, :, :]
                        else:
                            draw_image = samples.tensors[i].detach().cpu()
                            draw_mask = samples.mask[i].detach().cpu()

                        if coco_evaluator is not None:
                            scaled_boxes = res[image_id]['boxes'].detach().cpu()  # n_box, 4
                            res_labels = res[image_id]['labels'].detach().cpu()
                            res_scores = res[image_id]['scores'].detach().cpu()
                        else:
                            scaled_boxes = res_labels = res_scores = None

                        masked_image, unscaled_boxes = convert_sample_and_boxes(draw_image, draw_mask,
                                                                                scaled_boxes)

                        pil_image = tensor_to_pil(masked_image,
                                                  orig_size=(targets[i]['orig_size'][1].item(),
                                                             targets[i]['orig_size'][0].item()))
                        if coco_evaluator is not None:
                            # draw bboxes
                            pil_image = draw_bboxes_on_pil(pil_image, unscaled_boxes, res_labels,
                                                           scores=res_scores,
                                                           vis_th=vis_th)

                        if 'bbox_attn' in postprocessors.keys():
                            resultsG, resultsH = postprocessors['bbox_attn'](outputsG, outputsH, orig_target_sizes)

                            # resultsH['hs_attn_values']  # n_b or n_T, n_class-1, topk(3)
                            # resultsH['hs_attn_bbox']    # n_b or n_T, n_class-1, topk(3), 4

                            res_hs = {target['image_id'].item(): {'boxes': hs_bbox, 'scores': hs_scores} for target, hs_scores, hs_bbox in zip(targets, resultsH['hs_attn_values'], resultsH['hs_attn_bbox'])}
                            res_boxes_hs = res_hs[image_id]['boxes']
                            res_scores_hs = res_hs[image_id]['scores']

                            for i_c in range(res_boxes_hs.shape[0]):
                                pil_image = draw_bboxes_on_pil(pil_image, res_boxes_hs[i_c, :, :], [i_c] * 100,
                                                               scores=res_scores_hs[i_c, :],
                                                               vis_th=0.01)

                        # draw saclass predictions
                        unscaled_boxes = torch.tensor([[1, 1, 10, 10], [1, 21, 10, 30],
                                                       [1, 41, 10, 50], [1, 61, 10, 70]])
                        res_labels = ['refill', 'trash', 'dessert', 'lost']
                        saclass_logits = outputsH['service_pred_logits']
                        res_scores = torch.sigmoid(saclass_logits)[0, 1:].detach().cpu()
                        vis_th = 0.0
                        pil_image = draw_bboxes_on_pil(pil_image, unscaled_boxes, res_labels,
                                                       scores=res_scores,
                                                       vis_th=vis_th)

                        pil_image.save(save_file_name, 'JPEG')

            samples, targets = prefetcher.next()
            samples2, targets2 = prefetcher2.next()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

        sac_res = None
        if sac_evaluator is not None:
            sac_res = sac_evaluator.summarize()
            print(sac_res)
            # sac_evaluator.predictions

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if coco_evaluator is not None:
            if 'bbox' in postprocessors.keys():
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in postprocessors.keys():
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if sac_res is not None:
            stats.update(sac_res)

    return stats, coco_evaluator