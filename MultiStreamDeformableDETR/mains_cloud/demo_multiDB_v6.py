# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# copyed from https://github.com/ahmed-nady/Deformable-DETR/blob/main/demo_imgLst.py

import argparse
import datetime
import json
import pdb
import random
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T

import numpy as np
import torch
import util.misc as utils
# from util import box_ops
from models import build_multioutput_multidataset_model_multitrfmModule as build_model
import torch.nn.functional as F
import time
import os
from torchvision.ops import nms
# from demo_postproc import postprocessor, det_to_trk
import pickle

# import xml.etree.cElementTree as ET
from models.service_detector import build_SAclassifier
from my_debug import draw_bboxes_on_pil, images_to_video
from engine_saclassifier import get_duration

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


# v2. add nms for merging outputs
# v3. apply postproc
# v4. apply point-based matching (v3 used ROI-based matching)
# v5. predict the service proposals
# v6. remove trackers (no more used, heavy in real-time application)

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'],
                        type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    # parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')	# no more used
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--frozen_backbone_weights', action='store_true')
    parser.add_argument('--frozen_backbone_detr_weights_wo_heads', action='store_true')
    parser.add_argument('--frozen_backbone_detr_weights', action='store_true')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int,
                        help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--amount_loss_coef', default=1, type=float)
    parser.add_argument('--progress_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--TrainSetFilename', default='')
    parser.add_argument('--TrainTargetSetFilename', default=None, type=str, nargs='+')
    parser.add_argument('--ValSetFilename', default='')

    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true',
                        help='whether to cache images on memory')

    parser.add_argument('--use_amount', action='store_true')
    parser.add_argument('--amount_type', default=0, type=int)
    parser.add_argument('--n_amount_class', default=50, type=int)
    parser.add_argument('--use_progress', action='store_true')

    parser.add_argument('--num_classes_on_G', default=17, type=int)
    parser.add_argument('--num_classes_on_H', default=5, type=int)
    parser.add_argument('--num_trfms', default=3, type=int)

    parser.add_argument('--list_index_amount_trfm', default=[], type=int,
                        nargs='*')  # + >=1, * >= 0
    parser.add_argument('--list_index_progress_trfm', default=[], type=int,
                        nargs='*')  # + >=1, * >= 0

    parser.add_argument('--save_result_image', action='store_true')
    parser.add_argument('--num_saved_results', default=20, type=int)
    parser.add_argument('--use_tfboard', action='store_true')

    parser.add_argument('--max_iter_limit', default=1e10, type=int)
    parser.add_argument('--saclassifier_type', default='imagebased', type=str)

    parser.add_argument('--init_heads_in_final', action='store_true')

    # sequence input
    parser.add_argument('--num_prev_imgs', default=0, type=int)

    # for demo
    parser.add_argument('--class_list', default='coco')
    parser.add_argument('--sac_class_list', default='coco')
    parser.add_argument('--imgs_dir', type=str, help='input images folder for inference')
    parser.add_argument('--vis_th', type=float, default=0.7)
    parser.add_argument('--crop_ratio_ROI', nargs='+', type=float)
    # parser.add_argument('--process_per_n_image', default=1, type=int)
    parser.add_argument('--process_n_per_min', default=60, type=int)
    parser.add_argument('--skip_first_n_image', default=0, type=int)
    parser.add_argument('--display_class_names', default=[], type=str, nargs='*')

    return parser


# standard PyTorch mean-std input image normalization
transform_resize = T.Compose([
    # T.Resize(1000)
    # T.Resize(1376)
    T.Resize(800)
])

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#
# def alarm_to_object(alarm_list, im_w, im_h):
#     object_list = []
#     for key, values in alarm_list.items():
#         for item in values:
#             x1 = int(item['bboxes'][-1][0])
#             y1 = int(item['bboxes'][-1][1])
#             x2 = int(item['bboxes'][-1][2])
#             y2 = int(item['bboxes'][-1][3])
#
#             if x1 < 0: x1 = 0
#             if y1 < 0: y1 = 0
#             if x2 > im_w - 1: x2 = im_w - 1
#             if y2 > im_h - 1: y2 = im_h - 1
#
#             obj = {
#                 'name': key,
#                 'bndbox': [x1, y1, x2, y2],
#                 'track_id': item['track_id']
#             }
#             object_list.append(obj)
#
#     return object_list

#
# def save_xml_pascal(filename_alm, im_w, im_h, original_filename, object_list):
#     root = ET.Element('annotation')
#
#     ET.SubElement(root, 'folder').text = 'captured'
#     ET.SubElement(root, 'filename').text = original_filename
#
#     t_src = ET.SubElement(root, 'source')
#     ET.SubElement(t_src, 'database').text = 'Unknown'
#     ET.SubElement(t_src, 'annotation').text = 'Unknown'
#     ET.SubElement(t_src, 'image').text = 'Unknown'
#
#     t_size = ET.SubElement(root, 'size')
#     ET.SubElement(t_size, 'width').text = str(im_w)
#     ET.SubElement(t_size, 'height').text = str(im_h)
#     ET.SubElement(t_size, 'depth').text = str(3)
#
#     ET.SubElement(root, 'segmented').text = '0'
#
#     for obj in object_list:
#         t_obj = ET.SubElement(root, 'object')
#         ET.SubElement(t_obj, 'name').text = obj['name']
#         # ET.SubElement(t_obj, 'category').text = str()
#
#         ET.SubElement(t_obj, 'truncated').text = '0'
#         ET.SubElement(t_obj, 'occluded').text = '0'
#         ET.SubElement(t_obj, 'difficult').text = '0'
#         ET.SubElement(t_obj, 'pose').text = 'unknown'  # 'Unspecified'
#
#         bbox = obj['bndbox']
#         t_bndbox = ET.SubElement(t_obj, 'bndbox')
#         ET.SubElement(t_bndbox, 'xmin').text = str(bbox[0])
#         ET.SubElement(t_bndbox, 'ymin').text = str(bbox[1])
#         ET.SubElement(t_bndbox, 'xmax').text = str(bbox[2])
#         ET.SubElement(t_bndbox, 'ymax').text = str(bbox[3])
#
#         t_attr = ET.SubElement(t_obj, 'attributes')
#         t_attr1 = ET.SubElement(t_attr, 'attribute')
#         ET.SubElement(t_attr1, 'name').text = 'rotation'
#         ET.SubElement(t_attr1, 'value').text = '0.0'
#
#         t_attr2 = ET.SubElement(t_attr, 'attribute')
#         ET.SubElement(t_attr2, 'name').text = 'track_id'
#         ET.SubElement(t_attr2, 'value').text = str(obj['track_id'])
#
#         t_attr3 = ET.SubElement(t_attr, 'attribute')
#         ET.SubElement(t_attr3, 'name').text = 'keyframe'
#         ET.SubElement(t_attr3, 'value').text = 'True'
#
#     tree = ET.ElementTree(root)
#     tree.write(filename_alm)


def main(args, imgs_dir=None, output_dir=None):
    # args.output_dir = os.path.splitext(args.resume)[0]

    if imgs_dir is not None:
        args.imgs_dir = imgs_dir

    if output_dir is not None:
        args.output_dir = output_dir

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        args.output_dir = './dummy'
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    output_alarm_dir = os.path.join(args.output_dir, 'alarms')
    output_alarm_anno_dir = os.path.join(output_alarm_dir, 'Annotations')
    output_alarm_imgs_dir = os.path.join(output_alarm_dir, 'Images')

    Path(output_alarm_dir).mkdir(parents=True, exist_ok=True)
    Path(output_alarm_anno_dir).mkdir(parents=True, exist_ok=True)
    Path(output_alarm_imgs_dir).mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    print('Called with args:')
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # F = H( G( x )) )
    modelG, _, postprocessorG = build_model(args, num_classes=args.num_classes_on_G,
                                            num_trfms=args.num_trfms)
    modelH, _, postprocessorH = build_SAclassifier(args.num_classes_on_H, args.num_trfms,
                                                   saclassifier_type=args.saclassifier_type)
    postprocessors = {**postprocessorG, **postprocessorH}

    modelG.to(device)
    modelH.to(device)

    print('Load from resume_on_G_and_H')
    checkpoint = torch.load(args.resume, map_location='cpu')

    missing_keys_in_G, unexpected_keys_in_G = modelG.load_state_dict(checkpoint['modelG'],
                                                                     strict=False)
    missing_keys_in_H, unexpected_keys_in_H = modelH.load_state_dict(checkpoint['modelH'],
                                                                     strict=False)

    unexpected_keys_in_G = [k for k in unexpected_keys_in_G if
                            not (k.endswith('total_params') or k.endswith('total_ops'))]
    unexpected_keys_in_H = [k for k in unexpected_keys_in_H if
                            not (k.endswith('total_params') or k.endswith('total_ops'))]

    if len(missing_keys_in_G) > 0:
        print('Missing Keys: {}'.format(missing_keys_in_G))
    if len(unexpected_keys_in_G) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys_in_G))

    if len(missing_keys_in_H) > 0:
        print('Missing Keys: {}'.format(missing_keys_in_H))
    if len(unexpected_keys_in_H) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys_in_H))

    if torch.cuda.is_available():
        modelG.cuda()
        modelH.cuda()
    modelG.eval()
    modelH.eval()

    with open(args.class_list) as f:
        classes = f.readlines()
        classes = [line.rstrip('\n') for line in classes]
    print(classes)

    with open(args.sac_class_list) as f:
        sac_classes = f.readlines()
        sac_classes = [line.rstrip('\n') for line in sac_classes]
    print(sac_classes)

    # dataset_val = build_dataset(image_set='val', args=args)
    # base_ds = get_coco_api_from_dataset(dataset_val)

    list_files = sorted(os.listdir(args.imgs_dir))

    number_of_colors = len(classes)
    list_colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                   for i in range(number_of_colors)]

    # filename_list = os.path.join(output_alarm_dir, 'output_list.txt')
    # fid = open(filename_list, 'w')

    with torch.no_grad():
        last_cap_sec = 0
        duration_sec = 60.0 / float(args.process_n_per_min)
        for i_th, img_file in enumerate(list_files[args.skip_first_n_image:]):

            # get captured time
            im_date = img_file[:-4].split('_')[-1].split('-')
            if len(im_date) == 6:
                im_year, im_mon, im_day, im_hrs, im_min, im_sec = im_date
                im_sec = int(im_sec)
            else:
                im_year, im_mon, im_day, im_hrs, im_min, im_sec, im_msec = im_date
                im_sec = float(im_sec + '.' + im_msec)
            im_hrs = int(im_hrs)
            im_min = int(im_min)

            cur_cap_sec = 3600 * im_hrs + 60 * im_min + im_sec

            if cur_cap_sec - last_cap_sec > duration_sec:
                last_cap_sec = cur_cap_sec
                # if i_th % args.process_per_n_image == 0:
                t0 = time.time()
                img_path = os.path.join(args.imgs_dir, img_file)
                filename_omg = os.path.join(args.output_dir,
                                            'out_' + img_file[:-4] + '.jpg')
                filename_det = os.path.join(args.output_dir,
                                            'out_' + img_file[:-4] + '.pkl')
                filename_img = os.path.join(output_alarm_imgs_dir,
                                            img_file)
                filename_alm = os.path.join(output_alarm_anno_dir,
                                            img_file[:-4] + '.xml')

                im_org = Image.open(img_path)

                im_w, im_h = im_org.size
                print(img_path, im_h, im_w)

                if args.crop_ratio_ROI is not None:
                    im_w, im_h = im_org.size

                    im_org = im_org.crop((int(im_w * args.crop_ratio_ROI[0]),
                                          int(im_h * args.crop_ratio_ROI[1]),
                                          int(im_w * args.crop_ratio_ROI[2]),
                                          int(im_h * args.crop_ratio_ROI[3])))

                # mean-std normalize the input image (batch-size: 1)
                im = transform_resize(im_org)
                img = transform(im).unsqueeze(0)

                img = img.cuda()
                # propagate through the model
                outputsG = modelG(img)

                if True:
                    # dataset_train.coco.loadImgs(self.ids[idx])[0]['file_name']
                    x_duration = []

                    str_date = img_file.split('.')[0].split('_')[1]
                    duration_norm = get_duration(str_date)
                    x_duration.append(duration_norm)
                    x_duration = torch.stack(x_duration, dim=0)
                    x_duration = x_duration.unsqueeze(dim=1)
                    x_duration = x_duration.cuda()
                    outputsG['duration'] = x_duration
                    # add also in evaluate

                outputsH = modelH(outputsG)
                outputs = {**outputsG, **outputsH}

                im_w, im_h = im.size
                target_sizes = torch.tensor([[im_h, im_w]])
                target_sizes = target_sizes.cuda()

                if 'bbox_attn' in postprocessors.keys():
                    resultsG, resultsH = postprocessors['bbox_attn'](outputsG, outputsH,
                                                                     target_sizes)

                # out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
                # # out_logits: [n_batch(1), n_query(300 x n_trfm), n_class+1]
                # # out_bbox: [n_batch(1), n_query(300 x n_trfm), 4]
                # prob = out_logits.sigmoid()  # [n_batch(1), n_query, n_class+1]
                # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)  # just pick the top-k scores and indexes
                # scores = topk_values
                # topk_boxes = topk_indexes // out_logits.shape[2]  # share -> n_query_index
                # labels = topk_indexes % out_logits.shape[2]  # remain -> n_class_index (reordered by topk_boxes)
                # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
                # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))    # select topk_boxes
                #
                # # and from relative [0, 1] to absolute [0, height] coordinates
                # img_h, img_w = target_sizes.unbind(1)
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                # boxes = boxes * scale_fct[:, None, :]

                scores = resultsG[0]['scores']  # [100]
                labels = resultsG[0]['labels']
                boxes = resultsG[0]['boxes']

                keep = scores > args.vis_th
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]  # topk & greater than args.vis_th

                num_topk_vis_th = sum(keep).item()
                print(f'{num_topk_vis_th} bboxes are selected > {args.vis_th}')

                if args.use_amount:
                    assert 'amount_score' in resultsH.keys()  # 'amount_score' must be in resultsH
                    amount_score = resultsH['amount_score'][0]  # [100, 50]
                    res_amount_logits = amount_score[keep]  # [keep, 50]

                    if True:
                        # fill zero to amount_score at not bbox with 'food(1)' and 'drink(2)'
                        amount_score[(labels > 2).nonzero(as_tuple=True)] = 0.0  # this cannot change value.
                        # amount_score.index_fill_(1, (labels > 2).nonzero(as_tuple=True)[1], 0.)

                    # [1, n_q, n_amount] -> [100, n_amount] -> [n_keep, n_amount]
                    res_amount_prob = F.softmax(res_amount_logits, dim=1)  # [n_keep, n_amount]
                    amount_prob, res_amount = torch.topk(res_amount_prob, k=3,
                                                         dim=1)  # [n_q, k], topk_prob, topk_class_index
                    n_div = 100 // res_amount_logits.size(1)
                    amount_pred = ((res_amount.type(torch.float32) * n_div) + n_div / 2) * 0.01
                    # amount_prob: topk_prob  [100, topk]
                    # amount_pred: topk_class [100, topk]
                    # res_amount: 0, n_div = 2 -> amount_pred = 1
                    # res_amount: 1, n_div = 2 -> amount_pred = 3
                    # ...
                    # res_amount: 49, n_div = 2 -> amount_pred = 99

                    res_table = ((torch.arange(0, res_amount_logits.size(
                        1)) * n_div) + n_div / 2) * 0.01
                    res_table = res_table.unsqueeze(0)
                    res_table = res_table.expand(res_amount_prob.size(0), -1)  # n_q, n_class
                    amount_pred_weighted = torch.sum(res_amount_prob * res_table.cuda(), dim=1)
                    # amount_pred_weighted: top1_class (computed by prob_weighted_sum)

                    # remove 'amount' except for 'food' and 'drink' should be all same values,
                    # amount_pred[(labels > 2).nonzero(as_tuple=True)]
                    # amount_pred_weighted[(labels > 2).nonzero(as_tuple=True)]

                if args.use_progress:
                    out_progress_score = outputs['progress_score']
                    progress_prob = F.softmax(out_progress_score[0, :], dim=0)
                print('processing time: ', time.time() - t0)

                # labels: [N]
                # scores: [N]
                # boxes: [1, N, 4]

                filtered_labels = []
                filtered_scores = []
                filtered_boxes = []
                filtered_amount_pred = []
                filtered_amount_pred_weighted = []

                for j_class in range(0, args.num_classes_on_G):
                    inds = (labels == j_class).nonzero(as_tuple=True)[0].view(-1)
                    # inds = torch.nonzero(prob[0, :, j_class] > 0.05).view(-1) # ignore this line because already selected top-N scores

                    if inds.numel() > 0:
                        cls_labels = labels[inds]
                        cls_scores = scores[inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        cls_boxes = boxes[inds, :]
                        cls_amount_pred = amount_pred[inds, :]
                        cls_amount_pred_w = amount_pred_weighted[inds]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        cls_dets = cls_dets[order]
                        cls_labels = cls_labels[order]
                        cls_scores = cls_scores[order]
                        cls_boxes = cls_boxes[order, :]
                        cls_amount_pred = cls_amount_pred[order, :]
                        cls_amount_pred_w = cls_amount_pred_w[order]
                        # Overlap threshold used for non-maximum suppression (suppress boxes with
                        # IoU >= this threshold)
                        keep = nms(cls_boxes, cls_scores, 0.3)  #

                        # cls_dets = cls_dets[keep.view(-1).long()]

                        filtered_labels.append(cls_labels[keep.view(-1).long()])
                        filtered_scores.append(cls_scores[keep.view(-1).long()])
                        filtered_boxes.append(cls_boxes[keep.view(-1).long()])
                        filtered_amount_pred.append(cls_amount_pred[keep.view(-1).long()])
                        filtered_amount_pred_weighted.append(
                            cls_amount_pred_w[keep.view(-1).long()])

                labels = torch.cat(filtered_labels, dim=0)  # [n_bboxes]
                scores = torch.cat(filtered_scores, dim=0)
                boxes = torch.cat(filtered_boxes, dim=0)
                amount_pred_weighted = torch.cat(filtered_amount_pred_weighted, dim=0)  # [n_bboxes]

                source_img = im.convert("RGB")  # im == cropped image in PIL.Image type
                draw = ImageDraw.Draw(source_img)

                # font = ImageFont.truetype('arial.ttf', 15)

                # print("Boxes: ", boxes.tolist())
                for ith, (label, score, (xmin, ymin, xmax, ymax), amount_pred_weighted) in enumerate(
                        zip(labels.tolist(), scores.tolist(), boxes.tolist(),
                            amount_pred_weighted.tolist())):

                    label_index = label - 1  # background is in label, but not in label_index

                    class_name = classes[label_index]

                    # label_index = classes.index(class_name)

                    if class_name in ['food', 'drink', 'dish', 'cup']:
                        pass
                    else:
                        amount_pred_weighted = None

                    if args.use_amount and amount_pred_weighted is not None:
                        str_result = '%s (%.1f) \nam: %.0f' % (class_name, score,
                                                               amount_pred_weighted * 100)
                        if class_name in ['dish', 'cup']:
                            draw.rectangle([xmin, ymin - 10,
                                            xmin + (xmax - xmin) * amount_pred_weighted, ymin - 5],
                                           fill=list_colors[label_index], width=2)
                    else:
                        str_result = '%s (%.1f)' % (class_name, score)

                    draw.multiline_text((xmin, ymin), str_result, fill=list_colors[label_index],
                                        spacing=2)

                    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=list_colors[label_index],
                                   width=2)

                # draw only one in one image
                if args.use_progress:
                    # draw.text((10, 10), '%d' % topk_index_progress.item())
                    draw.text((10, 10), 'progress: %.1f, %.1f, %.1f' %
                              (progress_prob[0], progress_prob[1], progress_prob[2]))

                # print(alarms)
                str_alarms = []

                # draw saclass predictions
                unscaled_boxes = torch.tensor([[10, 1 + 20, 10, 21], [10, 21 + 20, 10, 41],
                                               [10, 41 + 20, 10, 61], [10, 61 + 20, 10, 81]])
                res_scores = torch.sigmoid(outputsH['service_pred_logits'])[0, 1:].detach().cpu()
                source_img = draw_bboxes_on_pil(source_img, unscaled_boxes, sac_classes,
                                                scores=res_scores,
                                                vis_th=0.0, no_bbox=True)

                draw.multiline_text((10, 25), '\n'.join(str_alarms), fill='white', spacing=2)

                # draw the attention
                if 'bbox_attn' in postprocessors.keys():
                    # resultsH['hs_attn_values']  # n_b, n_class-1, topk(3)
                    # resultsH['hs_attn_bbox']    # n_b, n_class-1, topk(3), 4

                    list_attn = [
                        ['hs_attn_values', 'hs_attn_bbox'],
                        ['enc_attn_values', 'enc_attn_bbox'],
                        ['pca_bbox_attn_values', 'pca_bbox_attn_bbox'],
                        ['pca_amount_attn_values', 'pca_amount_attn_bbox'],
                    ]

                    for key_attn_value, key_attn_bbox in list_attn:
                        if key_attn_value in resultsH:
                            attn_value = resultsH[key_attn_value][0]  # [4, 3] = [n_class,
                            attn_bbox = resultsH[key_attn_bbox][0]  # [4, 3, 4]

                            for i_c in range(attn_bbox.shape[0]):
                                if res_scores[i_c] >= 0.5:
                                    c_attn_bbox = attn_bbox[i_c, :, :]
                                    c_attn_val = attn_value[i_c, :]
                                    # keep = nms(c_attn_bbox, c_attn_val, 0.3)  #
                                    # c_attn_bbox = c_attn_bbox[keep]
                                    # c_attn_val = c_attn_val[keep]
                                    source_img = draw_bboxes_on_pil(source_img,
                                                                    c_attn_bbox,
                                                                    [i_c] * 100,
                                                                    scores=c_attn_val,
                                                                    vis_th=0.01)

                source_img.save(filename_omg)
                print('result image is saved in ', filename_omg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    # main(args)

    # add script's argument at here
    home_dir = '/home/yochin/Desktop/Deformable-DETR'
    task_name = 'mains_cloud_datsets'
    dataset_name = 'ETRIGJHall'
    project_name = 'Imageavgp_pca1dlcnmsSoftmaxAttnSimple_frzDETR_wDet21c_v3'
    cmd_args = [
        '--resume', f'{home_dir}/{task_name}/exps/{dataset_name}/{project_name}/checkpoint.pth',
        '--backbone', 'resnet50',
        '--class_list', f'{home_dir}/datasets/YMTestBed20class/YMTestBed20class_label.txt',
        '--sac_class_list', f'{home_dir}/datasets/ETRIGJHall/ETRIGJHall_label.txt',

        '--backbone', 'resnet50',

        '--use_progress',
        '--use_amount',
        '--amount_type', '6',
        '--list_index_amount_trfm', '0', '1', '2',
        '--list_index_progress_trfm', '0',

        '--n_amount_class', '50',
        '--num_trfms', '4',  # <<<<

        '--with_box_refine',
        '--two_stage',

        '--num_classes_on_G', '21',
        '--num_classes_on_H', '5',
        '--saclassifier_type', 'imageavgp_pca1dlcnmsattnsimple',  # 'roiv2attnsimple', # 'roibased', #

        # '--crop_ratio_ROI', '0.25', '0', '0.85', '1',
        # '--crop_ratio_ROI', '0.125', '0.5', '0.875', '1.0',

        # '--process_per_n_image', '10',
        # '--skip_first_n_image', '0',

        '--vis_th', '0.7',
        '--eval',

        '--display_class_names', 'food', 'drink', 'dish', 'cup', 'empty_container', 'mobile_phone', 'wallet', 'trash'
    ]
    # add-end

    args = parser.parse_args(cmd_args)

    list_of_date = ['2022-09-19', '2022-09-21', '2022-09-26', '2022-10-05', '2022-10-07',
                    '2022-10-12', '2022-10-14']  #

    for item in list_of_date:
        imgs_dir = f'{home_dir}/data/ETRI_GJHall/{item}'
        output_dir = f'./vis/{project_name}/{item}'
        output_video = f'./vis/{project_name}/{item}.mp4'

        main(args, imgs_dir=imgs_dir, output_dir=output_dir)

        images_to_video(output_dir, output_video)
