from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# interface
import pdb

import requests
import json
from PIL import Image
from io import BytesIO
import numpy as np

# model
import _init_paths
import os
import argparse
from MultiStreamDeformableDETR.mains_cloud.demo_multiDB_v6 import get_args_parser
import MultiStreamDeformableDETR.util.misc as utils
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from MultiStreamDeformableDETR.models import build_multioutput_multidataset_model_multitrfmModule as build_model
from MultiStreamDeformableDETR.models.service_detector import build_SAclassifier
import random
import time
from torchvision.ops import nms
from MultiStreamDeformableDETR.engine_saclassifier import get_duration_norm
from MultiStreamDeformableDETR.my_debug import draw_bboxes_on_pil
from PIL import ImageDraw
from MultiStreamDeformableDETR.mains_cloud.service_manager import ServiceManager


class TableServiceAlarm:
    def __init__(self, model_path):
        # setup arguments
        self.start_time_in_seconds = None
        self.list_img_seq = []
        self.list_x_duration = []

        parser = argparse.ArgumentParser('Deformable DETR training and evaluation script',
                                         parents=[get_args_parser()])
        # add script's argument at here
        cmd_args = [
            '--resume', os.path.join(model_path, 'checkpoint.pth'),
            '--backbone', 'resnet50',
            '--class_list', os.path.join(model_path, 'detection_label.txt'),
            '--sac_class_list', os.path.join(model_path, 'sac_label.txt'),

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
            # '--saclassifier_type', 'imageavgp_encv2avgp_pca1dlcnmsattnsimple',
            '--saclassifier_type', 'T5avgp_imageavgp_encv2avgp_pca1dlcnmsattnsimple',
            '--num_prev_imgs', '9',

            '--processing_per_frames', '5',

            '--vis_th', '0.7',
            '--eval',

            '--display_class_names', 'food', 'drink', 'dish', 'cup', 'empty_container',
            'mobile_phone', 'wallet', 'trash'
        ]
        # add-end

        args = parser.parse_args(cmd_args)

        utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(utils.get_sha()))

        if args.frozen_weights is not None:
            assert args.masks, "Frozen training is meant for segmentation only"

        print('Called with args:')
        print(args)

        self.device = torch.device(args.device)

        # F = H( G( x )) )
        self.modelG, _, postprocessorG = build_model(args, num_classes=args.num_classes_on_G,
                                                num_trfms=args.num_trfms)
        self.modelH, _, postprocessorH = build_SAclassifier(args.num_classes_on_H, args.num_trfms,
                                                       saclassifier_type=args.saclassifier_type)
        self.postprocessors = {**postprocessorG, **postprocessorH}

        print('Load from resume_on_G_and_H')
        checkpoint = torch.load(args.resume, map_location='cpu')

        missing_keys_in_G, unexpected_keys_in_G = self.modelG.load_state_dict(checkpoint['modelG'],
                                                                         strict=False)
        missing_keys_in_H, unexpected_keys_in_H = self.modelH.load_state_dict(checkpoint['modelH'],
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

        # if torch.cuda.is_available():
        #     self.modelG.cuda()
        #     self.modelH.cuda()

        self.modelG.to(self.device)
        self.modelH.to(self.device)

        self.modelG.eval()
        self.modelH.eval()

        with open(args.class_list) as f:
            classes = f.readlines()
            self.classes = [line.rstrip('\n') for line in classes]
        print(self.classes)

        with open(args.sac_class_list) as f:
            sac_classes = f.readlines()
            self.sac_classes = [line.rstrip('\n') for line in sac_classes]
        print(self.sac_classes)

        number_of_colors = len(classes)
        self.list_colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                       for i in range(number_of_colors)]

        # function
        self.transform_resize = T.Compose([
            # T.Resize(1000)
            # T.Resize(1376)
            T.Resize(800)
        ])

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.vis_th = 0.7
        self.args = args

        self.service_manager = ServiceManager()


    def setStartTime(self, start_time_in_seconds):
        self.start_time_in_seconds = start_time_in_seconds
        self.service_manager.set_start_time(start_time_in_seconds)


    def detect(self, ipl_img, current_time_in_seconds, draw_result=False):
        if self.start_time_in_seconds is None:
            print('self.start_time_in_seconds is None!')
            return None, None, None, None, None
        else:
            duration_seconds = current_time_in_seconds - self.start_time_in_seconds

            if duration_seconds < 0:
                print('duration_seconds is less than zero!')
                return None, None, None, None, None

        t0 = time.time()

        with torch.no_grad():
            im = self.transform_resize(ipl_img)
            img_x = self.transform(im).unsqueeze(0)
            self.list_img_seq.append(img_x)

            # add duration time
            duration_norm = get_duration_norm(duration_seconds)
            self.list_x_duration.append(duration_norm)

            # remove over given length
            if len(self.list_img_seq) > self.args.num_prev_imgs + 1:
                self.list_img_seq.pop(0)
                self.list_x_duration.pop(0)

            if int(duration_seconds) % self.args.processing_per_frames != 0:
                print('TableServiceAlarm.detect: image stack and skip process',
                      duration_seconds, len(self.list_img_seq))
                detection_results = []
                service_results = []
                repr_service_index = -1
                repr_service_name = 'no_service'
                if draw_result:
                    source_img = im.convert("RGB")  # im == cropped image in PIL.Image type
                else:
                    source_img = None
                return detection_results, service_results, repr_service_index, repr_service_name, source_img
            else:
                print('TableServiceAlarm.detect: process the img_seq',
                      duration_seconds, len(self.list_img_seq))

                # list to tensor
                img_seq = torch.cat(self.list_img_seq, dim=0)
                img_seq = img_seq.cuda()  # [1, 3, 800, 1422]

                x_duration = torch.stack(self.list_x_duration, dim=0)  #
                x_duration = x_duration.unsqueeze(dim=1).cuda()  # n_T, 1

                # propagate through the model
                outputsG = self.modelG(img_seq)
                outputsG['duration'] = x_duration
                outputsH = self.modelH(outputsG)
                outputs = {**outputsG, **outputsH}

                im_w, im_h = im.size
                target_sizes = torch.tensor([[im_h, im_w]])
                target_sizes = target_sizes.cuda()

                # if output is a multi_batch, get the last one only
                for key_outputsG in outputsG.keys():
                    if torch.is_tensor(outputsG[key_outputsG]):
                        outputsG[key_outputsG] = outputsG[key_outputsG][-1:]

                for key_outputsH in outputsH.keys():
                    if torch.is_tensor(outputsH[key_outputsH]):
                        outputsH[key_outputsH] = outputsH[key_outputsH][-1:]

                assert 'bbox_attn' in self.postprocessors.keys()
                resultsG, resultsH = self.postprocessors['bbox_attn'](outputsG, outputsH, target_sizes)

                scores = resultsG[0]['scores']  # [100]
                labels = resultsG[0]['labels']
                boxes = resultsG[0]['boxes']

                keep = scores > self.vis_th
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]  # topk & greater than args.vis_th

                if self.args.use_amount:
                    assert 'amount_score' in resultsH.keys()  # 'amount_score' must be in resultsH
                    amount_score = resultsH['amount_score'][0]  # [100, 50]
                    res_amount_logits = amount_score[keep]  # [keep, 50]

                    # fill zero to amount_score at not bbox with 'food(1)' and 'drink(2)'
                    amount_score[
                        (labels > 2).nonzero(as_tuple=True)] = 0.0  # this cannot change value.

                    res_amount_prob = F.softmax(res_amount_logits, dim=1)  # [n_keep, n_amount]
                    amount_prob, res_amount = torch.topk(res_amount_prob, k=3,
                                                         dim=1)  # [n_q, k], topk_prob, topk_class_index
                    n_div = 100 // res_amount_logits.size(1)
                    amount_pred = ((res_amount.type(torch.float32) * n_div) + n_div / 2) * 0.01

                    res_table = ((torch.arange(0, res_amount_logits.size(
                        1)) * n_div) + n_div / 2) * 0.01
                    res_table = res_table.unsqueeze(0)
                    res_table = res_table.expand(res_amount_prob.size(0), -1)  # n_q, n_class
                    amount_pred_weighted = torch.sum(res_amount_prob * res_table.cuda(), dim=1)

                if self.args.use_progress:
                    out_progress_score = outputs['progress_score']
                    progress_prob = F.softmax(out_progress_score[0, :], dim=0)

                num_topk_vis_th = sum(keep).item()
                print(f'{num_topk_vis_th} bboxes are selected > {self.vis_th}')

                # NMS among transformers
                filtered_labels = []
                filtered_scores = []
                filtered_boxes = []
                filtered_amount_pred = []
                filtered_amount_pred_weighted = []

                for j_class in range(0, self.args.num_classes_on_G):
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

                if draw_result:
                    source_img = im.convert("RGB")  # im == cropped image in PIL.Image type
                    draw = ImageDraw.Draw(source_img)
                else:
                    source_img = None

                detection_results = []
                service_results = torch.sigmoid(outputsH['service_pred_logits'])[0, 1:].detach().cpu()
                service_results = [item.item() for item in service_results]

                for ith, (label, score, (xmin, ymin, xmax, ymax), amount_pred_weighted) in enumerate(
                        zip(labels.tolist(), scores.tolist(), boxes.tolist(),
                            amount_pred_weighted.tolist())):

                    label_index = label - 1  # background is in label, but not in label_index
                    class_name = self.classes[label_index]

                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)
                    detection_results.append([xmin, ymin, xmax, ymax, label_index])

                    if draw_result:
                        if class_name in ['food', 'drink', 'dish', 'cup']:
                            pass
                        else:
                            amount_pred_weighted = None

                        if self.args.use_amount and amount_pred_weighted is not None:
                            str_result = '%s (%.1f) \nam: %.0f' % (class_name, score,
                                                                   amount_pred_weighted * 100)
                            # # draw bar graph
                            # if class_name in ['dish', 'cup']:
                            #     draw.rectangle([xmin, ymin - 10,
                            #                     xmin + (xmax - xmin) * amount_pred_weighted, ymin - 5],
                            #                    fill=self.list_colors[label_index], width=2)
                        else:
                            str_result = '%s (%.1f)' % (class_name, score)

                        draw.multiline_text((xmin, ymin), str_result, fill=self.list_colors[label_index],
                                            spacing=2)

                        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=self.list_colors[label_index],
                                       width=2)

                if draw_result:
                    # draw only one in one image
                    if self.args.use_progress:
                        # draw.text((10, 10), '%d' % topk_index_progress.item())
                        draw.text((10, 10), 'progress: %.1f, %.1f, %.1f' %
                                  (progress_prob[0], progress_prob[1], progress_prob[2]))

                    # print(alarms)
                    str_alarms = []

                    # draw saclass predictions
                    unscaled_boxes = torch.tensor([[10, 1 + 20, 10, 21], [10, 21 + 20, 10, 41],
                                                   [10, 41 + 20, 10, 61], [10, 61 + 20, 10, 81]])
                    res_scores = torch.sigmoid(outputsH['service_pred_logits'])[0, 1:].detach().cpu()
                    source_img = draw_bboxes_on_pil(source_img, unscaled_boxes, self.sac_classes,
                                                    scores=res_scores,
                                                    vis_th=0.0, no_bbox=True)

                    draw.multiline_text((10, 25), '\n'.join(str_alarms), fill='white', spacing=2)

                    # draw the attention
                    if 'bbox_attn' in self.postprocessors.keys():
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
                                    if res_scores[i_c] >= 0.5:      # services having more than 50 percentages
                                        c_attn_bbox = attn_bbox[i_c, :, :]
                                        c_attn_val = attn_value[i_c, :]
                                        # keep = nms(c_attn_bbox, c_attn_val, 0.3)  #
                                        # c_attn_bbox = c_attn_bbox[keep]
                                        # c_attn_val = c_attn_val[keep]
                                        source_img = draw_bboxes_on_pil(source_img,
                                                                        c_attn_bbox,
                                                                        [i_c] * 100,        # labels
                                                                        scores=c_attn_val,
                                                                        vis_th=0.01)

                print('processing time: ', time.time() - t0)

                repr_service_index, repr_service_name = self.service_manager.process(service_results,
                                                                 current_time_in_seconds)

                print('repr_service: ', repr_service_index, repr_service_name)

                if draw_result:
                    draw.text((10, 150), f'repr_service: {repr_service_index}, {repr_service_name}')

            return detection_results, service_results, repr_service_index, repr_service_name, source_img


class TableServiceAlarmRequestHandler(object):
    def __init__(self, model_path):
        self.tsa = TableServiceAlarm(model_path)
        self.det_classes = self.tsa.classes
        self.sac_classes = self.tsa.sac_classes

    def process_start_meal(self, start_time_in_seconds):
        self.tsa.setStartTime(start_time_in_seconds)

    def process_inference_request(self, ipl_img, current_time_in_seconds):
        # 1. Perform detection and classification
        # - detection_result is a list of [x1,y1,x2,y2,class_id]
        # - ex) result = [[100,100,200,200,154], [200,300,200,300,12]]
        # - service_result is a list of four service possible time (food refill, trash collection, serving dessert, lost item)
        # - ex) result = [0.7, 0.1, 0.1, 0.2]
        detection_results, service_results, repr_service_index, rep_service_name, vis_img = self.tsa.detect(ipl_img, current_time_in_seconds, draw_result=True)

        return detection_results, service_results, repr_service_index, rep_service_name, vis_img

    def process_inference_request_imgurl(self, image_url, current_time_in_seconds):
        # 1. Read an url image and convert it to an ipl image
        response = requests.get(image_url)
        ipl_img = Image.open(BytesIO(response.content))     # read as rgb

        # 1. Perform detection and classification
        # - detection_result is a list of [x1,y1,x2,y2,class_id]
        # - ex) result = [(100,100,200,200,154), (200,300,200,300,12)]
        # - service_result is a list of four service possible time (food refill, trash collection, serving dessert, lost item)
        # - ex) result = [0.7, 0.1, 0.1, 0.2]
        detection_results, service_results, repr_service_index, rep_service_name, vis_img = self.tsa.detect(ipl_img, current_time_in_seconds, draw_result=True)

        return detection_results, service_results, repr_service_index, rep_service_name, vis_img


# one time test
if __name__ == '__main__':
    # 1. Initialize
    model_path = '.'
    handler = TableServiceAlarmRequestHandler(model_path)   # init
    print('TableServiceAlarmRequestHandler is initialized!')

    # 2. Set the start time (This information is used as an input)
    handler.process_start_meal(0.0)

    # # One image test
    # # 3. Give an image and get the results
    # ipl_img = Image.open('examples/example.jpg')  # read as rgb
    # duration_time_in_sec = 300
    # detection_results, service_results, repr_service_index, repr_service_name, im2show = \
    #     handler.process_inference_request(ipl_img, duration_time_in_sec)        # request
    # # detection_results, service_results, rep_service_name, im2show = \
    # #     handler.process_inference_request_imgurl(image_url, duration_time_in_sec)        # request
    #
    # # 4. Print the results
    # print("Detection Result: {}".format(json.dumps(detection_results)))
    # for result in detection_results:
    #     print(f"  BBox(x1={result[0]},y1={result[1]},x2={result[2]},y2={result[3]}) => {handler.det_classes[result[4]]}")
    #
    # print("\n")
    # print("Possible Service Results: ")
    # for sac_name, result in zip(handler.sac_classes, service_results):
    #     print(f"  {sac_name}: {result:.4f}")
    #
    # print('\n')
    # print(f'Representative Service index and name: ', repr_service_index, repr_service_name)
    #
    # # 5. Save the result image
    # if im2show is not None:
    #     im2show.save('imgurl_debug_image.jpg')

    # Image sequence test
    for ith in range(1, 16):
        # 3. Give an image and get the results
        ipl_img = Image.open(f'examples/ex{ith:04}.jpg')  # read as rgb
        duration_time_in_sec = 300 + ith
        detection_results, service_results, repr_service_index, repr_service_name, im2show = \
            handler.process_inference_request(ipl_img, duration_time_in_sec)  # request
        # detection_results, service_results, rep_service_name, im2show = \
        #     handler.process_inference_request_imgurl(image_url, duration_time_in_sec)        # request

        # 4. Print the results
        print("Detection Result: {}".format(json.dumps(detection_results)))
        for result in detection_results:
            print(
                f"  BBox(x1={result[0]},y1={result[1]},x2={result[2]},y2={result[3]}) => {handler.det_classes[result[4]]}")

        print("\n")
        print("Possible Service Results: ")
        for sac_name, result in zip(handler.sac_classes, service_results):
            print(f"  {sac_name}: {result:.4f}")

        print('\n')
        print(f'Representative Service index and name: ', repr_service_index, repr_service_name)

        # 5. Save the result image
        if im2show is not None:
            im2show.save(f'imgurl_debug_image_{ith:04}.jpg')

    print('TableServiceAlarmRequestHandler request is processed!')
