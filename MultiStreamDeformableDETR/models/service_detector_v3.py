import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from .tcn import TemporalConvNet

import pdb
from typing import Optional, List
from util import box_ops
from util.misc import accuracy
import math
from models.deformable_detr_multioutput_multidataset_multitrfmModule import masked_mean
import torchvision

# v3: simple arch

class TemporalBinder(nn.Module):
    def __init__(self, num_feature=256, num_output=1, tbinder_type='bypass'):
        super().__init__()
        self.num_feature = num_feature
        self.tbinder_type = tbinder_type
        # self.num_temporal = num_temporal

        if self.tbinder_type == 'attn_qkv':
            self.embed_encoder = nn.Embedding(1, self.num_feature)
            self.attn_encoder = nn.MultiheadAttention(embed_dim=self.num_feature, num_heads=1)

        elif self.tbinder_type == 'attn_simple':
            self.attn_encoder = nn.Linear(self.num_feature, num_output)

    def forward(self, x):
        s1, s4 = x.shape # x:=[n_batch, n_feature]
        if self.tbinder_type == 'bypass':
            out = x
        else:
            # in: [n_batch, n_features]
            # out: [1, n_features]  except for 'bypass'

            if self.tbinder_type == 'maxpool':
                out, _index = x.max(dim=0)
            elif self.tbinder_type == 'avgpool':
                out = x.mean(dim=0)
            elif self.tbinder_type == 'attn_simple':
                memory_weight_score = self.attn_encoder(x)  # [n_temporal, 1024] > [n_temporal, 1]
                output_weights = torch.softmax(memory_weight_score, dim=0)  # > [n_temporal, 1]

                output_weights_swap = output_weights.permute(1, 0)  # > [1, n_temporal]
                memory_pool = torch.matmul(output_weights_swap, x)  # [1, n_temporal] x [n_temporal, 1024] > [1, 1024]
                # out # [1, 1024]
                out = memory_pool

            # elif self.tbinder_type == 'attn_qkv':
            #     enc_memory_pool, enc_output_weights = \
            #         self.attn_encoder(query=self.embed_encoder.weight.transpose(0, 1),
            #                           key=x.tranpose(0, 1),
            #                           value=x.transpose(0, 1))
            #     out = enc_memory_pool
            elif self.tbinder_type == 'tcn':
                # N_batch, n_channel, length
                # [n_temporal, n_dim] > [1, n_temporal, n_dim] > [1, n_dim, n_temporal]
                x_permute = torch.unsqueeze(x, dim=0).permute(0, 2, 1)
                out_temp = self.attn_encoder(x_permute)      # > [n_batch, n_hid, n_temporal]
                out = out_temp[0, :, -1]    # [n_hid]
            else:
                raise AssertionError('unsupported tbinder_type: ', self.tbinder_type)

            out = torch.unsqueeze(out, dim=0)   # [1, 256]

        return out

class ServiceAlarmClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def loss(self, inputs, targets):
        pass


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def masked_max(tensor, mask):
    s1, s2, s3, s4 = tensor.shape   # [n_b, 256, h, w]
    # mask [n_b, 1, h, w]
    # mask: True == real image, False == masked area
    min_filled_tensor = torch.where(mask, tensor, tensor.min())     # True <- tensor, False <- tensor.min()
    max_value, _ = min_filled_tensor.view(s1, s2, -1).max(dim=2)    # [n_batch, hidden_dim]

    return max_value

def masked_softmax(x, mask_T_alive, **kwargs):
    x_masked = x.clone()
    x_masked[mask_T_alive == 0] = -float('inf')

    return torch.softmax(x_masked, **kwargs)


def get_classwise_weight(x_pred_logits, x_pred_boxes, dict_classname, num_classes=5):
    mask_by_class = torch.zeros((num_classes, x_pred_logits.shape[2]))
    mask_by_bbox = torch.zeros((num_classes, x_pred_logits.shape[0], x_pred_logits.shape[1]))

    # (1)refill, (2)trash, (3)dessert, (4)lost
    mask_by_class[0, :] = True

    refill_classes = [dict_classname[item] for item in ['food', 'drink', 'dish', 'cup', 'empty_container', 'bottle']]
    mask_by_class[1, refill_classes] = True

    trash_classes = [dict_classname[item] for item in ['trash']]
    mask_by_class[2, trash_classes] = True

    lost_classes = [dict_classname[item] for item in ['mobile_phone', 'wallet', 'person']]
    mask_by_class[3, lost_classes] = True

    mask_by_class[4, :] = True

    # share all bbox limits
    x1 = 0.2686
    y1 = 0.1563
    x2 = 0.6305
    y2 = 0.5233
    refill_mask_by_bbox = (x_pred_boxes[:, :, 0] - x_pred_boxes[:, :, 2]/2 > x1) * (x_pred_boxes[:, :, 1] - x_pred_boxes[:, :, 3]/2 > y1) *\
                          (x_pred_boxes[:, :, 0] + x_pred_boxes[:, :, 2]/2 < x2) * (x_pred_boxes[:, :, 1] + x_pred_boxes[:, :, 3]/2 < y2)

    # refill_mask_by_bbox [16, 1200]
    # print(refill_mask_by_bbox.sum(dim=1))

    mask_by_bbox[0, :, :] = True
    mask_by_bbox[1, refill_mask_by_bbox] = True
    mask_by_bbox[2, :, :] = True
    mask_by_bbox[3, :, :] = True
    mask_by_bbox[4, :, :] = True


    mask_by_prog_duration = torch.zeros((num_classes, 1, 1))
    mask_by_prog_duration[0, :] = True
    mask_by_prog_duration[3, :] = True

    # float to boolean
    mask_by_class = mask_by_class.type(torch.bool)  # [n_class, n_det_class(10)]
    mask_by_bbox = mask_by_bbox.type(torch.bool)  # [n_class, n_b, n_roi]
    mask_by_prog_duration = mask_by_prog_duration.type(torch.bool)

    return mask_by_class, mask_by_bbox, mask_by_prog_duration


class ImageEncoderROIAttBasedSAClassifier(ServiceAlarmClassifier):
    # training model like a detector
    # making a decision like a classifier
    # just use more information given in the data
    # aggregate_type: 'maxpool', 'avgpool', 'attn'
    def __init__(self, num_classes, num_trfm,
                 use_pca=False, pca_aggregate_type='same',
                 use_duration=False, limit_det_class=False,
                 apply_nms_on_pca=False, apply_one_value_amount=False,
                 tbinder_type='bypass',
                 final_layer_type='linear', num_MLP_final_layers=3, abs_diff_feature=False):

        super().__init__()
        self.use_pca = use_pca
        self.use_duration = use_duration
        self.limit_det_class = limit_det_class
        self.apply_nms_on_pca = apply_nms_on_pca
        self.apply_one_value_amount = apply_one_value_amount
        self.abs_diff_feature = abs_diff_feature

        self.no_value_for_pred = 0.
        self.no_value_for_amount = 1.   # v3, 0. -> 1.
        self.no_value_for_progress = 0.
        self.no_value_for_duration = 0.5

        self.objectness_th = 0.7
        self.topk = 100

        list_det_classname = ['background',
                              'food', 'drink', 'dish', 'cup', 'empty_container',
                              'fork', 'spoon', 'knife', 'chopsticks', 'mobile_phone',
                              'wallet', 'trash', 'tissue_box', 'wet_tissue_pack', 'tableware',
                              'person', 'bottle', 'thing', 'scissors', 'tongs'
                              ]
        self.dict_det_classname = {item: i_th for i_th, item in enumerate(list_det_classname)}
        list_used_det_classname = [
            'food', 'drink', 'dish', 'cup', 'empty_container',
            'mobile_phone', 'wallet', 'trash', 'person', 'bottle'
        ]
        # self.used_det_class = [1, 2, 3, 4, 5, 10, 11, 12, 16, 17]
        self.used_det_class = [self.dict_det_classname[item] for item in list_used_det_classname]
        self.dict_used_det_classname = {item: i_th for i_th, item in enumerate(list_used_det_classname)}

        # self.amount_index = [0, 1, 2, 3, 9]  # only this index (after selecting used_det_class) is selected among proposals
        list_amount_classname = ['food', 'drink', 'bottle']
        self.amount_index = [self.dict_used_det_classname[item] for item in list_amount_classname]

        # 20 classes in detection are:
        # background(0),
        # food(1), drink, dish, cup, empty_container(5),
        # fork(6), spoon, knife, chopsticks, mobile_phone(10),
        # wallet(11), trash, tissue_box, wet_tissue_pack, tableware(15),
        # person(16), bottle, thing, scissors, tongs(20)

        # 4 classes in sac are:
        # background(0),
        # refill(1), trash(2), dessert(3), lost(4)

        self.pca_progress_dim = 3

        if self.apply_one_value_amount:
            self.pca_amount_dim = 1
        else:
            self.pca_amount_dim = 50

        if self.limit_det_class:
            self.pca_detclass_dim = len(self.used_det_class)
        else:
            self.pca_detclass_dim = 21
        print('\tself.pca_detclass_dim: ', self.pca_detclass_dim)
        self.pca_dim = (self.pca_detclass_dim+self.pca_amount_dim+self.pca_progress_dim+4)  # 4: bbox

        if use_duration:
            self.pca_dim += 1

        self.num_classes = num_classes
        self.num_trfm = num_trfm
        self.pca_aggregate_type = pca_aggregate_type

        self.final_layer_type = final_layer_type

        hidden_dim = 0
        if self.use_pca:
            hidden_dim += self.pca_dim

        # tbinder @ total features
        if self.pca_aggregate_type == 'attn_qkv' or self.pca_aggregate_type == 'attn_simple':       # DO NOT change self.aggregate_type to self.tbinder_type.
            # input feature per sac_class
            self.tbinder = nn.ModuleList([TemporalBinder(num_feature=hidden_dim,
                                                         tbinder_type=tbinder_type) for _ in range(num_classes)])
            if self.abs_diff_feature:
                self.tbinder_diff = nn.ModuleList([TemporalBinder(num_feature=hidden_dim,
                                                             tbinder_type=tbinder_type) for _ in
                                              range(num_classes)])

        else:
            self.tbinder = TemporalBinder(num_feature=hidden_dim, num_output=num_classes,
                                          tbinder_type=tbinder_type)
            if self.abs_diff_feature:
                self.tbinder_diff = TemporalBinder(num_feature=hidden_dim, num_output=num_classes,
                                              tbinder_type=tbinder_type)

        if self.use_pca:
            print('\tpca_aggregate_type:', self.pca_aggregate_type)
            if self.pca_aggregate_type == 'attn_qkv':
                raise AssertionError('unsupported yet')
            elif self.pca_aggregate_type == 'attn_simple':
                self.attn_pca_detclass_amount = nn.Linear(self.pca_dim, self.num_classes)


        in_dim = hidden_dim

        if self.abs_diff_feature:
            in_dim = in_dim * 2

        if 'attn' in self.pca_aggregate_type or 'attn' in tbinder_type:
            if self.final_layer_type == 'linear':
                print(f'Final layer: {num_classes} linears {in_dim}x1')
                self.service_class_embed = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_classes)])    # each class has its own linear layer mapping hidden_dim to 1
            elif self.final_layer_type == 'MLP':
                print(f'Final layer: {num_classes} MLP{num_MLP_final_layers} {in_dim}x1')
                self.service_class_embed = nn.ModuleList([
                    MLP(input_dim=in_dim, hidden_dim=in_dim,
                        output_dim=1, num_layers=num_MLP_final_layers) for _ in range(num_classes)])    # each class has its own linear layer mapping hidden_dim to 1
            else:
                raise AssertionError('unsupported')
        else:
            if self.final_layer_type == 'linear':
                print(f'Final layer: linear {in_dim}x{num_classes}')
                self.service_class_embed = nn.Linear(in_dim, num_classes)
            elif self.final_layer_type == 'MLP':
                print(f'Final layer: MLP{num_MLP_final_layers} {in_dim}x{num_classes}')
                self.service_class_embed = MLP(input_dim=in_dim, hidden_dim=in_dim,
                                               output_dim=num_classes, num_layers=num_MLP_final_layers)
            else:
                raise AssertionError('unsupported')

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if isinstance(self.service_class_embed, nn.ModuleList):
            for i in range(num_classes):
                if self.final_layer_type == 'linear':
                    self.service_class_embed[i].bias.data = torch.ones(1) * bias_value
                elif self.final_layer_type == 'MLP':
                    nn.init.constant_(self.service_class_embed[i].layers[-1].bias.data, bias_value)
        else:
            if self.final_layer_type == 'linear':
                self.service_class_embed.bias.data = torch.ones(num_classes) * bias_value
            elif self.final_layer_type == 'MLP':
                nn.init.constant_(self.service_class_embed.layers[-1].bias.data, bias_value)

    def apply_nms(self, boxes, scores, iou_threshold=0.3):
        # copied from https://github.com/pytorch/vision/issues/392
        # boxes: [16, 1200, 4]
        # scores: [16, 1200, 10]

        # boxes is a [batch_size, N, 4] tensor, and
        # scores a [batch_size, N] tensor.
        batch_size, N, _ = boxes.shape
        indices = torch.arange(batch_size, device=boxes.device) # [n_b]
        indices = indices[:, None].expand(batch_size, N).flatten()  # [n_b, 1] > [n_b, N]
        boxes_flat = boxes.flatten(0, 1)    # [n_b, N, 4] > [n_b*N, 4]
        scores_flat = scores.flatten()
        indices_flat = torchvision.ops.boxes.batched_nms(boxes_flat, scores_flat, indices, iou_threshold)
        # now reshape the indices as you want, maybe
        # projecting back to the [batch_size, N] space
        # I'm omitting this here
        # indices = indices_flat

        indices_mask = torch.zeros((batch_size*N))
        indices_mask[indices_flat] = 1.0
        indices_mask = indices_mask.view(batch_size, N)

        return indices_mask

    def forward(self, x):
        # y = h(x)
        if self.use_pca:
            x_pred_logits = x['pred_logits']    # [n_batch(10), n_roi(1200=300*4), 21]
            x_pred_bbox = x['pred_boxes']

            # x_pred_prob_softmax = torch.softmax(x_pred_logits, dim=2) # logit to probability @ FasterRCNN
            x_pred_prob = x_pred_logits.sigmoid() # @ DETR  [n_b, n_roi, n_class(21)

            # select useful classification results only
            if self.limit_det_class:
                # n_b, roi(1200), 21(det) > ", ", 10(det_selected w/ no_bg)
                x_pred_prob = x_pred_prob[:, :, self.used_det_class]
            del x_pred_logits

            x_amount_score = x['amount_score']  # n_b, roi(1200), n_class_amount(50)

            s1, s2, s3 = x_amount_score.shape  # [n_batch or n_seq, n_roi(1200), 50]
            # along with class, [n_batch, 1200]
            nz_index = x_amount_score.mean(dim=2).nonzero(as_tuple=True)
            # 1200 amounts came from 4 transformer
            # 3 yield amount, 1 not. (they are all zeros)
            # logit to probability # [n_batch, n_roi, 50]
            x_amount_score = torch.softmax(x_amount_score, dim=2)

            # change one values and fill no_value_for_amount in meaningless object (such as. phone)
            if self.apply_one_value_amount:
                # select one value in a weighted sum way and scaled to 0 ~ 1.0
                n_div = 100 // x_amount_score.size(2)   # 100 // 50 > n_div = 2
                res_table = ((torch.arange(0, x_amount_score.size(2)) * n_div) + n_div / 2) * 0.01
                # res_table = [0.01, 0.03, 0.05, ... 0.97, 0.99] # [50]
                res_table = res_table.unsqueeze(0).unsqueeze(0) # [1, 1, 50]
                res_table = res_table.expand(s1, s2, s3)    # [s1, s2, 50]
                x_amount_score = torch.sum(x_amount_score * res_table.cuda(), dim=2, keepdim=True)  # [n_b, n_roi, 1]
                s1, s2, s3 = x_amount_score.shape  # [n_batch, n_roi, 1]

                # added at v3, meaningless amount becomes 1.0 instead of 0.5
                mask_x_amount_score = torch.zeros(x_amount_score.shape).type(torch.bool)
                mask_x_amount_score[nz_index] = True
                x_amount_score[~mask_x_amount_score] = self.no_value_for_amount

            x_amount_score_nz = x_amount_score      # > [n_b, n_roi(1200), 50]
            del x_amount_score

            # too much rois > remove it
            if self.apply_nms_on_pca:
                # just pick the topk scores and indexes in batch-wise
                topk_values, topk_indexes = torch.topk(x_pred_prob.view(x_pred_prob.shape[0], -1),
                                                       self.topk, dim=1)
                x_pred_scores = topk_values # [n_b, n_topk]
                topk_boxes = topk_indexes // x_pred_prob.shape[2]  # share -> n_query_index
                x_pred_labels = topk_indexes % x_pred_prob.shape[2]  # remain -> n_class_index
                # x_pred_bbox_xyxy = box_ops.box_cxcywh_to_xyxy(x_pred_bbox)
                x_pred_bbox_cxcywh = x_pred_bbox
                topk_x_pred_bbox_cxcywh = torch.gather(x_pred_bbox_cxcywh, 1,
                                                       topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
                topk_x_pred_prob = torch.gather(x_pred_prob, 1,
                                                topk_boxes.unsqueeze(-1).repeat(1, 1, x_pred_prob.shape[2]))
                topk_x_amount_score_nz = torch.gather(x_amount_score_nz, 1,
                                                topk_boxes.unsqueeze(-1).repeat(1, 1, x_amount_score_nz.shape[2]))

                # x_pred_scores # [n_b, n_topk]
                # how to control different rois? generate a mask and give prob zero
                objectness_keep_masks = x_pred_scores > self.objectness_th  # T/F [n_b, 100]
                # x_pred_bbox_cxcywh = x_pred_bbox_cxcywh[0, keep]
                # x_pred_labels = x_pred_labels[0, keep]
                # x_pred_scores = x_pred_scores[0, keep]  # topk & greater than args.vis_th

                # topk_x_pred_prob: [n_b, n_topk, n_class]
                # topk_x_amount_score_nz: [n_b, n_topk, 1]
                # topk_x_pred_bbox_cxcywh: [n_b, n_topk, 4]

                x_pred_prob = topk_x_pred_prob
                x_pred_bbox = topk_x_pred_bbox_cxcywh
                x_amount_score_nz = topk_x_amount_score_nz

                s1, s2, s3 = x_pred_prob.shape        # n_b, n_roi(1200), n_class(10)
                nms_index_masks = []
                for i_th in range(s3):
                    x_pred_prob_c = x_pred_prob[:, :, i_th].detach()    # n_b, n_roi(1200)

                    keep_indices = self.apply_nms(x_pred_bbox, x_pred_prob_c)     # one bbox, n_class scores
                    # nms_index_masks += keep_indices
                    nms_index_masks.append(keep_indices)

                nms_index_masks = torch.stack(nms_index_masks, dim=2)   # n_b, n_roi, n_class
                nms_index_masks = nms_index_masks.type(torch.bool)
                # final keep_index_masks [n_b, 1200] inc. False or True
                # this will be used in final attnSimple at merge 1200 into 1.

            x_progress_score = x['progress_score']  # [n_batch, 3]
            x_progress_score = torch.softmax(x_progress_score, dim=1)   # [n_b, 3] # logit to probability

            if self.use_duration:
                x_duration = x['duration']      # 0~1, [n_b, 1]
            else:
                x_duration = None

            # at this point, we have all materials
            # x_pred_prob:      [n_b, n_roi, n_class] > make a list w/ limited thresholds
            # x_pred_bbox:      [n_b, n_roi, 4]
            # x_amount_score_nz:[n_b, n_roi, 1] > matched w/ x_pred_prob
            # x_progress_score: [n_b, 3]
            # x_duration:       [n_b, 1]
            # objectness_keep_masks: [n_b, n_roi]   T is alive
            # nms_index_masks:  [10, 1200, 10]

            if self.apply_nms_on_pca:
                # sum(nms_index_masks[0].sum(1) >= 0) : 1200 roi
                # sum(nms_index_masks[0].sum(1) > 0) : ex: 824 roi --> too much

                x_pred_prob[~nms_index_masks] = self.no_value_for_pred
                nms_index_masks_amount = nms_index_masks[:, :, self.amount_index].sum(
                    dim=2).type(torch.bool)
                x_amount_score_nz[~nms_index_masks_amount] = self.no_value_for_amount

            x_pred_prob[~objectness_keep_masks] = self.no_value_for_pred
            x_amount_score_nz[~objectness_keep_masks] = self.no_value_for_amount

            # TODO: dish find the food and get the amount of food, if not food then it has 0 amount

            if self.pca_aggregate_type == 'attn_simple':
                # same shape and only reduce 0 on weight is the best
                # x_pred_bbox:      [n_b, n_roi, 4]
                # x_pred_prob:      [n_b, n_roi, n_class]
                # x_amount_score_nz:[n_b, n_roi, 1] > matched w/ x_pred_prob
                # x_progress_score: [n_b, 3]
                # x_duration:       [n_b, 1]
                # objectness_keep_masks: [n_b, n_roi]   T is alive
                # nms_index_masks:  [10, 1200, 10]

                n_roi = x_pred_bbox.size(1)
                x_duration = x_duration.unsqueeze(dim=1).repeat(1, n_roi, 1)
                x_progress_score = x_progress_score.unsqueeze(dim=1).repeat(1, n_roi, 1)

                if self.use_duration:
                    x_feature = torch.cat([x_pred_bbox, x_pred_prob, x_amount_score_nz, x_progress_score, x_duration], dim=2)
                else:
                    x_feature = torch.cat([x_pred_bbox, x_pred_prob, x_amount_score_nz, x_progress_score], dim=2)

                x_feature_weight_score = self.attn_pca_detclass_amount(
                    x_feature)  # [", ", n_det_class+amount] -> [n_batch, n_roi, n_class]
                # x_feature_weight = torch.softmax(x_feature_weight_score, dim=1)  # [n_b, n_roi, n_class]
                x_feature_weight = masked_softmax(x_feature_weight_score, objectness_keep_masks, dim=1)

                x_feature_weight_swap = x_feature_weight.permute(0, 2, 1)  # [n_b, n_class, n_roi]

                x_feature_agg = torch.matmul(x_feature_weight_swap, x_feature)  # > [n_b, n_class(5), 21]
                x_feature_agg = x_feature_agg.permute(1, 0, 2)  # [n_class(5), n_b, 21]

                # # weight map - start
                # # x_pred_prob_weight_swap # [n_b, n_class(5), n_roi(1200)]
                # # x_amount_score_weight_swap # [n_b, n_class, n_roi_nz(variable)]
                # # x_amount_score_weight_swap_orgsize # [n_b, n_c, n_roi]
                # x_amount_score_weight_swap_orgsize = torch.zeros(x_pred_prob_weight_swap.shape, device=x_pred_prob_weight_swap.device)
                # x_amount_score_weight_swap_orgsize = x_amount_score_weight_swap_orgsize.permute(0, 2, 1)    # [1, 1200, 5]
                # x_amount_score_weight_swap_orgsize[nz_index] = x_amount_score_weight[?] # [1, 900, 5]
                # x_amount_score_weight_swap_orgsize = x_amount_score_weight_swap_orgsize.permute(0, 2, 1) # [1, 5, 1200]
                # pca_weights_amount = x_amount_score_weight_swap_orgsize
                x_feature_weights_bbox = x_feature_weight_swap.detach() # [n_b, n_class, n_roi]
                # weight map - stop
            else:
                raise AssertionError('Unsupported pca_aggregate_type: ', self.pca_aggregate_type)

        # x_feature_agg    [n_sac_class, n_b, n_ftr_dim]
        # x_feature_weights_bbox [n_b, n_sac_class, n_roi]

        # head classifier
        list_output_scores = []
        for i in range(self.num_classes):
            list_attn_features = []
            if self.use_pca:
                if self.pca_aggregate_type == 'avgpool' or self.pca_aggregate_type == 'maxpool':
                    list_attn_features.append(x_feature_agg)  # [n_batch, n_dim]
                else:
                    list_attn_features.append(x_feature_agg[i, :, :])  # [n_class(5), n_batch(16), n_dim(74)]

            x_cat = torch.cat(list_attn_features, dim=1)  # [n_batch, n_hid1+n_hid2+...]
            x_cat_temporal = self.tbinder[i](x_cat)

            if self.abs_diff_feature:
                if x_cat.shape[0] == 1:
                    x_cat_diff = x_cat
                else:
                    # x_cat_diff = torch.diff(x_cat, dim=0)
                    x_cat_diff = x_cat[1:] - x_cat[:-1]
                    x_cat_diff = torch.abs(x_cat_diff)
                x_cat_diff_temporal = self.tbinder_diff[i](x_cat_diff)
                x_cat_temporal = torch.cat([x_cat_temporal, x_cat_diff_temporal], dim=1)

            list_output_scores.append(self.service_class_embed[i](x_cat_temporal))   # n_batch, 1

        output_scores = torch.cat(list_output_scores,
                                  dim=1)  # list of [n_batch, 1] -> [n_batch, n_class]

        out = {'service_pred_logits': output_scores,
               'x_pred_bbox': x_pred_bbox,
               'x_pred_prob': x_pred_prob,
               'x_amount_score_nz': x_amount_score_nz,
               'x_progress_score': x_progress_score,
               'x_duration': x_duration,
               'x_feature_weights_bbox': x_feature_weights_bbox}
        return out

class SetCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.thresh = 0.5
        self.weight_dict = {
            'loss_bce_saclass': 1.0
        }

    def loss_service_classification(self, outputs, targets):
        # prob = inputs.sigmoid()
        src_logits = outputs['service_pred_logits']  # n_batch x n_class

        if len(src_logits.shape) == 3:
            src_logits = torch.squeeze(src_logits, dim=2)

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1]],
                                            dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)

        for i_t, tgt in enumerate(targets):
            for i_l in tgt['labels']:
                target_classes_onehot[i_t, i_l] = 1

        loss_bce = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot)

        ####
        num_samples = target_classes_onehot.shape[0]

        src_prob = torch.sigmoid(src_logits)
        prob_to_class = src_prob > self.thresh

        acc = torch.sum(prob_to_class == target_classes_onehot, dim=0).float() / float(
            num_samples)
        acc = acc[1:]       # 1st background
        class_error = 100 - torch.mean(acc) * 100.0
        ####

        return loss_bce, class_error

    def forward(self, outputs, targets):
        loss_bce, class_error = self.loss_service_classification(outputs, targets)

        losses = {'loss_bce_saclass': loss_bce,
                  'class_error': class_error}

        return losses


class PostProcessBBox(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    """ This module is copied and modified from Deformable DETR code """

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputsG, outputsH, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputsG['pred_logits'], outputsG['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        print('pred and bbox')
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]        # index of rois
        labels = topk_indexes % out_logits.shape[2]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes_xyxy, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        print('results_sac')
        results_sac = {}
        if 'amount_score' in outputsG.keys():
            results_sac['amount_score'] = outputsG['amount_score'][:, topk_boxes[0], :]  # [1, 100, 50]

        # To show attn
        # service_pred_logits = outputsH['service_pred_logits']
        # n_sac_classes = service_pred_logits.shape[1]    # 5
        #
        # print('hs_output_weights')
        # hs_output_weights = outputsH['hs_output_weights']  # [n_b, n_class(5), 1200]
        # if hs_output_weights is not None:
        #     results_sac['hs_attn_values'], results_sac['hs_attn_bbox'] = \
        #         self.get_weightedbbox_from_weight(hs_output_weights, n_sac_classes, boxes_xyxy, scale_fct,
        #                                  topk=3)
        #
        # # hs_output_weights = outputsH['hs_output_weights']  # [n_b, n_class(5), 1200]
        # # list_hs_attn_values = []
        # # # list_hs_attn_bbox_indexes = []
        # # list_hs_boxes = []
        # # if hs_output_weights is not None:
        # #     for i_c in range(1, n_sac_classes):
        # #         hs_attn_values, hs_attn_bbox_indexes = torch.topk(hs_output_weights[:, i_c, :], 3, dim=1)    # [n_b, 1200]
        # #         boxes_hs = torch.gather(boxes_xyxy, 1, hs_attn_bbox_indexes.unsqueeze(-1).repeat(1, 1, 4))
        # #
        # #         list_hs_attn_values.append(hs_attn_values.unsqueeze(1))
        # #         # list_hs_attn_bbox_indexes.append(hs_attn_bbox_indexes.unsqueeze(1))
        # #         list_hs_boxes.append(boxes_hs.unsqueeze(1))
        # #
        # #     results_sac['hs_attn_values'] = torch.cat(list_hs_attn_values, dim=1)
        # #     results_sac['hs_attn_bbox'] = torch.cat(list_hs_boxes, dim=1) * scale_fct[:, None, None, :]  # n_batch, n_class-1, topk, 4(xyxyx)
        #
        # print('enc_output_weights')
        # enc_output_weights = outputsH['enc_output_weights']
        # if enc_output_weights is not None:
        #     results_sac['enc_attn_values'], results_sac['enc_attn_bbox'] = \
        #         self.get_weightedbbox_from_weight(enc_output_weights, n_sac_classes, boxes_xyxy, scale_fct,
        #                                  topk=3)
        #
        # # enc_output_weights = outputsH['enc_output_weights']
        # # list_enc_attn_values = []
        # # # list_enc_attn_bbox_indexes = []
        # # list_enc_boxes = []
        # # if enc_output_weights is not None:
        # #     for i_c in range(1, n_sac_classes):
        # #         enc_attn_values, enc_attn_bbox_indexes = torch.topk(enc_output_weights[:, i_c, :], 3, dim=1)
        # #         boxes_enc = torch.gather(boxes_xyxy, 1, enc_attn_bbox_indexes.unsqueeze(-1).repeat(1, 1, 4))
        # #
        # #         list_enc_attn_values.append(enc_attn_values.unsqueeze(1))
        # #         # list_enc_attn_bbox_indexes.append(enc_attn_bbox_indexes.unsqueeze(1))
        # #         list_enc_boxes.append(boxes_enc.unsqueeze(1))
        # #
        # #     results_sac['enc_attn_values'] = torch.cat(list_enc_attn_values, dim=1)
        # #     results_sac['enc_attn_bbox'] = torch.cat(list_enc_boxes, dim=1) * scale_fct[:, None, None, :]
        # print('pca_weights_bbox')
        # pca_bbox_output_weights = outputsH['pca_weights_bbox']    # [1, 5, 1200]
        # if pca_bbox_output_weights is not None:
        #     results_sac['pca_bbox_attn_values'], results_sac['pca_bbox_attn_bbox'] = \
        #         self.get_weightedbbox_from_weight(pca_bbox_output_weights, n_sac_classes, boxes_xyxy, scale_fct,
        #                                  topk=3)
        #
        # print('pca_weights_bbox - plus')
        # pca_amount_output_weights = outputsH['pca_weights_amount']  # [1, 5, 900]
        # if pca_amount_output_weights is not None:
        #     s1, s2, s3 = boxes_xyxy.shape
        #     nz_index = outputsH['pca_weights_amount_nz_index']  # tuple w/ 2 elements
        #     boxes_xyxy_nz = boxes_xyxy[nz_index].view(s1, -1, s3)
        #
        #     results_sac['pca_amount_attn_values'], results_sac['pca_amount_attn_bbox'] = \
        #         self.get_weightedbbox_from_weight(
        #         pca_amount_output_weights, n_sac_classes, boxes_xyxy_nz, scale_fct,
        #         topk=3)
        #
        # # outputsH
        # # 'enc_output_weights' # encoder
        # # 'hs_output_weights' # ROI

        return results, results_sac

    @torch.no_grad()
    def get_weightedbbox_from_weight(self, _input_weights, n_sac_classes, boxes_xyxy, scale_fct, topk=3):
        list_attn_values = []
        list_boxes = []

        if _input_weights.shape[0] != boxes_xyxy.shape[0]:
            input_weights = _input_weights[-1:].clone().detach()
        else:
            input_weights = _input_weights.clone().detach()

        for i_c in range(1, n_sac_classes):
            # input_weights # [n_b, n_clas(5), 1200]
            attn_values, attn_bbox_indexes = torch.topk(input_weights[:, i_c, :], topk, dim=1)  # [n_b, 1200] -> [n_b, n_top_k]
            attn_boxes = torch.gather(boxes_xyxy, 1,
                                    attn_bbox_indexes.unsqueeze(-1).repeat(1, 1, 4))

            list_attn_values.append(attn_values.unsqueeze(1))   # [n_b, n_top_k] > [n_b, 1, n_top_k]
            list_boxes.append(attn_boxes.unsqueeze(1))

        attn_values = torch.cat(list_attn_values, dim=1)        # [n_b, n_class(4), n_top_k]
        attn_bbox = torch.cat(list_boxes, dim=1) * scale_fct[:, None, None, :]  # n_batch, n_class-1, topk, 4(xyxyx)

        return attn_values, attn_bbox

def build_SAclassifier(num_classes, num_trfm, saclassifier_type='imagebased'):
    print('selected saclassifier type: ', saclassifier_type)

    criterion = SetCriterion()
    postprocessors = {}

    if saclassifier_type == 'T5avgp_pca1dlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    apply_one_value_amount=True,
                                                    pca_aggregate_type='attn_simple',
                                                    tbinder_type='avgpool')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'T5avgp_pca1dlcnmsattnsimpleAbsdiff':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    apply_one_value_amount=True,
                                                    pca_aggregate_type='attn_simple',
                                                    tbinder_type='avgpool', abs_diff_feature=True)
        postprocessors['bbox_attn'] = PostProcess()

    # elif saclassifier_type == 'pca1dlcnmsattnsimple':
    #     model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm,
    #                                                 use_pca=True, use_duration=True,
    #                                                 limit_det_class=True, apply_nms_on_pca=True,
    #                                                 apply_one_value_amount=True,
    #                                                 pca_aggregate_type='attn_simple')
    #     postprocessors['bbox_attn'] = PostProcess()

    else:
        raise AssertionError(f'{saclassifier_type} is undefined!')

    # postprocessors['bbox'] = PostProcessBBox()
    # postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
