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

        elif self.tbinder_type == 'tcn':
            # input: [n_batch, n_channel, n_seq_len]
            nhid = 25 # 25 default at tcn
            nlevels = 8 # 8 default at tcn
            channel_sizes = [nhid] * nlevels
            self.attn_encoder = TemporalConvNet(num_inputs=self.num_feature,
                                                num_channels=channel_sizes)


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


# class ImageBasedSAClassifier(ServiceAlarmClassifier):
#     def __init__(self, hidden_dim, num_classes):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_classes = num_classes
#
#         self.service_class_embed = nn.Linear(hidden_dim, num_classes)
#
#         # initialization
#         prior_prob = 0.01
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         self.service_class_embed.bias.data = torch.ones(num_classes) * bias_value
#
#     def forward(self, x):
#         # y = h(x)
#         output_scores = self.service_class_embed(x['backbone_features'])
#         out = {'service_pred_logits': output_scores}  # n_batch, 4 (min:0.0001, max: 1.000)
#
#         return out


# class ROIBasedSAClassifier(ServiceAlarmClassifier):
#     # training model like a detector
#     # making a decision like a classifier
#     # just use more information given in the data
#     def __init__(self, hidden_dim, num_classes):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_classes = num_classes
#
#         self.service_class_embed = nn.Linear(hidden_dim, num_classes)
#
#         # initialization
#         prior_prob = 0.01
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         self.service_class_embed.bias.data = torch.ones(num_classes) * bias_value
#
#     def forward(self, x):
#         # y = h(x)
#         # x['backbone_features']    # [n_batch, 1024]
#         # x['hs_last']              # [n_batch, n_trfm(4), n_query(300), 256]
#         # x['enc_memory']           # [n_batch, n_trfm(4), n_spatial_pixels(15595), 256]
#
#         s1, s2, s3, s4 = x['hs_last'].shape
#         x_hs_last = x['hs_last'].view(s1, s2*s3, s4)
#         x_hs_last_max, _ = x_hs_last.max(dim=1)
#
#         x_cat = torch.cat((x_hs_last_max, x['backbone_features']), dim=1)
#
#         output_scores = self.service_class_embed(x_cat)
#         out = {'service_pred_logits': output_scores}       # n_batch, 4 (min:0.0001, max: 1.000)
#
#         return out

# class ROIAttBasedSAClassifier(ServiceAlarmClassifier):
#     # training model like a detector
#     # making a decision like a classifier
#     # just use more information given in the data
#     def __init__(self, hidden_dim, num_classes):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_classes = num_classes
#
#         self.service_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_classes)])
#
#         self.sac_embed = nn.Embedding(self.num_classes, 256)
#         self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=1)
#
#         # initialization
#         prior_prob = 0.01
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         for i in range(num_classes):
#             self.service_class_embed[i].bias.data = torch.ones(1) * bias_value
#
#     def forward(self, x):
#         # y = h(x)
#         # x['backbone_features']    # [n_batch, 1024]
#         # x['hs_last']              # [n_batch, n_trfm(4), n_query(300), 256]
#         # x['enc_memory']           # [n_batch, n_trfm(4), n_spatial_pixels(15595), 256]
#         s1, s2, s3, s4 = x['hs_last'].shape     # [n_batch, n_trfm(4), n_query(300), 256]
#         x_hs_last = x['hs_last'].view(s1, s2*s3, s4)    # [n_batch, n_trfm(4)*n_query(300), 256]
#
#         sac_embed = self.sac_embed.weight.unsqueeze(0).repeat(s1, 1, 1)
#
#         # input order: seq_len, n_batch, embedding_dim
#         attn_output, attn_output_weights = self.attn(query=sac_embed.transpose(0, 1), key=x_hs_last.transpose(0, 1), value=x_hs_last.transpose(0, 1))
#         # attn_output.shape     # [5, 16, 256]
#         # attn_output_weights.shape # [16, 5, 1200]
#         # attn_output = attn_output.transpose(0, 1)       # > [n_batch, n_query, emb]
#
#         list_output_scores = []
#         for i in range(self.num_classes):
#             x_cat = torch.cat((attn_output[i, :, :], x['backbone_features']), dim=1)        # 16, 1280 (1024+256)
#             list_output_scores.append(self.service_class_embed[i](x_cat))
#
#         output_scores = torch.cat(list_output_scores, dim=1)    # list of [n_batch, 1] -> [n_batch, n_class]
#
#         out = {'service_pred_logits': output_scores}       # n_batch, 4 (min:0.0001, max: 1.000)
#
#         return out

def masked_max(tensor, mask):
    s1, s2, s3, s4 = tensor.shape   # [n_b, 256, h, w]
    # mask [n_b, 1, h, w]
    # mask: True == real image, False == masked area
    min_filled_tensor = torch.where(mask, tensor, tensor.min())     # True <- tensor, False <- tensor.min()
    max_value, _ = min_filled_tensor.view(s1, s2, -1).max(dim=2)    # [n_batch, hidden_dim]

    return max_value

def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -float('inf')

    return torch.softmax(x_masked, **kwargs)


class ImageEncoderROIAttBasedSAClassifier(ServiceAlarmClassifier):
    # training model like a detector
    # making a decision like a classifier
    # just use more information given in the data
    # aggregate_type: 'maxpool', 'avgpool', 'attn'
    def __init__(self, num_classes, num_trfm,
                 use_backbone=True, backbone_aggregate_type='same', apply_btl_backbone=False,
                 use_encoder=True, apply_btl_encoder=False,
                 use_roi=True, aggregate_type='maxpool', apply_btl_roi=False,
                 use_pca=False, pca_aggregate_type='same',
                 use_duration=False, limit_det_class=False, use_merge_det_amount=False,
                 apply_nms_on_pca=False, apply_one_value_amount=False,
                 backbone_dim=1024, encoder_dim=256, roi_dim=256,
                 tbinder_type='bypass',
                 final_layer_type='linear', num_MLP_final_layers=3):

        super().__init__()
        self.use_backbone = use_backbone
        self.use_encoder = use_encoder
        self.use_roi = use_roi
        self.use_pca = use_pca
        self.use_duration = use_duration
        self.limit_det_class = limit_det_class
        self.use_merge_det_amount = use_merge_det_amount
        self.apply_nms_on_pca = apply_nms_on_pca
        self.apply_btl_backbone = apply_btl_backbone
        self.apply_btl_encoder = apply_btl_encoder
        self.apply_btl_roi = apply_btl_roi
        self.apply_one_value_amount = apply_one_value_amount

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
        list_amount_classname = ['food', 'drink', 'dish', 'cup', 'bottle']
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

        self.backbone_dim = backbone_dim
        self.encoder_dim = encoder_dim
        self.roi_dim = roi_dim

        self.pca_progress_dim = 3
        self.btl_div = 16.

        if self.apply_one_value_amount:
            self.pca_amount_dim = 1
        else:
            self.pca_amount_dim = 50

        if self.limit_det_class:
            self.pca_detclass_dim = len(self.used_det_class)
        else:
            self.pca_detclass_dim = 21
        print('\tself.pca_detclass_dim: ', self.pca_detclass_dim)
        self.pca_dim = (self.pca_detclass_dim+self.pca_amount_dim+self.pca_progress_dim)

        if use_duration:
            self.pca_dim += 1

        self.num_classes = num_classes
        self.num_trfm = num_trfm
        self.aggregate_type = aggregate_type

        if backbone_aggregate_type == 'same':
            self.backbone_aggregate_type = self.aggregate_type
        else:
            self.backbone_aggregate_type = backbone_aggregate_type

        if pca_aggregate_type == 'same':
            self.pca_aggregate_type = self.aggregate_type
        else:
            self.pca_aggregate_type = pca_aggregate_type

        self.lv_backbone = 4

        self.final_layer_type = final_layer_type

        if self.use_backbone and self.apply_btl_backbone:
            backbone_dim_input = self.backbone_dim
            self.backbone_dim = int(self.backbone_dim / self.btl_div)

            if self.backbone_aggregate_type == 'attn_qkv' or self.backbone_aggregate_type == 'attn_simple':
                self.backbone_btl = nn.ModuleList([nn.Sequential(nn.Linear(backbone_dim_input, self.backbone_dim),
                                                  nn.ReLU()) for _ in range(num_classes)])
            else:
                self.backbone_btl = nn.Sequential(nn.Linear(backbone_dim_input, self.backbone_dim),
                                                  nn.ReLU())

        if self.use_roi and self.apply_btl_roi:
            roi_dim_input = self.roi_dim*self.num_trfm
            self.roi_dim = int(self.roi_dim / self.btl_div)
            roi_dim_output = self.roi_dim*self.num_trfm

            if self.aggregate_type == 'attn_qkv' or self.aggregate_type == 'attn_simple':
                self.roi_btl = nn.ModuleList([nn.Sequential(nn.Linear(roi_dim_input, roi_dim_output), nn.ReLU()) for _ in range(num_classes)])
            else:
                self.roi_btl = nn.Sequential(nn.Linear(roi_dim_input, roi_dim_output), nn.ReLU())

        if self.use_encoder and self.apply_btl_encoder:
            enc_dim_input = self.encoder_dim
            self.encoder_dim = int(self.encoder_dim / self.btl_div)

            if self.aggregate_type == 'attn_qkv' or self.aggregate_type == 'attn_simple':
                self.encoder_btl = nn.ModuleList(
                    [nn.Sequential(nn.Linear(enc_dim_input, self.encoder_dim),
                                                 nn.ReLU()) for _ in range(num_classes)])
            else:
                self.encoder_btl = nn.Sequential(nn.Linear(enc_dim_input, self.encoder_dim),
                                                 nn.ReLU())

        hidden_dim = 0
        if self.use_pca:
            hidden_dim += self.pca_dim
        if self.use_backbone:
            hidden_dim += self.backbone_dim
        if self.use_encoder:
            hidden_dim += (self.encoder_dim * self.num_trfm)
        if self.use_roi:
            hidden_dim += (self.roi_dim * self.num_trfm)

        # tbinder @ total features
        if self.aggregate_type == 'attn_qkv' or self.aggregate_type == 'attn_simple':       # DO NOT change self.aggregate_type to self.tbinder_type.
            # input feature per class
            self.tbinder = nn.ModuleList([TemporalBinder(num_feature=hidden_dim,
                                                         tbinder_type=tbinder_type) for _ in range(num_classes)])
        else:
            self.tbinder = TemporalBinder(num_feature=hidden_dim, num_output=num_classes,
                                          tbinder_type=tbinder_type)

        if self.use_backbone:
            print('\tbackbone_aggregate_type:', self.backbone_aggregate_type)
            if self.backbone_aggregate_type == 'attn_qkv':
                self.sac_embed_backbone = nn.ModuleList([nn.Embedding(self.num_classes, 256) for _ in range(self.lv_backbone)])
                self.attn_backbone = nn.ModuleList([nn.MultiheadAttention(embed_dim=256, num_heads=1) for _ in range(self.lv_backbone)])
            elif self.backbone_aggregate_type == 'attn_simple':
                self.attn_backbone = nn.ModuleList(
                    [nn.Linear(256, self.num_classes) for _ in range(self.lv_backbone)])

        if self.use_pca:
            print('\tpca_aggregate_type:', self.pca_aggregate_type)
            if self.pca_aggregate_type == 'attn_qkv':
                raise AssertionError('unsupported yet')
            elif self.pca_aggregate_type == 'attn_simple':
                if self.use_merge_det_amount:
                    self.attn_pca_detclass_amount = nn.Linear(self.pca_amount_dim+self.pca_detclass_dim, self.num_classes)
                else:
                    self.attn_pca_amount = nn.Linear(self.pca_amount_dim, self.num_classes)
                    self.attn_pca_detclass = nn.Linear(self.pca_detclass_dim, self.num_classes)

        print('\taggregate_type:', self.aggregate_type)
        if self.aggregate_type == 'attn_qkv':
            # each class owns its embedding, but shares attention layers.
            if self.use_roi:
                self.sac_embed_roi = nn.ModuleList([nn.Embedding(self.num_classes, 256) for _ in range(self.num_trfm)])
                self.attn_roi = nn.ModuleList([nn.MultiheadAttention(embed_dim=256, num_heads=1) for _ in range(self.num_trfm)])
            if self.use_encoder:
                self.sac_embed_encoder = nn.ModuleList([nn.Embedding(self.num_classes, 256) for _ in range(self.num_trfm)])
                self.attn_encoder = nn.ModuleList([nn.MultiheadAttention(embed_dim=256, num_heads=1) for _ in range(self.num_trfm)])
        elif self.aggregate_type == 'attn_simple':
            if self.use_roi:
                self.attn_roi = nn.ModuleList([nn.Linear(256, self.num_classes) for _ in range(self.num_trfm)])
            if self.use_encoder:
                self.attn_encoder = nn.ModuleList([nn.Linear(256, self.num_classes) for _ in range(self.num_trfm)])

        if 'attn' in self.aggregate_type or 'attn' in tbinder_type:
            if self.final_layer_type == 'linear':
                print(f'Final layer: {num_classes} linears {hidden_dim}x1')
                self.service_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_classes)])    # each class has its own linear layer mapping hidden_dim to 1
            elif self.final_layer_type == 'MLP':
                print(f'Final layer: {num_classes} MLP{num_MLP_final_layers} {hidden_dim}x1')
                self.service_class_embed = nn.ModuleList([
                    MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
                        output_dim=1, num_layers=num_MLP_final_layers) for _ in range(num_classes)])    # each class has its own linear layer mapping hidden_dim to 1
            else:
                raise AssertionError('unsupported')
        else:
            if tbinder_type == 'tcn':
                in_dim = 25 # 25 is nhid of tcn output node
            else:
                in_dim = hidden_dim

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
        list_features = []

        # if self.use_backbone:
        #     # x['backbone_features']    # [n_batch, 1024]
        #     x_backbone = x['backbone_features']    # [n_batch, 1024]
        #     list_features.append(x_backbone)

        pca_weights_bbox = None
        pca_weights_amount = None
        pca_weights_amount_nz_index = None
        if self.use_pca:
            x_pred_logits = x['pred_logits']    # [n_batch, n_roi, 21]
            # # errorneous place to select det_class
            # if self.limit_det_class:
            #     x_pred_logits = x_pred_logits[:, :, self.used_det_class]
            x_pred_logits = torch.softmax(x_pred_logits, dim=2) # logit to probability
            # better place to select det_class
            if self.limit_det_class:
                x_pred_logits = x_pred_logits[:, :, self.used_det_class]    # n_b(16), roi(1200), 21(det) > ", ", 10(det_selected)

            x_amount_score = x['amount_score']

            s1, s2, s3 = x_amount_score.shape  # [n_batch, n_roi, 50]
            nz_index = x_amount_score.mean(dim=2).nonzero(
                as_tuple=True)  # along with class, [n_batch, 1200]
            x_amount_score = torch.softmax(x_amount_score, dim=2)  # logit to probability # [n_batch, n_roi, 50]

            if self.apply_one_value_amount:
                # select one value in a weighted sum way and scaled to 0 ~ 1.0
                n_div = 100 // x_amount_score.size(2)   # 100 // 50 > n_div = 2
                res_table = ((torch.arange(0, x_amount_score.size(2)) * n_div) + n_div / 2) * 0.01
                # res_table = [0.01, 0.03, 0.05, ... 0.97, 0.99] # [50]
                res_table = res_table.unsqueeze(0).unsqueeze(0) # [1, 1, 50]
                res_table = res_table.expand(s1, s2, s3)    # [s1, s2, 50]
                x_amount_score = torch.sum(x_amount_score * res_table.cuda(), dim=2, keepdim=True)  # [n_b, n_roi, 1]
                s1, s2, s3 = x_amount_score.shape  # [n_batch, n_roi, 50]

                # added at v3, meaningless amount becomes 1.0 instead of 0.5
                mask_x_amount_score = torch.zeros(x_amount_score.shape).type(torch.bool)
                mask_x_amount_score[nz_index] = True
                x_amount_score[~mask_x_amount_score] = 1.0

            if self.use_merge_det_amount:
                x_amount_score_nz = x_amount_score      # > [n_b, n_roi, 50]
            else:
                # every batch has same nonzero amount because 3 of 4 generates food amount, one stream dont
                x_amount_score_nz = x_amount_score[nz_index].view(s1, -1, s3)   # [n_b, nz_roi, 50]

            if self.apply_nms_on_pca:
                x_pred_bbox = x['pred_boxes']
                # x_amount_score_nz
                s1, s2, s3 = x_pred_logits.shape
                # nms_index_masks = torch.zeros((s1, s2))
                nms_index_masks = []
                for i_th in range(s3):
                    x_pred_logits_c = x_pred_logits[:, :, i_th].detach()

                    keep_indices = self.apply_nms(x_pred_bbox, x_pred_logits_c)
                    # nms_index_masks += keep_indices
                    nms_index_masks.append(keep_indices)

                nms_index_masks = torch.stack(nms_index_masks, dim=2)
                nms_index_masks = nms_index_masks.type(torch.bool)
                if not self.use_merge_det_amount:
                    nms_nz_index_masks = nms_index_masks[nz_index]
                    nms_nz_index_masks = nms_nz_index_masks.view(s1, -1, s3)
                # final keep_index_masks [n_b, 1200] inc. False or True
                # this will be used in final attnSimple at merge 1200 into 1.

            x_progress_score = x['progress_score']  # [n_batch, 3]
            x_progress_score = torch.softmax(x_progress_score, dim=1)   # [n_b, 3] # logit to probability

            if self.use_duration:
                x_duration = x['duration']      # 0~1, [n_b, 1]
            else:
                x_duration = None

            if self.pca_aggregate_type == 'maxpool':
                pred_logits_agg = x_pred_logits.max(dim=1)[0]
                amount_score_agg = x_amount_score_nz.max(dim=1)[0]
                if self.use_duration:
                    pca_feature = torch.cat([pred_logits_agg, amount_score_agg, x_progress_score, x_duration], dim=1)
                else:
                    pca_feature = torch.cat([pred_logits_agg, amount_score_agg, x_progress_score], dim=1)

            elif self.pca_aggregate_type == 'avgpool':
                pred_logits_agg = x_pred_logits.mean(dim=1)
                amount_score_agg = x_amount_score_nz.mean(dim=1)
                if self.use_duration:
                    pca_feature = torch.cat([pred_logits_agg, amount_score_agg, x_progress_score, x_duration],
                                            dim=1)
                else:
                    pca_feature = torch.cat([pred_logits_agg, amount_score_agg, x_progress_score], dim=1)

            elif self.pca_aggregate_type == 'attn_simple':
                # same shape and only reduce 0 on weight is the best
                if self.use_merge_det_amount:
                    if self.apply_nms_on_pca:
                        x_pred_logits[~nms_index_masks] = 0.
                        nms_index_masks_amount = nms_index_masks[:, :, self.amount_index].sum(
                            dim=2).type(torch.bool)
                        x_amount_score_nz[~nms_index_masks_amount] = 1.   # at v3: 0. -> 1.
                        x_det_amount = torch.cat([x_pred_logits, x_amount_score_nz], dim=2)
                        x_det_amount_weight_score = self.attn_pca_detclass_amount(
                            x_det_amount)  # [", ", n_det_class+amount] -> [n_batch, n_roi, n_class]
                        x_det_amount_weight = torch.softmax(x_det_amount_weight_score,
                                                            dim=1)  # [n_batch, n_roi, n_class]
                        # x_det_amount_weight = masked_softmax(x_det_amount_weight_score,
                        #                                      nms_index_masks, dim=1)  # [n_batch, n_roi, n_class]
                        # x_det_amount_weight[~nms_index_masks].max() == 0
                    else:
                        x_det_amount = torch.cat([x_pred_logits, x_amount_score_nz], dim=2)
                        x_det_amount_weight_score = self.attn_pca_detclass_amount(
                            x_det_amount)  # [", ", n_det_class+amount] -> [n_batch, n_roi, n_class]
                        x_det_amount_weight = torch.softmax(x_det_amount_weight_score, dim=1)  # [n_batch, n_roi, n_class]
                    x_det_amount_weight_swap = x_det_amount_weight.permute(0, 2,
                                                                             1)  # [n_b, n_class, n_roi]
                    pred_amount_agg = torch.matmul(x_det_amount_weight_swap,
                                                   x_det_amount)  # > [n_b, n_class(5), 21]
                    pred_amount_agg = pred_amount_agg.permute(1, 0, 2)  # [n_class(5), n_b, 21]
                else:
                    # [16, 5, sum_to_1]

                    if self.apply_nms_on_pca:
                        x_pred_logits[~nms_index_masks] = 0.
                        x_pred_logits_weight_score = self.attn_pca_detclass(x_pred_logits)  # [", ", n_det_class] -> [n_batch, n_roi, n_class]
                        x_pred_logits_weight = torch.softmax(x_pred_logits_weight_score,
                                                             dim=1)  # [n_batch, n_roi, n_class]
                        # x_pred_logits_weight = masked_softmax(x_pred_logits_weight_score, nms_index_masks, dim=1)  # [n_batch, n_roi, n_class]
                        # # x_pred_logits_weight[~nms_index_masks].max() == 0
                    else:
                        x_pred_logits_weight_score = self.attn_pca_detclass(
                            x_pred_logits)  # [", ", n_det_class] -> [n_batch, n_roi, n_class]
                        x_pred_logits_weight = torch.softmax(x_pred_logits_weight_score, dim=1) # [n_batch, n_roi, n_class]
                    x_pred_logits_weight_swap = x_pred_logits_weight.permute(0, 2, 1)   # [n_b, n_class, n_roi]
                    pred_logits_agg = torch.matmul(x_pred_logits_weight_swap, x_pred_logits)  # > [n_b, n_class(5), 21]
                    pred_logits_agg = pred_logits_agg.permute(1, 0, 2)    # [n_class(5), n_b, 21]

                    if self.apply_nms_on_pca:
                        nms_nz_index_masks = nms_nz_index_masks[:, :, self.amount_index].sum(dim=2).type(torch.bool)    # [n_b, 900, n_det_class(10)] > [n_b, 900]
                        x_amount_score_nz[~nms_nz_index_masks] = 1.     # at v3: 0. -> 1.
                        x_amount_score_weight_score = self.attn_pca_amount(
                            x_amount_score_nz)  # [n_b, nz_roi, 50] > [", ", n_class(5)]
                        # x_amount_score_weight = masked_softmax(x_amount_score_weight_score, nms_nz_index_masks, dim=1)
                        # # x_amount_score_weight[~nms_nz_index_masks].max() == 0
                        x_amount_score_weight = torch.softmax(x_amount_score_weight_score, dim=1)
                    else:
                        x_amount_score_weight_score = self.attn_pca_amount(
                            x_amount_score_nz)  # [n_b, nz_roi, 50] > [", ", n_class(5)]
                        x_amount_score_weight = torch.softmax(x_amount_score_weight_score, dim=1)
                    x_amount_score_weight_swap = x_amount_score_weight.permute(0, 2, 1) #
                    amount_score_agg = torch.matmul(x_amount_score_weight_swap, x_amount_score_nz)
                    amount_score_agg = amount_score_agg.permute(1, 0, 2)

                x_progress_score = x_progress_score.unsqueeze(dim=0).repeat(self.num_classes, 1, 1)     # [1, n_b, 3] > [n_class, n_b, 3]

                if self.use_duration:
                    x_duration = x_duration.unsqueeze(dim=0).repeat(self.num_classes, 1, 1)
                    if self.use_merge_det_amount:
                        pca_feature = torch.cat([pred_amount_agg, x_progress_score, x_duration], dim=2)
                    else:
                        pca_feature = torch.cat([pred_logits_agg, amount_score_agg, x_progress_score, x_duration],
                                                dim=2)
                else:
                    if self.use_merge_det_amount:
                        pca_feature = torch.cat([pred_amount_agg, x_progress_score], dim=2)
                    else:
                        pca_feature = torch.cat([pred_logits_agg, amount_score_agg, x_progress_score], dim=2)
                # [n_class, n_b, 21 + 50 + 3]

                # # weight map - start
                # # x_pred_logits_weight_swap # [n_b, n_class(5), n_roi(1200)]
                # # x_amount_score_weight_swap # [n_b, n_class, n_roi_nz(variable)]
                # # x_amount_score_weight_swap_orgsize # [n_b, n_c, n_roi]
                # x_amount_score_weight_swap_orgsize = torch.zeros(x_pred_logits_weight_swap.shape, device=x_pred_logits_weight_swap.device)
                # x_amount_score_weight_swap_orgsize = x_amount_score_weight_swap_orgsize.permute(0, 2, 1)    # [1, 1200, 5]
                # x_amount_score_weight_swap_orgsize[nz_index] = x_amount_score_weight[?] # [1, 900, 5]
                # x_amount_score_weight_swap_orgsize = x_amount_score_weight_swap_orgsize.permute(0, 2, 1) # [1, 5, 1200]
                # pca_weights_amount = x_amount_score_weight_swap_orgsize
                if self.use_merge_det_amount:
                    pca_weights_bbox = x_det_amount_weight_swap.detach() # [n_b, n_class, n_roi]
                else:
                    pca_weights_amount = x_amount_score_weight_swap.detach() # [n_b, n_class, n_roi_nz]
                    pca_weights_amount_nz_index = nz_index
                    pca_weights_bbox = x_pred_logits_weight_swap.detach()  # [n_b, n_class, n_roi]
                # weight map - stop

            list_features.append(pca_feature)

        backbone_weights = None
        if self.use_backbone:
            list_masked_srcs_weights = []
            list_masked_srcs_gpooled = []
            for l in range(self.lv_backbone):
                backbone_src_l = x['backbone_src'][l]
                backbone_mask_l = x['backbone_masks'][l]
                # srcs[l]: [n_b, 256, h, w]
                # masks[l]: [n_b, h, w], True == masking, False == Image

                if self.backbone_aggregate_type == 'avgpool':
                    masked_srcs_l = masked_mean(backbone_src_l, torch.unsqueeze(~backbone_mask_l, dim=1),
                                                dim=(2, 3))
                elif self.backbone_aggregate_type == 'maxpool':
                    masked_srcs_l = masked_max(backbone_src_l, torch.unsqueeze(~backbone_mask_l, dim=1))    # [n_batch, hidden_dim]
                elif self.backbone_aggregate_type == 'attn_qkv':
                    pass
                elif self.backbone_aggregate_type == 'attn_simple':
                    s1, s2, s3, s4 = backbone_src_l.shape   # backbone_src_l: [n_batch, 256, h, w]
                    backbone_src_l_3d = backbone_src_l.view(s1, s2, -1)
                    backbone_src_l_perm = backbone_src_l_3d.permute(0, 2, 1)  # n_b, h*w, 256
                    backbone_src_l_perm_weight_score = self.attn_backbone[l](backbone_src_l_perm)  # > [n_batch, h*w, 5]

                    backbone_mask_l_vec = torch.unsqueeze(backbone_mask_l.view(s1, -1), dim=2)  # [n_b, h, w] > [n_b, h*w] > [n_b, h*w, 1]
                    backbone_mask_l_vec = backbone_mask_l_vec.repeat(1, 1, self.num_classes)    # [n_b, h*w, 1] > [n_b, h*w, 5]
                    backbone_src_l_perm_weight = masked_softmax(backbone_src_l_perm_weight_score,   # [n_batch, h*w, 5]
                                                                ~backbone_mask_l_vec, dim=1)        # [n_b, h*w, 5]

                    # check below cal results and dimension
                    backbone_src_l_perm_weight_swap = backbone_src_l_perm_weight.permute(0, 2, 1)   # [n_b, 5, h*w]
                    backbone_src_l_pool = torch.matmul(backbone_src_l_perm_weight_swap, backbone_src_l_perm)  # [n_batch, 5, 256]
                    backbone_src_l_pool = backbone_src_l_pool.permute(1, 0, 2)  # [5, n_b, 256]
                    masked_srcs_l = backbone_src_l_pool

                    list_masked_srcs_weights.append(backbone_src_l_perm_weight_swap)

                list_masked_srcs_gpooled.append(masked_srcs_l)

            if self.backbone_aggregate_type == 'avgpool' or self.backbone_aggregate_type == 'maxpool':
                masked_srcs_gpooled = torch.cat(list_masked_srcs_gpooled, dim=1)  # [n_b, 256 * 4 = 1024]
                if self.apply_btl_backbone:
                    masked_srcs_gpooled = self.backbone_btl(masked_srcs_gpooled)
            elif self.backbone_aggregate_type == 'attn_qkv' or self.backbone_aggregate_type == 'attn_simple':
                masked_srcs_gpooled = torch.cat(list_masked_srcs_gpooled, dim=2)  # [5, n_b, 256*n_lv]
                if self.apply_btl_backbone:
                    list_btl_masked_srcs_gpooled = []
                    for i_batch in range(masked_srcs_gpooled.shape[0]):
                        btl_masked_srcs_gpooled = self.backbone_btl[i_batch](masked_srcs_gpooled[i_batch,:,:])
                        list_btl_masked_srcs_gpooled.append(btl_masked_srcs_gpooled)
                    masked_srcs_gpooled = torch.stack(list_btl_masked_srcs_gpooled, dim=0)
                backbone_weights = torch.cat(list_masked_srcs_weights, dim=2)  # [n_b, 5, 256*n_lv]
            else:
                raise AssertionError('unsupported!')
            list_features.append(masked_srcs_gpooled)

        hs_output_weights = None
        if self.use_roi:
            # x['hs_last']              # [n_batch, n_trfm(4), n_query(300), 256]
            s1, s2, s3, s4 = x['hs_last'].shape     # [n_batch, n_trfm(4), n_query(300), 256]
            # x_hs_last = x['hs_last'].view(s1, s2*s3, s4)    # [n_batch, n_trfm(4)*n_query(300), 256]
            x_hs_last = x['hs_last']    # [n_batch, n_trfm(4), n_query(300), 256]

            if self.aggregate_type == 'maxpool':
                x_hs_last_pool, _index = x_hs_last.max(dim=2)   # [n_batch, n_trfm(4), 256]
                x_hs_last_pool = x_hs_last_pool.view(s1, s2*s4)
            elif self.aggregate_type == 'avgpool':
                x_hs_last_pool = x_hs_last.mean(dim=2)
                x_hs_last_pool = x_hs_last_pool.view(s1, s2*s4)
            elif self.aggregate_type == 'attn_qkv':
                # concat x_hs_last_pool,
                # how to deal with hs_output_weights
                trfm_x_hs_last_pool = []
                trfm_hs_output_weights = []
                for i_trfm in range(self.num_trfm):
                    sac_embed_roi = self.sac_embed_roi[i_trfm].weight.unsqueeze(0).repeat(s1, 1, 1)     # [n_class, 256] > [n_b, n_class, 256]
                    # input order: seq_len, n_batch, embedding_dim
                    x_hs_last_pool, hs_output_weights = self.attn_roi[i_trfm](query=sac_embed_roi.transpose(0, 1),
                                                                 key=x_hs_last[:, i_trfm, :, :].transpose(0, 1),
                                                                 value=x_hs_last[:, i_trfm, :, :].transpose(0, 1))
                    trfm_x_hs_last_pool.append(x_hs_last_pool)
                    trfm_hs_output_weights.append(hs_output_weights)

                # x_hs_last_pool.shape     # [5, 16, 256]
                # hs_output_weights.shape # [16, 5, 1200].sum(dim=2) = 1.0000

                x_hs_last_pool = torch.cat(trfm_x_hs_last_pool, dim=2)  # [5, 16, 256*n_trfm]
                hs_output_weights = torch.cat(trfm_hs_output_weights, dim=2)    # [16, 5, 300*n_trfm]
            elif self.aggregate_type == 'attn_simple':
                trfm_x_hs_last_pool = []
                trfm_hs_output_weights = []
                for i_trfm in range(self.num_trfm):
                    x_hs_last_weight_score = self.attn_roi[i_trfm](x_hs_last[:, i_trfm, :, :])   # [n_batch, n_trfm(4), n_query(300), 256] > [n_batch, n_query, 256] > [n_batch, n_query, 5]
                    hs_output_weights = torch.softmax(x_hs_last_weight_score, dim=1)    # [n_batch, n_query, 5]

                    # check below cal results and dimension
                    # x_hs_last[:, i_trfm, :, :]: [n_batch, n_query, 256]
                    x_hs_last_temp = x_hs_last[:, i_trfm, :, :]       # [n_batch, n_query, 256]
                    hs_output_weights_swap = hs_output_weights.permute(0, 2, 1)
                    # hs_output_weights_swap:     [n_batch, 5, n_query]
                    x_hs_last_pool = torch.matmul(hs_output_weights_swap, x_hs_last_temp)     # [n_batch, 5, 256]
                    x_hs_last_pool = x_hs_last_pool.permute(1, 0, 2)

                    trfm_x_hs_last_pool.append(x_hs_last_pool)  # [5, n_batch, 256]
                    trfm_hs_output_weights.append(hs_output_weights_swap)   # [n_batch, 5, n_query]

                x_hs_last_pool = torch.cat(trfm_x_hs_last_pool, dim=2)  # [5, 16, 256*n_trfm]
                hs_output_weights = torch.cat(trfm_hs_output_weights, dim=2)  # [16, 5, 300*n_trfm]
            else:
                raise AssertionError('Unsupported aggreate_type: ', self.aggregate_type)

            if self.apply_btl_roi:
                if self.aggregate_type == 'maxpool' or self.aggregate_type == 'avgpool':
                    x_hs_last_pool = self.roi_btl(x_hs_last_pool)
                elif self.aggregate_type == 'attn_qkv' or self.aggregate_type == 'attn_simple':
                    list_btl_x_hs_last_pool = []
                    for i_batch in range(x_hs_last_pool.shape[0]):
                        btl_x_hs_last_pool = self.roi_btl[i_batch](x_hs_last_pool[i_batch,:,:]) # [16, 1024] > [16, 64]
                        list_btl_x_hs_last_pool.append(btl_x_hs_last_pool)
                    x_hs_last_pool = torch.stack(list_btl_x_hs_last_pool, dim=0)

            list_features.append(x_hs_last_pool)

        enc_output_weights = None
        if self.use_encoder:
            # x['enc_memory']           # [n_batch, n_trfm(4), n_spatial_pixels(17867), 256]
            s1, s2, s3, s4 = x[
                'enc_memory'].shape  # [n_batch, n_trfm(4), n_spatial_pixels(17867), 256]
            enc_memory = x['enc_memory']  # [n_b, n_trfm(4), s3(71,468), 256]

            if self.aggregate_type == 'maxpool':
                enc_memory_pool, _index = enc_memory.max(dim=2)  # [n_b, n_trfm(4), 256]
                enc_memory_pool = enc_memory_pool.view(s1, s2 * s4)
            elif self.aggregate_type == 'avgpool':
                enc_memory_pool = enc_memory.mean(dim=2)
                enc_memory_pool = enc_memory_pool.view(s1, s2 * s4)
            elif self.aggregate_type == 'attn_qkv':
                trfm_enc_memory_pool = []
                trfm_enc_output_weights = []
                for i_trfm in range(self.num_trfm):
                    sac_embed_encoder = self.sac_embed_encoder[i_trfm].weight.unsqueeze(0).repeat(
                        s1, 1, 1)  # [n_class, 256] > [n_b, n_class, 256]
                    enc_memory_pool, enc_output_weights = self.attn_encoder[i_trfm](
                        query=sac_embed_encoder.transpose(0, 1),
                        key=enc_memory[:, i_trfm, :, :].transpose(0, 1),
                        value=enc_memory[:, i_trfm, :, :].transpose(0, 1))
                    trfm_enc_memory_pool.append(enc_memory_pool)
                    trfm_enc_output_weights.append(enc_output_weights)

                enc_memory_pool = torch.cat(trfm_enc_memory_pool, dim=2)  # [5, 16, 256*n_trfm]
                enc_output_weights = torch.cat(trfm_enc_output_weights,
                                               dim=2)  # [16, 5, 300*n_trfm]
            elif self.aggregate_type == 'attn_simple':
                trfm_enc_memory_pool = []
                trfm_enc_output_weights = []
                for i_trfm in range(self.num_trfm):
                    enc_memory_weight_score = self.attn_encoder[i_trfm](enc_memory[:, i_trfm, :, :])
                    enc_output_weights = torch.softmax(enc_memory_weight_score,
                                                       dim=1)  # [n_batch, n_query, 5]

                    enc_memory_temp = enc_memory[:, i_trfm, :, :]  # [n_batch, n_query, 256]
                    enc_output_weights_swap = enc_output_weights.permute(0, 2, 1)
                    enc_memory_pool = torch.matmul(enc_output_weights_swap,
                                                   enc_memory_temp)  # [n_batch, 5, 256]
                    enc_memory_pool = enc_memory_pool.permute(1, 0, 2)

                    trfm_enc_memory_pool.append(enc_memory_pool)  # [5, n_batch, 256]
                    trfm_enc_output_weights.append(enc_output_weights_swap)  # [n_batch, 5, n_query]

                enc_memory_pool = torch.cat(trfm_enc_memory_pool, dim=2)  # [5, 16, 256*n_trfm]
                enc_output_weights = torch.cat(trfm_enc_output_weights,
                                               dim=2)  # [16, 5, 300*n_trfm]
            else:
                raise AssertionError('Unsupported aggreate_type: ', self.aggregate_type)

            if self.apply_btl_encoder:
                if self.aggregate_type == 'maxpool' or self.aggregate_type == 'avgpool':
                    enc_memory_pool = self.encoder_btl(enc_memory_pool)
                elif self.aggregate_type == 'attn_qkv' or self.aggregate_type == 'attn_simple':
                    list_btl_enc_memory_pool = []
                    for i_batch in range(enc_memory_pool.shape[0]):
                        btl_enc_memory_pool = self.encoder_btl[i_batch](enc_memory_pool[i_batch,:,:])
                        list_btl_enc_memory_pool.append(btl_enc_memory_pool)
                    enc_memory_pool = torch.stack(list_btl_enc_memory_pool, dim=0)

            list_features.append(enc_memory_pool)


        # head classifier
        if self.aggregate_type == 'attn_qkv' or self.aggregate_type == 'attn_simple':
            # concat + attn + classifier
            list_output_scores = []
            for i in range(self.num_classes):
                list_attn_features = []
                if self.use_pca:
                    if self.pca_aggregate_type == 'avgpool' or self.pca_aggregate_type == 'maxpool':
                        list_attn_features.append(pca_feature)  # [n_batch, n_dim]
                    else:
                        list_attn_features.append(pca_feature[i, :, :])  # [n_class(5), n_batch(16), n_dim(74)]
                if self.use_backbone:
                    if self.backbone_aggregate_type == 'avgpool' or self.backbone_aggregate_type == 'maxpool':
                        list_attn_features.append(masked_srcs_gpooled)   # n_batch, 1024 == x['backbone_features']
                    else:
                        list_attn_features.append(masked_srcs_gpooled[i, :, :]) # [5, n_b, 256*n_lv] > [n_b, 256*n_lv]
                if self.use_roi:
                    list_attn_features.append(x_hs_last_pool[i, :, :])  # [n_class(i), n_batch, 256] > [n_batch, 256]
                if self.use_encoder:
                    list_attn_features.append(enc_memory_pool[i, :, :]) # n_batch, 256

                x_cat = torch.cat(list_attn_features, dim=1)  # [n_batch, n_hid1+n_hid2+...]
                x_cat = self.tbinder[i](x_cat)
                list_output_scores.append(self.service_class_embed[i](x_cat))   # n_batch, 1

            output_scores = torch.cat(list_output_scores,
                                      dim=1)  # list of [n_batch, 1] -> [n_batch, n_class]
        else:
            # concat + classifier
            x_cat = torch.cat(list_features, dim=1)     # [n_b, n_hid1], [n_b, n_hid2], => [n_b, n_hid1+n_hid2]
            x_cat = self.tbinder(x_cat) # [n_b, n_classes -> 1, n_classes or bypass]

            if 'attn' in self.tbinder.tbinder_type:
                list_output_scores = []
                for i in range(self.num_classes):
                    list_output_scores.append(self.service_class_embed[i](x_cat[:, i, :]))  # [1, n_class, 1024]
                output_scores = torch.cat(list_output_scores, dim=1)  # list of [n_batch, 1] -> [n_batch, n_class]
            else:
                # check dimension
                output_scores = self.service_class_embed(x_cat)     # [n_b, n_classes]

        out = {'service_pred_logits': output_scores,
               'pca_weights_bbox': pca_weights_bbox,
               'pca_weights_amount': pca_weights_amount,
               'pca_weights_amount_nz_index': pca_weights_amount_nz_index,
               'backbone_weights': backbone_weights,
               'hs_output_weights': hs_output_weights,
               'enc_output_weights': enc_output_weights}       # [n_batch, n_classes] (min:0.0001, max: 1.000)

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

        results_sac = {}
        if 'amount_score' in outputsG.keys():
            results_sac['amount_score'] = outputsG['amount_score'][:, topk_boxes[0], :]  # [1, 100, 50]

        service_pred_logits = outputsH['service_pred_logits']
        n_sac_classes = service_pred_logits.shape[1]    # 5

        hs_output_weights = outputsH['hs_output_weights']  # [n_b, n_class(5), 1200]
        if hs_output_weights is not None:
            results_sac['hs_attn_values'], results_sac['hs_attn_bbox'] = self.get_weightedbbox_from_weight(hs_output_weights, n_sac_classes, boxes_xyxy, scale_fct,
                                         topk=3)

        # hs_output_weights = outputsH['hs_output_weights']  # [n_b, n_class(5), 1200]
        # list_hs_attn_values = []
        # # list_hs_attn_bbox_indexes = []
        # list_hs_boxes = []
        # if hs_output_weights is not None:
        #     for i_c in range(1, n_sac_classes):
        #         hs_attn_values, hs_attn_bbox_indexes = torch.topk(hs_output_weights[:, i_c, :], 3, dim=1)    # [n_b, 1200]
        #         boxes_hs = torch.gather(boxes_xyxy, 1, hs_attn_bbox_indexes.unsqueeze(-1).repeat(1, 1, 4))
        #
        #         list_hs_attn_values.append(hs_attn_values.unsqueeze(1))
        #         # list_hs_attn_bbox_indexes.append(hs_attn_bbox_indexes.unsqueeze(1))
        #         list_hs_boxes.append(boxes_hs.unsqueeze(1))
        #
        #     results_sac['hs_attn_values'] = torch.cat(list_hs_attn_values, dim=1)
        #     results_sac['hs_attn_bbox'] = torch.cat(list_hs_boxes, dim=1) * scale_fct[:, None, None, :]  # n_batch, n_class-1, topk, 4(xyxyx)


        enc_output_weights = outputsH['enc_output_weights']
        if enc_output_weights is not None:
            results_sac['enc_attn_values'], results_sac['enc_attn_bbox'] = self.get_weightedbbox_from_weight(enc_output_weights, n_sac_classes, boxes_xyxy, scale_fct,
                                         topk=3)

        # enc_output_weights = outputsH['enc_output_weights']
        # list_enc_attn_values = []
        # # list_enc_attn_bbox_indexes = []
        # list_enc_boxes = []
        # if enc_output_weights is not None:
        #     for i_c in range(1, n_sac_classes):
        #         enc_attn_values, enc_attn_bbox_indexes = torch.topk(enc_output_weights[:, i_c, :], 3, dim=1)
        #         boxes_enc = torch.gather(boxes_xyxy, 1, enc_attn_bbox_indexes.unsqueeze(-1).repeat(1, 1, 4))
        #
        #         list_enc_attn_values.append(enc_attn_values.unsqueeze(1))
        #         # list_enc_attn_bbox_indexes.append(enc_attn_bbox_indexes.unsqueeze(1))
        #         list_enc_boxes.append(boxes_enc.unsqueeze(1))
        #
        #     results_sac['enc_attn_values'] = torch.cat(list_enc_attn_values, dim=1)
        #     results_sac['enc_attn_bbox'] = torch.cat(list_enc_boxes, dim=1) * scale_fct[:, None, None, :]

        pca_bbox_output_weights = outputsH['pca_weights_bbox']    # [1, 5, 1200]
        if pca_bbox_output_weights is not None:
            results_sac['pca_bbox_attn_values'], results_sac['pca_bbox_attn_bbox'] = self.get_weightedbbox_from_weight(pca_bbox_output_weights, n_sac_classes, boxes_xyxy, scale_fct,
                                         topk=3)

        pca_amount_output_weights = outputsH['pca_weights_amount']  # [1, 5, 900]
        if pca_amount_output_weights is not None:
            s1, s2, s3 = boxes_xyxy.shape
            nz_index = outputsH['pca_weights_amount_nz_index']  # tuple w/ 2 elements
            boxes_xyxy_nz = boxes_xyxy[nz_index].view(s1, -1, s3)

            results_sac['pca_amount_attn_values'], results_sac['pca_amount_attn_bbox'] = self.get_weightedbbox_from_weight(
                pca_amount_output_weights, n_sac_classes, boxes_xyxy_nz, scale_fct,
                topk=3)

        # outputsH
        # 'enc_output_weights' # encoder
        # 'hs_output_weights' # ROI

        return results, results_sac

    @torch.no_grad()
    def get_weightedbbox_from_weight(self, input_weights, n_sac_classes, boxes_xyxy, scale_fct, topk=3):
        list_attn_values = []
        list_boxes = []

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

    # if saclassifier_type == 'imagebased':
    #     hidden_dim = 1024
    #     model = ImageBasedSAClassifier(hidden_dim, num_classes)

    # elif saclassifier_type == 'roibased':
    #     hidden_dim = 256 + 1024
    #     model = ROIBasedSAClassifier(hidden_dim, num_classes)
    #
    # elif saclassifier_type == 'roiattbased':
    #     hidden_dim = 256 + 1024
    #     model = ROIAttBasedSAClassifier(hidden_dim, num_classes)

    if saclassifier_type == 'imageavgp_roiv2attnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgp_pcaavgp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True,
                                                    aggregate_type='avgpool')

    elif saclassifier_type == 'imageavgp_pcaattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgp_pcadattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    aggregate_type='attn_simple')
        # postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgp_pcadlcattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgp_pcadlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgp_pca1dlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    apply_one_value_amount=True,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'T5maxp_imageavgp_pca1dlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    apply_one_value_amount=True,
                                                    aggregate_type='attn_simple',
                                                    tbinder_type='maxpool')
        # postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'T5avgp_imageavgp_pca1dlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    apply_one_value_amount=True,
                                                    aggregate_type='attn_simple',
                                                    tbinder_type='avgpool')
        # postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'T5attnsimple_imageavgp_pca1dlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    apply_one_value_amount=True,
                                                    aggregate_type='attn_simple',
                                                    tbinder_type='attn_simple')
        # postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgp_roiv2attnsimplebtl_pca1dlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False,
                                                    use_roi=True, apply_btl_roi=True,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    apply_one_value_amount=True,
                                                    aggregate_type='attn_simple')
        # postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgp_roiv2attnsimple_pca1dlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False,
                                                    use_roi=True,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    apply_one_value_amount=True,
                                                    aggregate_type='attn_simple')
        # postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgpbtl_pca1dlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    apply_btl_backbone=True,
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_nms_on_pca=True,
                                                    apply_one_value_amount=True,
                                                    aggregate_type='attn_simple')
        # postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgp_pcadlcmgattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, use_merge_det_amount=True,
                                                    aggregate_type='attn_simple')
        # postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgp_pcadlcmgnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, use_merge_det_amount=True,
                                                    apply_nms_on_pca=True,
                                                    aggregate_type='attn_simple')
        # postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageavgp_roiv2attnsimple_pcaattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    use_pca=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgp_roiv2attnsimple_pcadattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    use_pca=True, use_duration=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgp_roiv2attnsimple_pcadlcattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgpbtl_roiv2attnsimplebtl_pca1dlcnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm,
                                                    use_backbone=True, apply_btl_backbone=True, backbone_aggregate_type='avgpool',
                                                    use_encoder=False,
                                                    use_roi=True, apply_btl_roi=True,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_one_value_amount=True,
                                                    apply_nms_on_pca=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgpbtl_roiv2attnsimplebtl_pca1dlcmgnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm,
                                                    use_backbone=True, apply_btl_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False,
                                                    use_roi=True, apply_btl_roi=True,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_one_value_amount=True,
                                                    apply_nms_on_pca=True, use_merge_det_amount=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgp_roiv2attnsimplebtl_pca1dlcmgnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm,
                                                    use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False,
                                                    use_roi=True, apply_btl_roi=True,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_one_value_amount=True,
                                                    apply_nms_on_pca=True, use_merge_det_amount=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgp_roiv2attnsimple_pca1dlcmgnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm,
                                                    use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False,
                                                    use_roi=True,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_one_value_amount=True,
                                                    apply_nms_on_pca=True, use_merge_det_amount=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgp_pca1dlcmgnmsattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm,
                                                    use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, apply_one_value_amount=True,
                                                    apply_nms_on_pca=True, use_merge_det_amount=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgp_roiv2attnsimple_pcadlcmgattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    use_pca=True, use_duration=True,
                                                    limit_det_class=True, use_merge_det_amount=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imageavgp_roiv2attnsimple_pcaavgp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='attn_simple',
                                                    use_pca=True,
                                                    pca_aggregate_type='avgpool')

    elif saclassifier_type == 'pcaavgp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True,
                                                    aggregate_type='avgpool')

    elif saclassifier_type == 'pcadattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    use_pca=True, use_duration=True,
                                                    aggregate_type='attn_simple')

    elif saclassifier_type == 'imagemaxp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='maxpool')

    elif saclassifier_type == 'imageavgp_MLP2':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='avgpool',
                                                    final_layer_type='MLP', num_MLP_final_layers=2)

    elif saclassifier_type == 'imageavgp_MLP3':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='avgpool',
                                                    final_layer_type='MLP', num_MLP_final_layers=3)

    elif saclassifier_type == 'imageavgp_roiv2attnsimple_MLP2':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='attn_simple',
                                                    final_layer_type='MLP', num_MLP_final_layers=2)

    elif saclassifier_type == 'imageavgp_roiv2attnsimple_MLP3':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='attn_simple',
                                                    final_layer_type='MLP', num_MLP_final_layers=3)

    elif saclassifier_type == 'imageattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageT5':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='avgpool',
                                                    tbinder_type='maxpool', num_temporal=5)

    elif saclassifier_type == 'T5_imageavgp_roiv2attnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='attn_simple',
                                                    tbinder_type='avgpool', num_temporal=5)

    elif saclassifier_type == 'T5maxp_pcaattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='attn_simple',
                                                    use_pca=True,
                                                    pca_aggregate_type='attn_simple',
                                                    tbinder_type='maxpool', num_temporal=5)

    elif saclassifier_type == 'T5maxp_imageavgp_roiv2attnsimple_pcaavgp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='attn_simple',
                                                    use_pca=True,
                                                    pca_aggregate_type='avgpool',
                                                    tbinder_type='maxpool', num_temporal=5)

    elif saclassifier_type == 'T5maxp_imageavgp_pcadattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='attn_simple',
                                                    use_pca=True, use_duration=True,
                                                    pca_aggregate_type='attn_simple',
                                                    tbinder_type='maxpool', num_temporal=5)



    elif saclassifier_type == 'T5avgp_imageavgp_pcadattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='attn_simple',
                                                    use_pca=True, use_duration=True,
                                                    pca_aggregate_type='attn_simple',
                                                    tbinder_type='avgpool', num_temporal=5)

    elif saclassifier_type == 'T5attnsimple_imageavgp_pcadattnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    backbone_aggregate_type='avgpool',
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='attn_simple',
                                                    use_pca=True, use_duration=True,
                                                    pca_aggregate_type='attn_simple',
                                                    tbinder_type='attn_simple', num_temporal=5)

    elif saclassifier_type == 'imageT5avgp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='avgpool',
                                                    tbinder_type='avgpool', num_temporal=5)
    elif saclassifier_type == 'imageT5attnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='avgpool',
                                                    tbinder_type='attn_simple', num_temporal=5)

    elif saclassifier_type == 'imageT5tcn':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=False, use_roi=False,
                                                    aggregate_type='avgpool',
                                                    tbinder_type='tcn', num_temporal=5)

    elif saclassifier_type == 'roiv2maxp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='maxpool')

    elif saclassifier_type == 'roiv2avgp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='avgpool')

    elif saclassifier_type == 'roiv2attnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'roiv2attnqkv':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='attn_qkv')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'encv2maxp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    use_encoder=True, use_roi=False,
                                                    aggregate_type='maxpool')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'encv2avgp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    use_encoder=True, use_roi=False,
                                                    aggregate_type='avgpool')

    elif saclassifier_type == 'encv2attnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    use_encoder=True, use_roi=False,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'encv2attnqkv':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=False,
                                                    use_encoder=True, use_roi=False,
                                                    aggregate_type='attn_qkv')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageroiv2attnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=False, use_roi=True,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageencv2attnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=True, use_roi=False,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageroiv2attnsimpleencv2attnsimple':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=True, use_roi=True,
                                                    aggregate_type='attn_simple')
        postprocessors['bbox_attn'] = PostProcess()

    elif saclassifier_type == 'imageroiv2maxpencv2maxp':
        model = ImageEncoderROIAttBasedSAClassifier(num_classes, num_trfm, use_backbone=True,
                                                    use_encoder=True, use_roi=True,
                                                    aggregate_type='maxpool')
        postprocessors['bbox_attn'] = PostProcess()

    #
    # elif saclassifier_type == 'imageencmaxp':
    #     model = ImageEncoderROIAttBasedSAClassifier(num_classes, use_backbone=True,
    #                                                 use_encoder=True, use_roi=False,
    #                                                 aggregate_type='maxpool')
    else:
        raise AssertionError(f'{saclassifier_type} is undefined!')

    # postprocessors['bbox'] = PostProcessBBox()
    # postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
