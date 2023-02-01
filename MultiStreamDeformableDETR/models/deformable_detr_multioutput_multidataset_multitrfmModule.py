# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .backbone_uda import build_backbone as build_backbone_uda
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss, sigmoid_focal_loss_asis)
from .deformable_transformer import build_deforamble_transformer
import copy
import pdb

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# https://www.codefull.net/2020/03/masked-tensor-operations-in-pytorch/
def masked_mean(tensor, mask, dim):
    masked = torch.mul(tensor, mask)  # Apply the mask using an element-wise multiplication
    return masked.sum(dim=dim) / mask.sum(dim=dim)  # Find the average!

class DeformableDETRafterBackbone(nn.Module):
    """ This is the Deformable DETR module w/o backbone module """

    def __init__(self, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, use_progress=False,
                 use_amount=False, amount_type=0, n_amount_class=50):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.use_amount = use_amount
        self.use_progress = use_progress
        self.amount_type = amount_type
        self.n_amount_class = n_amount_class

        if self.use_amount:
            if self.amount_type == 0:
                # 1-layer
                self.food_amount_embed = nn.Linear(hidden_dim, 1)
            elif self.amount_type == 1:
                # 2-layers
                self.food_amount_embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                       nn.ReLU(),
                                                       nn.Linear(hidden_dim, 1))
            elif self.amount_type == 2:
                # 2-layers + 5 mixtures
                self.n_amount_sigmoid = 5
                self.food_amount_embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                       nn.ReLU(),
                                                       nn.Linear(hidden_dim, self.n_amount_sigmoid))
                self.food_amount_embed_weight = nn.Linear(hidden_dim, self.n_amount_sigmoid)
            elif self.amount_type == 3:
                # 1-layers + 5 mixtures
                self.n_amount_sigmoid = 5
                self.food_amount_embed = nn.Linear(hidden_dim, self.n_amount_sigmoid)
                self.food_amount_embed_weight = nn.Linear(hidden_dim, self.n_amount_sigmoid)
            elif self.amount_type == 4:
                # 1-layers + 5 mixtures
                self.n_amount_sigmoid = 5
                self.food_amount_embed = nn.Linear(hidden_dim + num_classes, self.n_amount_sigmoid)
                self.food_amount_embed_weight = nn.Linear(hidden_dim, self.n_amount_sigmoid)
            elif self.amount_type == 5:
                # 1-layers + 5 mixtures
                self.n_amount_sigmoid = 5
                self.food_amount_embed = nn.Linear(hidden_dim + num_classes + hidden_dim,
                                                   self.n_amount_sigmoid)
                self.food_amount_embed_weight = nn.Linear(hidden_dim, self.n_amount_sigmoid)
            elif self.amount_type == 6:
                print('classification-typed amount model with %d classes' % self.n_amount_class)
                self.food_amount_embed = nn.Linear(hidden_dim, self.n_amount_class)
            else:
                raise AssertionError('{} is unsupported'.format(self.amount_type))

        if self.use_progress:
            self.table_progress_embed = nn.Linear(hidden_dim * self.num_feature_levels, 3)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if self.use_amount:
            if self.amount_type == 0 or self.amount_type == 6:
                self.food_amount_embed.bias.data = torch.ones(
                    self.food_amount_embed.out_features) * bias_value
            elif self.amount_type == 1:
                self.food_amount_embed[0].bias.data = torch.ones(
                    self.food_amount_embed[0].out_features) * bias_value
                self.food_amount_embed[2].bias.data = torch.ones(
                    self.food_amount_embed[2].out_features) * bias_value
            elif self.amount_type == 2:
                self.food_amount_embed[0].bias.data = torch.ones(
                    self.food_amount_embed[0].out_features) * bias_value
                self.food_amount_embed[2].bias.data = torch.ones(
                    self.food_amount_embed[2].out_features) * bias_value
                self.food_amount_embed_weight.bias.data = torch.ones(
                    self.food_amount_embed_weight.out_features) * bias_value
            elif self.amount_type == 3 or self.amount_type == 4 or self.amount_type == 5:
                self.food_amount_embed.bias.data = torch.ones(
                    self.food_amount_embed.out_features) * bias_value
                self.food_amount_embed_weight.bias.data = torch.ones(
                    self.food_amount_embed_weight.out_features) * bias_value

        if self.use_progress:
            self.table_progress_embed.bias.data = torch.ones(
                self.table_progress_embed.out_features) * bias_value

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            # def _get_clones(module, N):
            #     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, srcs, masks, pos,
                targets_for_amount=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if self.use_progress:
            # masked global pool
            # src: list of 4,
            list_masked_srcs_gpooled = []
            for l in range(self.num_feature_levels):
                # area (mask == False) has real image value
                masked_srcs_l = masked_mean(srcs[l], torch.unsqueeze(~masks[l], dim=1), dim=(2, 3))
                list_masked_srcs_gpooled.append(masked_srcs_l)

            masked_srcs_gpooled = torch.cat(list_masked_srcs_gpooled, dim=1)

            # apply nn.progress
            progress_score = self.table_progress_embed(masked_srcs_gpooled)
            # progress_pred = F.softmax(progress_score, 1)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
            # pdb.set_trace() query_embed.weight: [300(n_query), 512]

        hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord_unact, \
        enc_memory = self.transformer(srcs, masks, pos, query_embeds,
                                      targets_for_amount=targets_for_amount)

        outputs_classes = []
        outputs_coords = []
        # hs: [num_dec_layers, num_batch, num_query, hid_dim]
        # init_reference: [num_batch, num_query, 2]
        #               : (--two_stage on) [num_batch, num_query, 4]
        # inter_references: [num_dec_layers, num_batch_num_query, hid_dim]
        for lvl in range(hs.shape[0]):  # for each decoding layers
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]  # just see the keywords in references
            reference = inverse_sigmoid(reference)  # 0~1 -> real-value
            outputs_class = self.class_embed[lvl](hs[lvl])  # each dec_layer has its outputs for aux loss
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'hs_last': hs[-1], 'enc_memory': enc_memory}
        # hs_last: [n_batch, n_query, 256]
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class,
                                  'pred_boxes': enc_outputs_coord}  # predicted region proposals

        if self.use_progress:
            out['progress_score'] = progress_score
            # out['progress_pred'] = progress_pred

        if self.use_amount:
            if self.amount_type == 0 or self.amount_type == 1:
                amount_score = self.food_amount_embed(hs[-1])  # [n_batch, n_query, 256] -> [n_batch, n_query, 1]
                amount_pred = torch.sigmoid(amount_score)
                # out['amount_score'] = amount_score
                out['amount_pred'] = amount_pred
            elif self.amount_type == 2 or self.amount_type == 3:
                amount_mix_scores = self.food_amount_embed(
                    hs[-1])  # [n_batch, n_query, 256] -> [n_batch, n_query, n_sigmoid]
                amount_mix_preds = torch.sigmoid(amount_mix_scores)  # [n_batch, n_query, n_sigmoid]
                amount_mix_weights = torch.softmax(self.food_amount_embed_weight(hs[-1]),
                                                   dim=2)  # -> [n_batch, n_query, n_sigmoid]
                amount_pred = torch.sum(amount_mix_preds * amount_mix_weights, dim=2, keepdim=True)
                out['amount_pred'] = amount_pred
            elif self.amount_type == 4:
                hs_outputs = torch.cat((hs[-1], outputs_class[-1]), dim=2)  # [n_batch, n_query, 256 + n_class]
                amount_mix_scores = self.food_amount_embed(hs_outputs)  # -> [n_batch, n_query, n_sigmoid]
                amount_mix_preds = torch.sigmoid(amount_mix_scores)  # [n_batch, n_query, n_sigmoid]
                amount_mix_weights = torch.softmax(self.food_amount_embed_weight(hs[-1]),
                                                   dim=2)  # -> [n_batch, n_query, n_sigmoid]
                amount_pred = torch.sum(amount_mix_preds * amount_mix_weights, dim=2, keepdim=True)
                out['amount_pred'] = amount_pred
            elif self.amount_type == 5:
                # torch.expand: repeat tensor with size 1 w/o extra memory
                masked_srcs_last = masked_mean(srcs[-1], torch.unsqueeze(~masks[-1], dim=1), dim=(2, 3))  # 4, 256
                # masked_srcs_gpooled: all layers, [n_batch, 1024]
                masked_srcs_gpooled_exp = torch.unsqueeze(masked_srcs_last, dim=1)
                masked_srcs_gpooled_exp = masked_srcs_gpooled_exp.expand((-1, hs[-1].size(1), -1))
                # [n_batch, 1024] -> [n_batch, n_query, 1024]
                hs_outputs = torch.cat((hs[-1], outputs_class[-1], masked_srcs_gpooled_exp),
                                       dim=2)  # [n_batch, n_query, 256 + n_class + 256]
                amount_mix_scores = self.food_amount_embed(hs_outputs)  # -> [n_batch, n_query, n_sigmoid]
                amount_mix_preds = torch.sigmoid(amount_mix_scores)  # [n_batch, n_query, n_sigmoid]
                amount_mix_weights = torch.softmax(self.food_amount_embed_weight(hs[-1]),
                                                   dim=2)  # -> [n_batch, n_query, n_sigmoid]
                amount_pred = torch.sum(amount_mix_preds * amount_mix_weights, dim=2, keepdim=True)
                out['amount_pred'] = amount_pred
                # print(torch.argmax(amount_mix_weights, dim=2))
                # pdb.set_trace()
            elif self.amount_type == 6:
                amount_score = self.food_amount_embed(hs[-1])  # [n_batch, n_query, 256] -> [n_batch, n_query, n_amount_class]
                out['amount_score'] = amount_score

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, list_transformers, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, use_progress=False, use_amount=False,
                 amount_type=0, n_amount_class=50,
                 list_index_amount_trfm=[], list_index_progress_trfm=[]):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.num_feature_levels = num_feature_levels
        self.list_index_amount_trfm = list_index_amount_trfm
        self.list_index_progress_trfm = list_index_progress_trfm

        self.list_deformDETRs = nn.ModuleList([])
        self.n_amount_class = n_amount_class

        for i_th, i_transformer in enumerate(list_transformers):
            if i_th in self.list_index_progress_trfm:
                do_evaluate_progress = True
            else:
                do_evaluate_progress = False

            if use_progress is False:
                do_evaluate_progress = False

            if i_th in self.list_index_amount_trfm:
                self.list_deformDETRs.append(DeformableDETRafterBackbone(i_transformer, num_classes, num_queries,
                                                              self.num_feature_levels,
                                                              aux_loss=aux_loss,
                                                              with_box_refine=with_box_refine,
                                                              two_stage=two_stage,
                                                              use_progress=do_evaluate_progress,
                                                              use_amount=use_amount,
                                                              amount_type=amount_type,
                                                              n_amount_class=n_amount_class)
                                         )
            else:
                self.list_deformDETRs.append(
                    DeformableDETRafterBackbone(i_transformer, num_classes, num_queries,
                                                self.num_feature_levels,
                                                aux_loss=aux_loss,
                                                with_box_refine=with_box_refine,
                                                two_stage=two_stage,
                                                use_progress=do_evaluate_progress,
                                                use_amount=False)
                    )

        self.num_deformDETRs = len(self.list_deformDETRs)

        if len(self.list_deformDETRs) > 1:
            assert self.list_deformDETRs[0].transformer.d_model == self.list_deformDETRs[1].transformer.d_model
        hidden_dim = self.list_deformDETRs[0].transformer.d_model

        if self.num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            # the last feature map is generated from the last backbone layers' output
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)  # list to moduleList
        else:
            # input_proj makes feature maps has same dims. ex: 1024, 2048 -> 512, 512
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone

        # initialization
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)


    def forward(self, samples: NestedTensor, samples_t: NestedTensor = None,
                targets_for_amount=None, ds_index=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # features, pos = self.backbone(samples)  # get three level feature maps as a list
        if samples_t is not None:
            if not isinstance(samples_t, NestedTensor):
                samples_t = nested_tensor_from_tensor_list(samples_t)
            features, pos = self.backbone(samples, samples_t)
        else:
            features, pos = self.backbone(samples)  # get three level feature maps as a list
        # samples: [num_batch, 3, H, W]
        # features: nestedTensor (list of lv tensor), [0] [2, 512, h0, w0], [1] [2, 1024, h1, w1], [2] [2, 2048, h2, w2]
        # pos: list of tensor, [0] [2, 256 (d_model), h0, w0]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))  # diff ch -> d_model(256) ch
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):  # generate more features from existing features
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)  # pos_encoding
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if ds_index is None:
            # aggregate results of all trfm
            assert targets_for_amount is None   # assert (True), go next line.

            # get the outputs from each deformDETR
            list_outputs = []
            for i_defDETR, defDETR in enumerate(self.list_deformDETRs):
                list_outputs.append(self.list_deformDETRs[i_defDETR](srcs, masks, pos,
                                                              targets_for_amount=None))

            # 'pred_logits': [2, 300, 5]
            # 'pred_boxes': [2, 300, 4]
            # 'amount_score': [2, 300, 50]
            # 'progress_score': [2, 3]

            # 'aux_outputs': list of dict of 'pred_logits', 'pred_boxes'
            # 'enc_outputs': dict of 'pred_logits', 'pred_boxes'

            # 'hs_last': hs[-1]: [n_batch, n_query(300), 256]
            # 'enc_memory':      [n_batch, 18669, d_model(256)]
            #         # 18669 = spatial_shapes's pixels = 84*167 + 42*84 + 21*42 + 11*21

            # merge into one in n_query axis => [n_batch, n_qeury x 2, n_class+1]
            out = {}
            for k in ['pred_logits', 'pred_boxes']:
                list_targets = [item[k] for item in list_outputs]
                out[k] = torch.cat(list_targets, dim=1)

            for k in ['hs_last', 'enc_memory']:
                list_targets = [item[k].unsqueeze(1) for item in list_outputs]
                out[k] = torch.cat(list_targets, dim=1)

            # [1, 300, 50]
            list_targets = [item['amount_score'] if ith_output in self.list_index_amount_trfm
                            else torch.zeros([item['pred_logits'].size(0), item['pred_logits'].size(1), self.n_amount_class]).cuda()
                            for ith_output, item in enumerate(list_outputs)
                            ]
            if len(self.list_index_amount_trfm) > 0:
                out['amount_score'] = torch.cat(list_targets, dim=1)

            # pdb.set_trace()
            # 'food' only has 'amount'?. No. All outputs from amount models has 'amount'
            # But 'amount' except for 'food'/'drink' is not accurate.

            # progress score from backbone, same result, but get avg
            # [1, 3]
            list_targets = [item['progress_score']
                            for ith_output, item in enumerate(list_outputs)
                            if ith_output in self.list_index_progress_trfm
                            ]
            if len(self.list_index_progress_trfm) > 0:
                out['progress_score'] = torch.mean(
                    torch.stack(list_targets), dim=0)

            # [enc_outputs][pred_logits] is a pred_logits of encoder (every pixels' estimation)
            # so, almost same values btw ds1 and ds2.
            out['enc_outputs'] = {}
            for k in ['pred_logits', 'pred_boxes']:
                list_targets = [item['enc_outputs'][k] for item in list_outputs]
                out['enc_outputs'][k] = torch.mean(torch.stack(list_targets), dim=0)


            out['aux_outputs'] = []
            for ao_l in range(len(list_outputs[0]['aux_outputs'])):
                out['aux_outputs'].append({
                    'pred_logits': torch.cat([item['aux_outputs'][ao_l]['pred_logits']
                                              for item in list_outputs],
                                             dim=1),
                    'pred_boxes': torch.cat([item['aux_outputs'][ao_l]['pred_boxes']
                                              for item in list_outputs],
                                            dim=1)
                })
        else:
            # evaluate only ds_index trfm
            out = self.list_deformDETRs[ds_index](srcs, masks, pos,
                                                  targets_for_amount=targets_for_amount)

        # feature map for active learning
        list_masked_srcs_gpooled = []
        for l in range(self.num_feature_levels):
            # area (mask == False) has real image value
            # ~mask means True having real image value
            # srcs[l]: [n_b, 256, h, w]
            # masks[l]: [n_b, h, w], True == masking, False == Image
            masked_srcs_l = masked_mean(srcs[l], torch.unsqueeze(~masks[l], dim=1), dim=(2, 3))

            list_masked_srcs_gpooled.append(masked_srcs_l)

        masked_srcs_gpooled = torch.cat(list_masked_srcs_gpooled, dim=1)    # 256 * 4 = 1024
        out['backbone_features'] = masked_srcs_gpooled
        out['backbone_src'] = srcs
        out['backbone_masks'] = masks

        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, use_progress=False,
                 use_amount=False, ds1_valid_classes=None, ds2_valid_classes=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.use_progress = use_progress
        self.use_amount = use_amount

        self.ds1_valid_classes = ds1_valid_classes
        self.ds2_valid_classes = ds2_valid_classes

    def loss_progress(self, outputs, targets, indices, num_boxes, ds_index=None):
        # outputs: dict, 'pred_logits', 'pred_boxes', 'aux_outputs', 'progress_pred', 'amount_pred'
        # targets: list,
        # indices: list, (indices of query, indices of target) matched by the last layer
        # assert 'progress_score' in outputs and 'progress_index' in targets
        if 'progress_index' not in targets[0]:      # targets is list
            losses = {'loss_progress': torch.tensor(0., device=outputs['pred_logits'].device)}
        else:
            if 'progress_score' in outputs:
                src_progress_logits = outputs['progress_score']
                target_progress = torch.tensor([t["progress_index"] for t in targets], dtype=torch.long,
                                               device=src_progress_logits.device)
                loss_progress = F.cross_entropy(src_progress_logits, target_progress)

                losses = {'loss_progress': loss_progress}
            else:
                losses = {'loss_progress': torch.tensor(0., device=outputs['pred_logits'].device)}

        return losses

    # def loss_amount_original(self, outputs, targets, indices, num_boxes, ds_index=None):
    #     # assert 'amount_pred' in outputs and 'amounts' in targets
    #     if 'amounts' not in targets[0]:        # targets is list
    #         losses = {'loss_amount': torch.tensor(0., device=outputs['pred_logits'].device)}
    #     else:
    #         if 'amount_pred' in outputs:
    #             src_amount = outputs['amount_pred']  # [num_batch, num_query, 1]
    #             idx = self._get_src_permutation_idx(indices)  # from [idx_query, idx_tgt], return [idx_batch, idx_query]
    #             target_amount = torch.cat([t["amounts"][J] for t, (_, J) in zip(targets, indices)])  # list to torch
    #             source_amount = src_amount[idx].squeeze()
    #
    #             nonzero_index = torch.nonzero(target_amount, as_tuple=True)
    #             nonzero_index = nonzero_index[0]
    #
    #             if len(nonzero_index) > 0:
    #                 # nonzero_index = [] makes loss_amount be nan
    #                 # loss_amount = F.mse_loss(source_amount[nonzero_index], target_amount[nonzero_index])
    #                 loss_amount = F.l1_loss(source_amount[nonzero_index], target_amount[nonzero_index])
    #
    #                 losses = {'loss_amount': loss_amount}
    #             else:
    #                 losses = {'loss_amount': torch.tensor(0., device=outputs['pred_logits'].device)}
    #         elif 'amount_score' in outputs:
    #             src_score = outputs['amount_score']  # [1/num_batch, 300/num_query, n_amount_class/50]
    #             idx = self._get_src_permutation_idx(indices)  # from [idx_query, idx_tgt], return [idx_batch, idx_query]
    #             target_amount = torch.cat([t["amounts"][J] for t, (_, J) in zip(targets, indices)])  # list to torch
    #
    #             # source_score = src_score[idx].squeeze() # idx: two dim (n_batch, n_query)
    #             source_score = src_score[idx]  # idx: two dim (n_batch, n_query)
    #             # [n_index, n_amount_class]
    #
    #             pdb.set_trace()
    #
    #             # food (1) only has amount
    #             # empty_container (5) also has amount as 0, but not used in amount estimation
    #
    #             nonzero_index = torch.nonzero(target_amount, as_tuple=True)
    #             nonzero_index = nonzero_index[0]        # 2-elem tuple -> 1-dim tensor
    #
    #             if len(nonzero_index) > 0:
    #                 source_score = source_score[nonzero_index]
    #                 target_amount = target_amount[nonzero_index]
    #
    #                 target_amount_class = (target_amount * 100)  # 0~1 -> 0~100
    #                 n_amount_classes = source_score.size(1)     # 1-dim data
    #
    #                 n_div = 100 // n_amount_classes
    #                 # 100 classes > 1 n_div
    #                 # 50 classes > 2 n_div
    #                 target_amount_class = target_amount_class.int() // n_div  # 50 CLASSES
    #                 target_amount_class = torch.clamp(target_amount_class, min=0, max=n_amount_classes - 1)
    #
    #                 # cross entropy loss
    #                 loss_amount = F.cross_entropy(source_score, target_amount_class.type(torch.long))
    #
    #                 losses = {'loss_amount': loss_amount}
    #             else:
    #                 losses = {'loss_amount': torch.tensor(0., device=outputs['pred_logits'].device)}
    #         else:
    #             losses = {'loss_amount': torch.tensor(0., device=outputs['pred_logits'].device)}
    #
    #     return losses

    def loss_amount(self, outputs, targets, indices, num_boxes, ds_index=None):
        # assert 'amount_pred' in outputs and 'amounts' in targets
        if 'amounts' not in targets[0]:        # targets is list
            losses = {'loss_amount': torch.tensor(0., device=outputs['pred_logits'].device)}
        else:
            if 'amount_pred' in outputs:
                src_amount = outputs['amount_pred']  # [num_batch, num_query, 1]
                idx = self._get_src_permutation_idx(indices)  # from [idx_query, idx_tgt], return [idx_batch, idx_query]
                target_amount = torch.cat([t["amounts"][J] for t, (_, J) in zip(targets, indices)])  # list to torch
                source_amount = src_amount[idx].squeeze()

                nonzero_index = torch.nonzero(target_amount, as_tuple=True)
                nonzero_index = nonzero_index[0]

                if len(nonzero_index) > 0:
                    # nonzero_index = [] makes loss_amount be nan
                    # loss_amount = F.mse_loss(source_amount[nonzero_index], target_amount[nonzero_index])
                    loss_amount = F.l1_loss(source_amount[nonzero_index], target_amount[nonzero_index])

                    losses = {'loss_amount': loss_amount}
                else:
                    losses = {'loss_amount': torch.tensor(0., device=outputs['pred_logits'].device)}
            elif 'amount_score' in outputs:
                src_score = outputs['amount_score']  # [1/num_batch, 300/num_query, n_amount_class/50]
                idx = self._get_src_permutation_idx(indices)  # from [idx_query, idx_tgt], return [idx_batch, idx_query]
                target_amount = torch.cat([t["amounts"][J] for t, (_, J) in zip(targets, indices)])  # list to torch

                # source_score = src_score[idx].squeeze() # idx: two dim (n_batch, n_query)
                source_score = src_score[idx]  # idx: two dim (n_batch, n_query)
                # [n_index, n_amount_class]

                # food (1) only has amount
                # empty_container (5) also has amount as 0, but not used in amount estimation

                nonzero_index = torch.nonzero(target_amount, as_tuple=True)
                nonzero_index = nonzero_index[0]        # 2-elem tuple -> 1-dim tensor

                weight_amounts = torch.zeros(target_amount.shape, device=target_amount.device)
                weight_amounts[nonzero_index] = 1.0

                # target_amount (0.40) -> target_amount_class (8)
                n_amount_classes = source_score.size(1)  # 1-dim data
                target_amount_class = (target_amount * 100)  # 0~1 -> 0~100
                n_div = 100 // n_amount_classes
                # 100 classes > 1 n_div
                # 50 classes > 2 n_div
                # target_amount_class = (target_amount_class.int()) // n_div  # 50 CLASSES
                # # if n_div == 2 (one bin cover 2 values), 0 <- 0~1, 1 <- 2~3, 2 <- 4~5, ... 49 <- 98, 49 <- 99, 50 <- 100
                target_amount_class = (target_amount_class.int() - 1) // n_div  # 50 CLASSES
                # if n_div == 2 (one bin cover 2 values), -1 <- 0, 0 <- 1~2, ... 48 <- 98, 49 <- 99~100
                # if n_div == 5 (one bin covers 5 values), -1 <- 0, 0 <- 1~5, ... 19 <- 96~100
                target_amount_class = torch.clamp(target_amount_class.type(torch.long),
                                                  min=-1, max=n_amount_classes - 1)

                # cross entropy loss
                # loss_amount_old = F.cross_entropy(source_score[nonzero_index], target_amount_class[nonzero_index])    # for verification
                loss_amount_alt = F.cross_entropy(source_score, target_amount_class, reduction='none')
                loss_amount = loss_amount_alt * weight_amounts

                denom = torch.sum(weight_amounts)
                if denom == 0:
                    denom = 1
                loss_amount = torch.sum(loss_amount) / denom

                # print(loss_amount_old)
                # print(loss_amount)
                # print('\n')

                losses = {'loss_amount': loss_amount}
            else:
                losses = {'loss_amount': torch.tensor(0., device=outputs['pred_logits'].device)}

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, ds_index=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_selective_labels(self, outputs, targets, indices, num_boxes, log=True, ds_index=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # print('in loss_selective_labels\n')

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])   # target label index, start from 1
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)                # list of labels filled with max index + 1
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]    # delete the last one (max index + 1)

        # src_logits_selected = src_logits
        # target_classes_onehot_selected = target_classes_onehot
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                     alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, ds_index=None):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, ds_index=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, ds_index=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }

        if self.ds1_valid_classes is None and self.ds2_valid_classes is None:
            loss_map['labels'] = self.loss_labels
        else:
            loss_map['labels'] = self.loss_selective_labels

        if self.use_amount:
            loss_map['amount'] = self.loss_amount
        if self.use_progress:
            loss_map['progress'] = self.loss_progress
        assert loss in loss_map, 'do you really want to compute [%s] loss?' % loss

        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, ds_index=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # print(outputs_without_aux["pred_logits"].shape)
        # print(outputs_without_aux["pred_boxes"].shape)
        #
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # print(tgt_ids.shape)
        # print(tgt_bbox.shape)
        # pdb.set_trace()

        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {'ds_index': ds_index}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                        kwargs.update({'ds_index': ds_index})
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                    kwargs.update({'ds_index': ds_index})
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

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
        topk_boxes = topk_indexes // out_logits.shape[2]        # index of rois
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


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


def build(args, num_classes, num_trfms):
    print('\n')
    print('Deformable DETR with multi-output and multi-dataset and multi-transformer!!!!!')
    print('\n')

    print('model has %d classes for classification' % num_classes)

    device = torch.device(args.device)

    if not hasattr(args, 'TrainTargetSetFilename'):
        backbone = build_backbone(args)
    else:
        if args.TrainTargetSetFilename == '' or args.TrainTargetSetFilename is None:
            backbone = build_backbone(args)
        elif args.TrainTargetSetFilename[0] == '':
            backbone = build_backbone(args)
        elif args.TrainTargetSetFilename is not '':
            backbone = build_backbone_uda(args)

    list_transformers = []
    for _ in range(num_trfms):
        list_transformers.append(build_deforamble_transformer(args))

    if args.use_amount:
        amount_type = args.amount_type
    else:
        amount_type = -1

    model = DeformableDETR(
        backbone,
        list_transformers,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        use_amount=args.use_amount,
        use_progress=args.use_progress,
        amount_type=amount_type,
        n_amount_class=args.n_amount_class,
        list_index_amount_trfm=args.list_index_amount_trfm,
        list_index_progress_trfm=args.list_index_progress_trfm
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    if args.use_amount:
        weight_dict['loss_amount'] = args.amount_loss_coef
    if args.use_progress:
        weight_dict['loss_progress'] = args.progress_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    if args.use_amount:
        losses += ['amount']
    if args.use_progress:
        losses += ['progress']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses,
                             focal_alpha=args.focal_alpha,
                             use_amount=args.use_amount,
                             use_progress=args.use_progress
                             )
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    if args.use_amount:
        postprocessors['amount'] = True

    if args.use_progress:
        postprocessors['progress'] = True

    return model, criterion, postprocessors
