# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

# from .deformable_detr import build as build_singleoutput
# def build_singleoutput_model(args, dataset=None):
#     return build_singleoutput(args, dataset=dataset)
#
# from .deformable_detr_multioutput import build as build_multioutput
# def build_multioutput_model(args, dataset=None):
#     return build_multioutput(args, dataset=dataset)
#
# from .deformable_detr_multioutput_multidataset import build as build_multioutput_multidataset
# def build_multioutput_multidataset_model(args, dataset1=None, dataset2=None,
#                                          ds1_valid_classes=None, ds2_valid_classes=None):
#     return build_multioutput_multidataset(args, dataset1=dataset1, dataset2=dataset2,
#                                           ds1_valid_classes=ds1_valid_classes,
#                                           ds2_valid_classes=ds2_valid_classes)
#
# from .deformable_detr_multioutput_multidataset_twohead import build as build_multioutput_multidataset_twohead
# def build_multioutput_multidataset_model_twohead(args, dataset1=None, dataset2=None,
#                                                  ds1_valid_classes=None, ds2_valid_classes=None):
#     return build_multioutput_multidataset_twohead(args, dataset1=dataset1, dataset2=dataset2,
#                                           ds1_valid_classes=ds1_valid_classes,
#                                           ds2_valid_classes=ds2_valid_classes)
#
# from .deformable_detr_multioutput_multidataset_twohead_pgt import build as build_multiout_multidataset_twohead_pgt
# def build_multioutput_multidataset_model_twohead_pgt(args, dataset1=None, dataset2=None):
#     return build_multiout_multidataset_twohead_pgt(args, dataset1=dataset1, dataset2=dataset2)
#
# from .deformable_detr_multioutput_multidataset_twotrfm import build as build_multioutput_multidataset_twotrfm
# def build_multioutput_multidataset_model_twotrfm(args, dataset1=None, dataset2=None):
#     return build_multioutput_multidataset_twotrfm(args, dataset1=dataset1, dataset2=dataset2)
#
# from .deformable_detr_multioutput_multidataset_twotrfm_rot import build as build_multioutput_multidataset_twotrfm_rot
# def build_multioutput_multidataset_model_twotrfm_rot(args, dataset1=None, dataset2=None):
#     return build_multioutput_multidataset_twotrfm_rot(args, dataset1=dataset1, dataset2=dataset2)
#
# from .deformable_detr_multioutput_multidataset_twotrfm_reweight import build as build_multioutput_multidataset_twotrfm_reweight
# def build_multioutput_multidataset_model_twotrfm_reweight(args, dataset1=None, dataset2=None, weight_focus_fg_bg=0):
#     return build_multioutput_multidataset_twotrfm_reweight(args, dataset1=dataset1, dataset2=dataset2,
#                                                            weight_focus_fg_bg=weight_focus_fg_bg)
#
# from .deformable_detr_multioutput_multidataset_twotrfm_grStop import build as build_multioutput_multidataset_twotrfm_grStop
# def build_multioutput_multidataset_model_twotrfm_grStop(args, dataset1=None, dataset2=None,
#                                                  ds1_valid_classes=None, ds2_valid_classes=None):
#     return build_multioutput_multidataset_twotrfm_grStop(args, dataset1=dataset1, dataset2=dataset2,
#                                           ds1_valid_classes=ds1_valid_classes,
#                                           ds2_valid_classes=ds2_valid_classes)
#
# from .deformable_detr_multioutput_multidataset_twotrfm_reweight_TS import build as build_multioutput_multidataset_twotrfm_reweight_TS
# def build_multioutput_multidataset_model_twotrfm_reweight_TS(args, dataset1=None, dataset2=None, weight_focus_fg_bg=0):
#     return build_multioutput_multidataset_twotrfm_reweight_TS(args, dataset1=dataset1, dataset2=dataset2,
#                                                            weight_focus_fg_bg=weight_focus_fg_bg)
#
#
# from .deformable_detr_multioutput_multidataset_twotrfm_reweight_TS_pretTwo import build as build_multioutput_multidataset_twotrfm_reweight_TS_pretTwo
# def build_multioutput_multidataset_model_twotrfm_reweight_TS_pretTwo(args, dataset1=None, dataset2=None, weight_focus_fg_bg=0):
#     return build_multioutput_multidataset_twotrfm_reweight_TS_pretTwo(args, dataset1=dataset1, dataset2=dataset2,
#                                                            weight_focus_fg_bg=weight_focus_fg_bg)
#
# from .deformable_detr_multioutput_multidataset_twotrfm_reweight_TS_pretTwo_SADA import build as build_multioutput_multidataset_twotrfm_reweight_TS_pretTwo_SADA
# def build_multioutput_multidataset_model_twotrfm_reweight_TS_pretTwo_SADA(args, dataset1=None, dataset2=None, weight_focus_fg_bg=0):
#     return build_multioutput_multidataset_twotrfm_reweight_TS_pretTwo_SADA(args, dataset1=dataset1, dataset2=dataset2,
#                                                            weight_focus_fg_bg=weight_focus_fg_bg)
#
# from .deformable_detr_multioutput_multidataset_multitrfm import build as build_multioutput_multidataset_multitrfm
# def build_multioutput_multidataset_model_multitrfm(args, num_classes, num_trfms):
#     return build_multioutput_multidataset_multitrfm(args, num_classes=num_classes,
#                                                     num_trfms=num_trfms)

from .deformable_detr_multioutput_multidataset_multitrfmModule import build as build_multioutput_multidataset_multitrfmModule
def build_multioutput_multidataset_model_multitrfmModule(args, num_classes, num_trfms):
    return build_multioutput_multidataset_multitrfmModule(args, num_classes=num_classes,
                                                    num_trfms=num_trfms)