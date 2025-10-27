# coding: utf-8

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .nms.py_cpu_nms import py_cpu_nms as cpu_nms
def cpu_soft_nms(*args, **kwargs):
    return py_cpu_nms(*args, **kwargs)


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return cpu_nms(dets, thresh)
    # return gpu_nms(dets, thresh)
