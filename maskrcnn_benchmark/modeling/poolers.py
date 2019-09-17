# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List

import torch
import torch.jit
from torch import nn

from maskrcnn_benchmark.layers import ROIAlign

from .utils import cat


# when JIT supports indexing of a module list in script_methods,
# this could be merged with the for loop at the end of Pooler.forward
# into a single script_method
@torch.jit.script
def merge_levels(levels, unmerged_results):
    # type: (Tensor, List[Tensor]) -> Tensor
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros((levels.size(0), first_result.size(1),
                       first_result.size(2), first_result.size(3)),
                      dtype=dtype, device=device)
    for l in range(len(unmerged_results)):
        res.masked_scatter_((levels == l).view(-1, 1, 1, 1), unmerged_results[l])
    return res

def merge_levels_onnx(levels, unmerged_results):
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros((levels.size(0), first_result.size(1),
                       first_result.size(2), first_result.size(3)),
                      dtype=dtype, device=device)
    for l in range(len(unmerged_results)):
        index = (levels == l).nonzero().view(-1, 1, 1, 1)
        # WORK AROUND: masked_scatter_ not in ONNX
        index = index.expand(index.size(0),
                        unmerged_results[l].size(1),
                        unmerged_results[l].size(2),
                        unmerged_results[l].size(3)).to(torch.long)
        res.scatter_(0, index, unmerged_results[l])
    return res

class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(torch.tensor(self.eps, dtype=torch.float32) + s / self.s0))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

        self.onnx_export = False

    def prepare_onnx_export(self):
        self.onnx_export = True

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                # we use full_like to allow tracing with flexible shape
                torch.full_like(b.bbox[:, :1], i)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        # num_rois = len(rois)
        # num_channels = x[0].shape[1]
        # output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        unmerged_results = []
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            unmerged_results.append(pooler(per_level_feature, rois_per_level).to(dtype))

        if self.onnx_export:
            result = merge_levels_onnx(levels, unmerged_results)
        else:
            result = merge_levels(levels, unmerged_results)
        return result
