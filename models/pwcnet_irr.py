
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 20:22:09 2020

@author: ali
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .pwc_modules import conv, rescale_flow, upsample2d_as, initialize_msra, compute_cost_volume
from .pwc_modules import WarpingLayer, FeatureExtractor, ContextNetwork, FlowEstimatorDense, FeatureExtractor2

class PWCNet(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(PWCNet, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}

        initialize_msra(self.modules())

    def forward(self, input_dict):

        x1_raw = input_dict[0]
        x2_raw = input_dict[1]
        _, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = upsample2d_as(flow, x1, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)

            # correlation
            out_corr = compute_cost_volume(x1, x2_warp, self.corr_params)
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=True)

            x1_1by1 = self.conv_1x1[l](x1)
            x_intm, flow_res = self.flow_estimators(torch.cat([out_corr_relu, x1_1by1, flow], dim=1))
            flow = flow + flow_res

            flow_fine = self.context_networks(torch.cat([x_intm, flow], dim=1))
            flow = flow + flow_fine

            flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=False)
            # if (l==4):
            #     x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)
            #     my_refine = self.refine(flow,x1-x2_warp,x1)
            #     flows.append(my_refine)
            # else:
            flows.append(flow)

            # upsampling or post-processing
            if l == self.output_level:
                break


        output = flows

        if self.training:
            return output
        else:
            output_eval = upsample2d_as(flow, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
            return output_eval


class MPWCNet(nn.Module):
    def __init__(self, args, div_flow=0.1):
        super(MPWCNet, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 5
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor2(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}

        initialize_msra(self.modules())

    def forward(self, input_dict):

        x1_raw = input_dict[0]
        x2_raw = input_dict[1]
        _, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = upsample2d_as(flow, x1, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)

            # correlation
            out_corr = compute_cost_volume(x1, x2_warp, self.corr_params)
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=True)

            x1_1by1 = self.conv_1x1[l](x1)
            x_intm, flow_res = self.flow_estimators(torch.cat([out_corr_relu, x1_1by1, flow], dim=1))
            flow = flow + flow_res

            flow_fine = self.context_networks(torch.cat([x_intm, flow], dim=1))
            flow = flow + flow_fine

            flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=False)
            # if (l==4):
            #     x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)
            #     my_refine = self.refine(flow,x1-x2_warp,x1)
            #     flows.append(my_refine)
            # else:
            flows.append(flow)

            # upsampling or post-processing
            if l == self.output_level:
                break


        output = flows

        if self.training:
            return output
        else:
            output_eval = upsample2d_as(flow, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
            return output_eval

