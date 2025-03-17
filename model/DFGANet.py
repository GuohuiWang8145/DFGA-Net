# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .backbone import CNNEncoder
from .matching import (global_correlation_softmax_stereo, local_correlation_with_disp)
from .GatedAttention import GatedAttn, SelfAttnPropagation
# from .AFT import AFTModel, SelfAttnPropagation
from .utils import feature_add_position, upsample_disp_with_mask
from .reg_refine import BasicUpdateBlock


class DFGANet(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 ):
        super(DFGANet, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # CNN  0.1775 s || 0.0438 s [16, 32, 64] || 0.0229 s [8, 16, 32]
        self.backbone = CNNEncoder(in_dim=3, output_dim=feature_channels, norm_layer=nn.BatchNorm2d)
        self.local = CNNEncoder(in_dim=3, output_dim=feature_channels, norm_layer=nn.InstanceNorm2d)

        self.GatedAttn = GatedAttn(feature_channels, num_head, num_layers=num_transformer_layers)

        self.feature_disp_attn = SelfAttnPropagation(in_channels=feature_channels)  # 0.0259 s

        # concat feature0 and low res disp as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
        # thus far, all the learnable parameters are task-agnostic

        # optional task-specific local regression refinement
        self.refine_proj = nn.Conv2d(128, 256, 1)
        self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                       downsample_factor=upsample_factor,
                                       flow_dim=1,
                                       bilinear_up=False,
                                       )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]

        features = self.backbone(concat)  # downsample | list of [2B, C, H, W], resolution from high to low
        feature0, feature1 = torch.chunk(features, 2, 0)

        return feature0, feature1

    def extract_geometry(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]

        features = self.local(concat)  # downsample | list of [2B, C, H, W], resolution from high to low
        feature0, feature1 = torch.chunk(features, 2, 0)

        return feature0, feature1

    def upsample_disp(self, disp, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_disp = F.interpolate(disp, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((disp, feature), dim=1)
            mask = self.upsampler(concat)
            up_disp = upsample_disp_with_mask(disp, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_disp

    def forward(self, img0, img1,
                attn_type='Full',
                attn_splits_list=None,
                num_reg_refine=1,
                **kwargs,
                ):

        upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1))
        results_dict = {}
        disp_preds = []

        # context features
        feature0, feature1 = self.extract_feature(img0, img1)

        # geometry information
        geometry0, geometry1 = self.extract_geometry(img0, img1)
        # geometry0, geometry1 = feature0, feature1
        attn_splits = attn_splits_list[0]
        disp = None

        feature0, feature1 = feature_add_position(feature0, feature1, 1, self.feature_channels)
        feature0, feature1 = self.GatedAttn(feature0, feature1, False, False, attn_type, attn_splits, False)

        # correlation and softmax
        disp_pred = global_correlation_softmax_stereo(geometry0, geometry1)[0]

        # disp or residual disp
        disp = disp + disp_pred if disp is not None else disp_pred

        disp = disp.clamp(min=0)  # positive disparity

        # upsample to the original resolution for supervison at training time only
        if self.training:
            disp_bilinear = self.upsample_disp(disp, None, bilinear=True, upsample_factor=upsample_factor,
                                               is_depth=False)
            disp_preds.append(disp_bilinear)

        # disp propagation with self-attn
        disp = self.feature_disp_attn(feature0, disp.detach(),
                                      local_window_attn=False,
                                      local_window_radius=-1,
                                      )

        if not self.reg_refine:
            disp_pad = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
            disp_up_pad = self.upsample_disp(disp_pad, feature0)
            disp_up = -disp_up_pad[:, :1]  # [B, 1, H, W]
            disp_preds.append(disp_up)
        else:

            # task-specific local regression refinement
            # supervise current flow
            if self.training:
                disp_up = self.upsample_disp(disp, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=False)
                disp_preds.append(disp_up)

            assert num_reg_refine > 0
            for refine_iter_idx in range(num_reg_refine):
                disp = disp.detach()

                zeros = torch.zeros_like(disp)  # [B, 1, H, W]
                # NOTE: reverse disp, disparity is positive
                displace = torch.cat((-disp, zeros), dim=1)  # [B, 2, H, W]
                correlation = local_correlation_with_disp(
                    feature0, feature1, 
                    disp=displace,
                    local_radius=4,
                )  # [B, (2R+1)^2, H, W]

                proj = self.refine_proj(feature0)

                net, inp = torch.chunk(proj, chunks=2, dim=1)

                net = torch.tanh(net)
                inp = torch.relu(inp)

                net, up_mask, residual_disp = self.refine(net, inp, correlation, disp.clone(),
                                                          )

                disp = disp + residual_disp
                disp = disp.clamp(min=0)  # positive

                if self.training or refine_iter_idx == num_reg_refine - 1:
                    disp_up = upsample_disp_with_mask(disp, up_mask, upsample_factor=self.upsample_factor,
                                                      is_depth=False)

                    disp_preds.append(disp_up)

        for i in range(len(disp_preds)):
            disp_preds[i] = disp_preds[i].squeeze(1)  # [B, H, W]

        results_dict.update({'disp_preds': disp_preds})

        return results_dict


if __name__ == "__main__":
    x = torch.Tensor(torch.randn(1, 3, 36, 64))
    y = torch.Tensor(torch.randn(1, 3, 36, 64))
    # x = x.flatten(-2).permute(0, 2, 1)
    # x = x.permute(0, 2, 3, 1)
    # z = rope(x, dim=1)
    model = DFGANet()
    # outputs = model(x, y)
    flops, params = profile(model, inputs=(x, y))
    print(flops / 1e9)
    print(params)
