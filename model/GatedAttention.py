import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import split_feature, merge_splits, generate_shift_window_attn_mask
import time
from thop import profile


# GAU
class GatedAttentionUnit(nn.Module):
    def __init__(self, dim=128, nHead=1):
        super(GatedAttentionUnit, self).__init__()

        self.dim = dim
        self.query_key_dim = dim if nHead == 1 else dim * nHead
        self.nHead = nHead
        self.norm = nn.LayerNorm(dim)
        self.to_gq = nn.Sequential(nn.Linear(dim, self.query_key_dim * 2), nn.SiLU())
        self.to_kv = nn.Sequential(nn.Linear(dim, self.query_key_dim * 2), nn.SiLU())
        self.to_out = nn.Sequential(nn.Linear(self.query_key_dim, dim),
                                    nn.SiLU(),
                                    nn.Linear(dim, dim, bias=False)
                                    )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source, target=None, h=None, w=None,
                mask=None, attn_type="full", num_splits=2, with_shift=True):
        # source, target:[B, L, C]
        global shift_size_h, shift_size_w, b_new, window_size_h, window_size_w
        assert self.nHead > 0
        assert h is not None and w is not None
        assert source.size(1) == h * w
        b, _, c = source.size()

        is_self_attn = (source - target).abs().max() < 1e-6

        res = source
        seq_len = source.shape[1]
        source = self.norm(source)
        target = self.norm(target)

        gate, Q = self.to_gq(source).chunk(2, dim=-1)
        K, V = self.to_kv(target).chunk(2, dim=-1)

        if not is_self_attn:
            Q = Q.view(b, h, w, c * self.nHead)
            K = K.view(b, h, w, c * self.nHead)
            V = V.view(b, h, w, c * self.nHead)

        if "swin" in attn_type:
            # assert self.nHead <= 1
            b_new = b * num_splits * num_splits

            window_size_h = h // num_splits
            window_size_w = w // num_splits

            Q = Q.view(b, h, w, c * self.nHead)
            K = K.view(b, h, w, c * self.nHead)
            V = V.view(b, h, w, c * self.nHead)

            if with_shift:
                assert mask is not None  # compute once
                shift_size_h = window_size_h // 2
                shift_size_w = window_size_w // 2

                Q = torch.roll(Q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
                K = torch.roll(K, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
                V = torch.roll(V, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

            Q = split_feature(Q, num_splits=num_splits, channel_last=True)  # [B*K*K, H/K, W/K, C]
            K = split_feature(K, num_splits=num_splits, channel_last=True)
            V = split_feature(V, num_splits=num_splits, channel_last=True)

        if self.nHead == 1:
            if is_self_attn:
                if "swin" not in attn_type:
                    attention_scores = torch.matmul(Q, K.permute(0, 2, 1)) / (self.dim * seq_len)
                    attention_scores = self.softmax(attention_scores)  # [B, L, L]
                    gated_attention_output = torch.matmul(attention_scores, V)  # [B, L, C]
                else:
                    attention_scores = torch.matmul(Q.view(b_new, -1, c), K.view(b_new, -1, c).permute(0, 2, 1)
                                                    ) / (self.dim * seq_len)  # [B*K*K, H/K*W/K, H/K*W/K]

                    if with_shift:
                        attention_scores += mask.repeat(b, 1, 1)

                    attn = torch.softmax(attention_scores, dim=-1)

                    gated_attention_output = torch.matmul(attn, V.view(b_new, -1, c))  # [B*K*K, H/K*W/K, C]

                    gated_attention_output = merge_splits(
                        gated_attention_output.view(b_new, h // num_splits, w // num_splits, c),
                        num_splits=num_splits, channel_last=True)  # [B, H, W, C]

                    # shift back
                    if with_shift:
                        gated_attention_output = torch.roll(gated_attention_output, shifts=(shift_size_h, shift_size_w),
                                                            dims=(1, 2))

                    gated_attention_output = gated_attention_output.view(b, -1, c)
            else:
                attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (
                        self.dim * seq_len)  # [B, H, W, W]
                if mask is not None:
                    attention_scores[~mask] = -1e9
                attention_scores = self.softmax(attention_scores)
                gated_attention_output = torch.matmul(attention_scores, V).view(b, -1, c)  # [B, H*W, C]

            gated_attention_output = self.to_out(gated_attention_output * gate)  # hadamard product

        else:
            split_size = self.query_key_dim // self.nHead
            # process_dim = -1
            Q = torch.stack(torch.split(Q, split_size, dim=-1), dim=0)  # [h, N, T_q, num_units/h]
            K = torch.stack(torch.split(K, split_size, dim=-1), dim=0)  # [h, N, T_k, num_units/h]
            V = torch.stack(torch.split(V, split_size, dim=-1), dim=0)  # [h, N, T_k, num_units/h]
            if is_self_attn:
                if "swin" not in attn_type:
                    attention_scores = torch.matmul(Q, K.transpose(2, 3)) / (split_size * seq_len)
                    attention_scores = self.softmax(attention_scores)  # [B, L, L]
                    gated_attention_output = torch.matmul(attention_scores, V)  #
                    gated_attention_output = torch.cat(torch.split(gated_attention_output, 1, dim=0), dim=3).squeeze(0)
                else:
                    attention_scores = torch.matmul(Q.view(self.nHead, b_new, -1, c), K.view(self.nHead, b_new, -1, c).permute(0, 1, 3, 2)
                                                    ) / (split_size * seq_len)  # [H, B*K*K, H/K*W/K, H/K*W/K]

                    if with_shift:
                        attention_scores += mask.repeat(self.nHead, b, 1, 1)

                    attn = torch.softmax(attention_scores, dim=-1)

                    gated_attention_output = torch.matmul(attn, V.view(self.nHead, b_new, -1, c))  # [H, B*K*K, H/K*W/K, C]
                    gated_attention_output = torch.cat(torch.split(gated_attention_output, 1, dim=0), dim=3).squeeze(0)
                    gated_attention_output = merge_splits(
                        gated_attention_output.view(b_new, h // num_splits, w // num_splits, c * self.nHead),
                        num_splits=num_splits, channel_last=True)  # [B, H, W, C]

                    # shift back
                    if with_shift:
                        gated_attention_output = torch.roll(gated_attention_output, shifts=(shift_size_h, shift_size_w),
                                                            dims=(1, 2))

                    gated_attention_output = gated_attention_output.view(b, -1, c * self.nHead)
            else:
                attention_scores = torch.matmul(Q, K.transpose(3, 4)) / (
                        split_size * seq_len)  # [B, H, W, W]
                attention_scores = self.softmax(attention_scores)
                gated_attention_output = torch.matmul(attention_scores, V).view(self.nHead, b, -1, c)  # [B, H*W, C]
                gated_attention_output = torch.cat(torch.split(gated_attention_output, 1, dim=0), dim=3).squeeze(0)

            gated_attention_output = self.to_out(gated_attention_output * gate)  # hadamard product

        attention_out = gated_attention_output
        gated_attention_output += res

        return gated_attention_output, attention_out


class Conv2FormerAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.norm = LayerNorm(dim, eps=1e-6, data_format='channel_first')
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        x = a * v
        x = self.proj(x)
        return x


# GFAU
class GatedFreeAttentionUnit(nn.Module):
    def __init__(self, dim=128, hidden_dim=64):
        super(GatedFreeAttentionUnit, self).__init__()

        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.hidden_dim = hidden_dim
        self.to_q = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
        self.to_kv = nn.Sequential(nn.Linear(dim, hidden_dim * 2))
        # self.to_kv = nn.Linear(dim, hidden_dim*2)
        self.softmax = nn.Softmax(dim=1)
        # self.project = nn.Linear(hidden_dim, dim)
        self.project = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(hidden_dim, dim, bias=False)
                                     )

    def forward(self, FeaL, FeaR, h=None, w=None):
        # source, target:[B, L, C]

        source = self.norm(FeaL)
        target = self.norm(FeaR)
        # source, target = FeaL, FeaR

        K, V = self.to_kv(source).chunk(2, dim=-1)
        Q = self.to_q(target)
        K = self.softmax(K)
        weights = torch.mul(K, V).sum(dim=1, keepdim=True)
        weights = torch.mul(Q, weights)  # [B, L, C]
        # weights = self.project(gate * weights)

        weights = self.project(weights)
        weights += FeaL

        return weights


# AttnBlock
class AttnBlock(nn.Module):
    def __init__(self, dim=128, nHead=1):
        super(AttnBlock, self).__init__()
        self.GAU = GatedAttentionUnit(dim, nHead)
        # self.GAU = GatedFreeAttentionUnit(query_key_dim)

    def forward(self, source, target, h=None, w=None, mask=None, swin_mask=None,
                attn_type="full", num_splits=2, with_shift=True):
        # self-attn
        source, _ = self.GAU(source, source, h, w, swin_mask, attn_type, num_splits, with_shift)
        # source = self.GAU(source, source)

        # cross-attn
        source, attn_out = self.GAU(source, target, h, w, mask, "full")

        # source = self.GAU(source, target, h, w)

        return source, attn_out


# GatedAttention
class GatedAttn(nn.Module):
    def __init__(self, dim=128, nHead=1, num_layers=6):
        super(GatedAttn, self).__init__()
        self.layers = nn.ModuleList([
            AttnBlock(dim, nHead)
            for i in range(num_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # self.deconv = nn.Sequential(nn.ConvTranspose2d(query_key_dim, query_key_dim, 4, 2, 1),
        #                             nn.BatchNorm2d(query_key_dim),
        #                             nn.LeakyReLU(inplace=True))

    def forward(self, source, target, if_mask=True, need_attnmap=False, attn_type="full",
                attn_num_splits=2, with_shift=True):

        global valid_mask, attn_out
        b, c, h, w = source.shape
        source = source.flatten(-2).permute(0, 2, 1)
        target = target.flatten(-2).permute(0, 2, 1)

        # concat feature0 and feature1 in batch dimension to compute in parallel
        concat0 = torch.cat((source, target), dim=0)  # [2B, H*W, C]
        concat1 = torch.cat((target, source), dim=0)  # [2B, H*W, C]

        if if_mask:
            mask = torch.triu(torch.ones((w, w)), diagonal=1).type_as(source)  # [W, W]
            valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(b * 2, h, 1, 1)  # [B, H, W, W]
        else:
            valid_mask = None

        # 2d attention
        if 'swin' in attn_type and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=source.device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        for i, layer in enumerate(self.layers):
            # source = layer(source, target, h, w)
            concat0, attn_out = layer(concat0, concat1, h, w, valid_mask, shifted_window_attn_mask,
                                      attn_type, attn_num_splits, with_shift)
            # update feature1
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        source, target = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]
        source = source.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        target = target.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        if need_attnmap:
            left_attn, right_attn = attn_out.chunk(chunks=2, dim=0)
            left_attn = left_attn.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            right_attn = right_attn.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return left_attn, right_attn
        else:
            return source, target


class SelfAttnPropagation(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(SelfAttnPropagation, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                local_window_attn=False,
                local_window_radius=1,
                **kwargs,
                ):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow,
                                                  local_window_radius=local_window_radius)

        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # a note: the ``correct'' implementation should be:
        # ``query = self.q_proj(query), key = self.k_proj(query)''
        # this problem is observed while cleaning up the code
        # however, this doesn't affect the performance since the projection is a linear operation,
        # thus the two projection matrices for key can be merged
        # so I just leave it as is in order to not re-train all models :)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out

    def forward_local_window_attn(self, feature0, flow,
                                  local_window_radius=1,
                                  ):
        assert flow.size(1) == 2 or flow.size(1) == 1  # flow or disparity or depth
        assert local_window_radius > 0

        b, c, h, w = feature0.size()

        value_channel = flow.size(1)

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b * h * w, 1, c)  # [B*H*W, 1, C]

        kernel_size = 2 * local_window_radius + 1

        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)

        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  # [B, C*(2R+1)^2), H*W]

        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)  # [B*H*W, C, (2R+1)^2]

        flow_window = F.unfold(flow, kernel_size=kernel_size,
                               padding=local_window_radius)  # [B, 2*(2R+1)^2), H*W]

        flow_window = flow_window.view(b, value_channel, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, value_channel)  # [B*H*W, (2R+1)^2, 2]

        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B*H*W, 1, (2R+1)^2]

        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, flow_window).view(b, h, w, value_channel
                                                   ).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        return out


def testModel():
    c = 128
    model = GatedAttn(c, 1, 1)
    x = torch.Tensor(torch.randn(1, c, 36, 64))
    y = torch.Tensor(torch.randn(1, c, 36, 64))
    start_time = time.time()
    out = model(x, y, False, False, "swin", 2, True)
    end_time = time.time()
    print(model)
    # GatedAttn(
    #   (layers): ModuleList(
    #     (0): AttnBlock(
    #       (GAU): GatedAttentionUnit(
    #         (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    #         (to_gq): Sequential(
    #           (0): Linear(in_features=128, out_features=256, bias=True)
    #           (1): SiLU()
    #         )
    #         (to_kv): Sequential(
    #           (0): Linear(in_features=128, out_features=256, bias=True)
    #           (1): SiLU()
    #         )
    #         (to_out): Sequential(
    #           (0): Linear(in_features=128, out_features=128, bias=True)
    #           (1): SiLU()
    #           (2): Linear(in_features=128, out_features=128, bias=False)
    #         )
    #         (softmax): Softmax(dim=-1)
    #       )
    #     )
    #   )
    # )
    #
    print('Time taken = {} sec'.format(end_time - start_time))
    for i in range(len(out)):
        print(out[i].shape)
    flops, params = profile(model, inputs=(x, y))

    print(flops / 1e9)
    print(params / 1e6)


if __name__ == "__main__":
    testModel()
