"""
FluxCodec LatentCodec - 基于 DiT-IC 设计，适配 FLUX AE (16x downsample)。

关键改动（相对 DiT-IC）：
- FLUX AE 16x = ELIC 16x，所以 concat 后只需一个 Conv 融合
- g_a 多下采样 2x（总 4x），使得像素域总下采样 = 16x * 4x = 64x
- g_s 多上采样 2x（总 4x）
- AuxDecoder 多上采样 2x（总 4x）
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from .modules import DepthConvBlock, ResidualBlockUpsample2, ResidualBlockWithStride2

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def ste_round(x):
    return (torch.round(x) - x).detach() + x


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def find_named_buffer(module, query):
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(module, buffer_name, state_dict_key, state_dict,
                               policy="resize_if_empty", dtype=torch.int):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)
    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')
        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)
    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')
        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))
    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(module, module_name, buffer_names, state_dict,
                               policy="resize_if_empty", dtype=torch.int):
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')
    for buffer_name in buffer_names:
        _update_registered_buffer(module, buffer_name,
                                   f"{module_name}.{buffer_name}", state_dict, policy, dtype)


# ======================== Building Blocks ========================

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.up(x)


class AnalysisTransform(nn.Module):
    """
    g_a: 4x 下采样。
    FLUX AE (16ch, H/16) 和 ELIC (320ch, H/16) 空间尺寸相同，
    直接 concat 后用一个 Conv 融合通道，然后做 4x 下采样到 H/64。
    """
    def __init__(self, ch_emd=128, channel=320):
        super().__init__()
        # FLUX AE 16x 和 ELIC 16x 空间尺寸一致，直接 concat + conv 融合
        self.fuse = nn.Conv2d(ch_emd + 320, 192, kernel_size=3, padding=1)  # 128+320=448 → 192

        self.analysis_transform = nn.Sequential(
            DepthConvBlock(192, 192),
            DepthConvBlock(192, 192),
            Downsample(192, 320),          # 2x down
            DepthConvBlock(320, 320),
            Downsample(320, channel),      # 2x down → 总 4x
            DepthConvBlock(channel, channel),
        )

    def forward(self, latent, latent2):
        """
        latent:  FLUX AE 输出 (B, 16, H/16, W/16)
        latent2: ELIC 辅助编码器输出 (B, 320, H/16, W/16)
        """
        x = torch.cat((latent, latent2), dim=1)
        x = self.fuse(x)
        x = self.analysis_transform(x)
        return x


class SynthesisTransform(nn.Module):
    """g_s: 4x 上采样（比 DiT-IC 多一层 Upsample）。"""
    def __init__(self, channel=320, channel_out=128):
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            DepthConvBlock(channel, 320),
            DepthConvBlock(320, 320),
            Upsample(320, 320),            # 2x up
            DepthConvBlock(320, 320),
            Upsample(320, 320),            # 2x up → 总 4x
            nn.Conv2d(320, channel_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.synthesis_transform(x)


class AuxDecoder(nn.Module):
    """辅助解码器: 4x 上采样（比 DiT-IC 多一层 Upsample）。"""
    def __init__(self, ch_emd=128, channel=320):
        super().__init__()
        self.block = nn.Sequential(
            DepthConvBlock(channel, 320),
            DepthConvBlock(320, 320),
            Upsample(320, 320),            # 2x up
            DepthConvBlock(320, 320),
            Upsample(320, 320),            # 2x up → 总 4x
            nn.Conv2d(320, ch_emd, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class HyperAnalysis(nn.Module):
    def __init__(self, channel=320):
        super().__init__()
        self.reduction = nn.Sequential(
            DepthConvBlock(channel, channel),
            DepthConvBlock(channel, channel // 2),
            ResidualBlockWithStride2(channel // 2, channel // 2),
            ResidualBlockWithStride2(channel // 2, channel // 2),
        )

    def forward(self, x):
        return self.reduction(x)


class HyperSynthesis(nn.Module):
    def __init__(self, channel=320):
        super().__init__()
        self.increase = nn.Sequential(
            ResidualBlockUpsample2(channel // 2, channel // 2),
            ResidualBlockUpsample2(channel // 2, channel // 2),
            DepthConvBlock(channel // 2, channel),
            DepthConvBlock(channel, channel),
        )

    def forward(self, x):
        return self.increase(x)


class Adapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, (in_ch + out_ch) // 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, (in_ch + out_ch) // 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.branch1(x)


class SpatialContext(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(
            DepthConvBlock(in_ch, in_ch),
            DepthConvBlock(in_ch, in_ch),
            DepthConvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, in_ch, 1),
        )

    def forward(self, x):
        return self.block(x)


class LRP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(Adapter(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


# ======================== Main Codec ========================

class LatentCodec(nn.Module):
    def __init__(self, ch_emd=128, channel=320, channel_out=128,
                 num_slices=5, max_support_slices=5, **kwargs):
        super().__init__()

        M = channel
        self.M = channel
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices

        self.g_a = AnalysisTransform(ch_emd, channel)
        self.g_s = SynthesisTransform(channel, channel_out)
        self.h_a = HyperAnalysis(channel)
        self.h_s = HyperSynthesis(channel)
        self.aux = AuxDecoder(ch_emd, channel)

        context_dim = M * 2
        self.adapter_in = nn.ModuleList(
            Adapter(in_ch=M, out_ch=context_dim) for _ in range(4))
        self.g_c = SpatialContext(in_ch=context_dim)
        self.adapter_out = nn.ModuleList(
            Adapter(in_ch=context_dim, out_ch=M * 2) for _ in range(4))
        self.LRP = nn.ModuleList(
            LRP(in_ch=M * 2, out_ch=M) for _ in range(4))

        self.entropy_bottleneck = EntropyBottleneck(channel // 2)
        self.gaussian_conditional = GaussianConditional(None)
        self.masks = {}

        self.apply(self._init_weights)

        # Zero-init aux decoder's final conv to prevent shortcut domination
        nn.init.constant_(self.aux.block[-1].weight, 0)
        nn.init.constant_(self.aux.block[-1].bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def aux_loss(self):
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=False):
        update_registered_buffers(
            self.entropy_bottleneck, "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"], state_dict)
        update_registered_buffers(
            self.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
        super().load_state_dict(state_dict, strict=strict)

    # ---- Mask utilities (identical to DiT-IC) ----

    def get_mask_four_parts(self, batch, channel, height, width, device='cuda'):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_m0 = torch.tensor(((1., 0), (0, 0)), device=device)
            m0 = micro_m0.repeat((height + 1) // 2, (width + 1) // 2)[:height, :width].unsqueeze(0).unsqueeze(0)
            micro_m1 = torch.tensor(((0, 1.), (0, 0)), device=device)
            m1 = micro_m1.repeat((height + 1) // 2, (width + 1) // 2)[:height, :width].unsqueeze(0).unsqueeze(0)
            micro_m2 = torch.tensor(((0, 0), (1., 0)), device=device)
            m2 = micro_m2.repeat((height + 1) // 2, (width + 1) // 2)[:height, :width].unsqueeze(0).unsqueeze(0)
            micro_m3 = torch.tensor(((0, 0), (0, 1.)), device=device)
            m3 = micro_m3.repeat((height + 1) // 2, (width + 1) // 2)[:height, :width].unsqueeze(0).unsqueeze(0)

            m = torch.ones((batch, channel // 4, height, width), device=device)
            mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
            mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
            mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
            mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def sequeeze_with_mask(self, latent, mask):
        g1, g2, g3, g4 = latent.chunk(4, 1)
        m1, m2, m3, m4 = mask.chunk(4, 1)
        return g1 * m1 + g2 * m2 + g3 * m3 + g4 * m4

    def unsequeeze_with_mask(self, latent_sq, mask):
        m1, m2, m3, m4 = mask.chunk(4, 1)
        return torch.cat((latent_sq * m1, latent_sq * m2, latent_sq * m3, latent_sq * m4), dim=1)

    def forward_with_mask(self, latent, scales, means, mask):
        latent_sq = self.sequeeze_with_mask(latent, mask)
        scales_sq = self.sequeeze_with_mask(scales, mask)
        means_sq = self.sequeeze_with_mask(means, mask)
        _, y_lk = self.gaussian_conditional(latent_sq, scales_sq, means=means_sq)
        latent_hat = ste_round(latent_sq - means_sq) + means_sq
        return self.unsequeeze_with_mask(latent_hat, mask), self.unsequeeze_with_mask(y_lk, mask)

    def compress_group_with_mask(self, gc, latent, scales, means, mask, sym_list, idx_list):
        latent_sq = self.sequeeze_with_mask(latent, mask)
        scales_sq = self.sequeeze_with_mask(scales, mask)
        means_sq = self.sequeeze_with_mask(means, mask)
        indexes = gc.build_indexes(scales_sq)
        latent_sq_hat = gc.quantize(latent_sq, "symbols", means_sq)
        sym_list.extend(latent_sq_hat.reshape(-1).tolist())
        idx_list.extend(indexes.reshape(-1).tolist())
        return self.unsequeeze_with_mask(latent_sq_hat + means_sq, mask)

    def decompress_group_with_mask(self, gc, scales, means, mask, decoder, cdf, cdf_len, offsets):
        scales_sq = self.sequeeze_with_mask(scales, mask)
        means_sq = self.sequeeze_with_mask(means, mask)
        indexes = gc.build_indexes(scales_sq)
        latent_sq_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_len, offsets)
        latent_sq_hat = torch.Tensor(latent_sq_hat).reshape(scales_sq.shape).to(scales.device)
        return self.unsequeeze_with_mask(latent_sq_hat + means_sq, mask)

    # ---- Forward / Compress / Decompress ----

    def forward(self, latent, latent2):
        y = self.g_a(latent, latent2)
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        base = self.h_s(z_hat)
        means_0, scales_0 = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0, y_lk_0 = self.forward_with_mask(y, scales_0, means_0, mask_0)
        lrp = 0.5 * torch.tanh(self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1, scales_1 = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1, y_lk_1 = self.forward_with_mask(y, scales_1, means_1, mask_1)
        lrp = 0.5 * torch.tanh(self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2, scales_2 = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2, y_lk_2 = self.forward_with_mask(y, scales_2, means_2, mask_2)
        lrp = 0.5 * torch.tanh(self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3, scales_3 = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        y_hat_3, y_lk_3 = self.forward_with_mask(y, scales_3, means_3, mask_3)
        lrp = 0.5 * torch.tanh(self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3)
        y_hat_3 = y_hat_3 + lrp

        y_hat = y_hat_0 + y_hat_1 + y_hat_2 + y_hat_3
        y_likelihoods = y_lk_0 + y_lk_1 + y_lk_2 + y_lk_3

        x_hat = self.g_s(y_hat)
        res = self.aux(y_hat)

        return {
            "x_hat": x_hat,
            "res": res,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, latent, latent2):
        y = self.g_a(latent, latent2)
        z = self.h_a(y)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list, indexes_list, y_strings = [], [], []

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        base = self.h_s(z_hat)
        means_0, scales_0 = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_0, means_0, mask_0, symbols_list, indexes_list)
        lrp = 0.5 * torch.tanh(self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1, scales_1 = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_1, means_1, mask_1, symbols_list, indexes_list)
        lrp = 0.5 * torch.tanh(self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2, scales_2 = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_2, means_2, mask_2, symbols_list, indexes_list)
        lrp = 0.5 * torch.tanh(self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3, scales_3 = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        y_hat_3 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_3, means_3, mask_3, symbols_list, indexes_list)
        lrp = 0.5 * torch.tanh(self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3)
        y_hat_3 = y_hat_3 + lrp

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_strings.append(encoder.flush())
        torch.backends.cudnn.deterministic = False

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        torch.backends.cudnn.deterministic = True
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        B, C, H, W = z_hat.shape
        # h_a 做了 4x 下采样，所以 y 的空间尺寸是 z 的 4 倍
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(
            B, C * 2, H * 4, W * 4, device=z_hat.device)

        base = self.h_s(z_hat)
        means_0, scales_0 = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0 = self.decompress_group_with_mask(self.gaussian_conditional, scales_0, means_0, mask_0, decoder, cdf, cdf_lengths, offsets)
        lrp = 0.5 * torch.tanh(self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1, scales_1 = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1 = self.decompress_group_with_mask(self.gaussian_conditional, scales_1, means_1, mask_1, decoder, cdf, cdf_lengths, offsets)
        lrp = 0.5 * torch.tanh(self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2, scales_2 = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2 = self.decompress_group_with_mask(self.gaussian_conditional, scales_2, means_2, mask_2, decoder, cdf, cdf_lengths, offsets)
        lrp = 0.5 * torch.tanh(self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3, scales_3 = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        y_hat_3 = self.decompress_group_with_mask(self.gaussian_conditional, scales_3, means_3, mask_3, decoder, cdf, cdf_lengths, offsets)
        lrp = 0.5 * torch.tanh(self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3)
        y_hat_3 = y_hat_3 + lrp

        y_hat = y_hat_0 + y_hat_1 + y_hat_2 + y_hat_3
        torch.backends.cudnn.deterministic = False

        x_hat = self.g_s(y_hat)
        res = self.aux(y_hat)

        return {"x_hat": x_hat, "res": res}
