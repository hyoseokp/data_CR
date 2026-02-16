"""
MetaSpec_CNNXAttn: CNN backbone + Transformer decoder with cross-attention
입력: (B, 1, 128, 128) float32
출력: (B, 2, 2, 30) float32 (BGGR spectra)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c, groups=16):
    """GroupNorm with adaptive group count (ensure c is divisible by g)."""
    g = min(groups, c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)


def circular_pad_2d(x, pad_h, pad_w):
    """
    2D Circular padding that respects diagonal periodicity.
    대각선 주기성까지 고려한 2D 원형 패딩.

    x: (B, C, H, W)
    pad_h, pad_w: (top, bottom) and (left, right) padding sizes
                 단, top=bottom, left=right 가정 (symmetric padding)

    작동 방식:
    1. 상하 방향: top과 bottom rows를 wrap
    2. 좌우 방향: left와 right columns을 wrap
    3. 결과: 꼭짓점도 대각선 정보 반영
    """
    if isinstance(pad_h, int):
        pad_h = (pad_h, pad_h)
    if isinstance(pad_w, int):
        pad_w = (pad_w, pad_w)

    top, bottom = pad_h
    left, right = pad_w

    B, C, H, W = x.shape

    # Step 1: Vertical circular padding (H 방향)
    if top > 0 or bottom > 0:
        x_top = x[:, :, -top:, :] if top > 0 else torch.empty(B, C, 0, W, device=x.device)
        x_bottom = x[:, :, :bottom, :] if bottom > 0 else torch.empty(B, C, 0, W, device=x.device)
        x = torch.cat([x_top, x, x_bottom], dim=2)

    # Step 2: Horizontal circular padding (W 방향)
    if left > 0 or right > 0:
        x_left = x[:, :, :, -left:] if left > 0 else torch.empty(B, C, x.shape[2], 0, device=x.device)
        x_right = x[:, :, :, :right] if right > 0 else torch.empty(B, C, x.shape[2], 0, device=x.device)
        x = torch.cat([x_left, x, x_right], dim=3)

    return x


class ResBlockGN(nn.Module):
    """Residual block with GroupNorm, optional dropout, and circular padding."""

    def __init__(self, in_ch, out_ch, stride=1, dropout=0.05, use_circular=True):
        super().__init__()
        self.use_circular = use_circular

        # Conv without padding, we'll pad manually
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=0, bias=False)
        self.gn1 = _gn(out_ch)
        self.drop1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=0, bias=False)
        self.gn2 = _gn(out_ch)
        self.drop2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        identity = x

        # Apply circular padding manually before conv (respects diagonal periodicity)
        if self.use_circular:
            x = circular_pad_2d(x, pad_h=1, pad_w=1)
        else:
            x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)

        x = F.silu(self.gn1(self.conv1(x)))
        x = self.drop1(x)

        if self.use_circular:
            x = circular_pad_2d(x, pad_h=1, pad_w=1)
        else:
            x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)

        x = self.gn2(self.conv2(x))
        x = self.drop2(x)
        if self.skip is not None:
            identity = self.skip(identity)
        return F.silu(x + identity)


class XAttnDecoderBlock(nn.Module):
    """Transformer decoder block: self-attention + cross-attention + FFN (pre-norm)."""

    def __init__(self, d_model=256, nhead=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        hidden = ff_mult * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tgt, memory):
        """tgt: (L, B, D), memory: (HW, B, D)"""
        x = self.norm1(tgt)
        x2, _ = self.self_attn(x, x, x, need_weights=False)
        tgt = tgt + x2
        x = self.norm2(tgt)
        x2, _ = self.cross_attn(x, memory, memory, need_weights=False)
        tgt = tgt + x2
        x = self.norm3(tgt)
        tgt = tgt + self.ff(x)
        return tgt


class MetaSpec_CNNXAttn(nn.Module):
    """
    CNN backbone (5 stages, 128→4 spatial) + Transformer decoder with cross-attention.
    입력: (B, 1, 128, 128)
    출력: (B, 2, 2, 30)
    """

    def __init__(
        self,
        out_len=30,
        d_model=256,
        nhead=8,
        dec_layers=4,
        ff_mult=4,
        cnn_dropout=0.05,
        tr_dropout=0.1,
        head_dropout=0.2,
        enforce_diag_sym=False,
        use_circular_padding=True,
    ):
        super().__init__()
        self.out_len = out_len
        self.d_model = d_model
        self.enforce_diag_sym = enforce_diag_sym
        self.use_circular_padding = use_circular_padding

        # Stem: 128 → 64 (5x5 conv with circular padding)
        self.stem_conv = nn.Conv2d(1, 64, 5, stride=2, padding=0, bias=False)
        self.stem_gn = _gn(64)
        self.stem_dropout = nn.Dropout2d(cnn_dropout) if cnn_dropout > 0 else nn.Identity()

        # Stage 1: 64 → 32
        self.stage1 = nn.Sequential(
            ResBlockGN(64, 96, stride=2, dropout=cnn_dropout, use_circular=use_circular_padding),
            ResBlockGN(96, 96, stride=1, dropout=cnn_dropout, use_circular=use_circular_padding),
        )

        # Stage 2: 32 → 16
        self.stage2 = nn.Sequential(
            ResBlockGN(96, 128, stride=2, dropout=cnn_dropout, use_circular=use_circular_padding),
            ResBlockGN(128, 128, stride=1, dropout=cnn_dropout, use_circular=use_circular_padding),
        )

        # Stage 3: 16 → 8
        self.stage3 = nn.Sequential(
            ResBlockGN(128, 192, stride=2, dropout=cnn_dropout, use_circular=use_circular_padding),
            ResBlockGN(192, 192, stride=1, dropout=cnn_dropout, use_circular=use_circular_padding),
        )

        # Stage 4: 8 → 4
        self.stage4 = nn.Sequential(
            ResBlockGN(192, 256, stride=2, dropout=cnn_dropout, use_circular=use_circular_padding),
            ResBlockGN(256, 256, stride=1, dropout=cnn_dropout, use_circular=use_circular_padding),
        )

        # Spatial feature projection
        self.s_proj = nn.Conv2d(256, d_model, 1, bias=False)
        self.s_pos = nn.Parameter(torch.randn(16, 1, d_model) * 0.02)  # 16 = 4x4

        # Global context: average pooling + MLP
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_z = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(256, d_model),
            nn.SiLU(),
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
        )

        # Wavelength tokens: learned query + position encoding
        self.q_film = nn.Linear(d_model, d_model)  # FiLM: feature-wise linear modulation
        self.query = nn.Parameter(torch.randn(out_len, d_model) * 0.02)
        self.w_pos = nn.Parameter(torch.randn(out_len, 1, d_model) * 0.02)

        # Transformer decoder
        self.dec = nn.ModuleList(
            [
                XAttnDecoderBlock(
                    d_model=d_model, nhead=nhead, ff_mult=ff_mult, dropout=tr_dropout
                )
                for _ in range(dec_layers)
            ]
        )

        # Output head: (L, B, D) → (B, 4, L)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Linear(d_model, 4),
        )
        self.spec_bias = nn.Parameter(torch.zeros(1, 4, out_len))

    def forward(self, x):
        """
        x: (B, 1, 128, 128) float32
        Returns: (B, 2, 2, out_len) float32
        """
        if self.enforce_diag_sym:
            x = 0.5 * (x + x.transpose(-1, -2))

        # CNN backbone
        # Stem with circular padding (5x5 conv needs padding=2, respects diagonal periodicity)
        if self.use_circular_padding:
            x = circular_pad_2d(x, pad_h=2, pad_w=2)
        else:
            x = F.pad(x, (2, 2, 2, 2), mode='constant', value=0)
        f = self.stem_conv(x)
        f = self.stem_gn(f)
        f = nn.functional.silu(f)
        f = self.stem_dropout(f)  # (B, 64, 64, 64)
        f = self.stage1(f)  # (B, 96, 32, 32)
        f = self.stage2(f)  # (B, 128, 16, 16)
        f = self.stage3(f)  # (B, 192, 8, 8)
        f = self.stage4(f)  # (B, 256, 4, 4)

        # Spatial features: (B, 256, 4, 4) → (B, d_model, 4, 4)
        s = self.s_proj(f)
        B, D, H, W = s.shape
        s = s.flatten(2).permute(2, 0, 1).contiguous()  # (16, B, d_model)
        s = s + self.s_pos

        # Global context: (B, 256, 4, 4) → (B, d_model)
        z = self.to_z(self.pool(f))

        # Wavelength tokens with context modulation
        q = self.query.unsqueeze(1).expand(self.out_len, B, -1)  # (L, B, d_model)
        q = q + self.w_pos
        q = q + self.q_film(z).unsqueeze(0)

        # Transformer decoder
        for blk in self.dec:
            q = blk(q, s)  # self-attention + cross-attention to spatial features

        # Output head: (L, B, d_model) → (B, 4, L)
        y4 = self.head(q).permute(1, 2, 0) + self.spec_bias
        y4 = torch.sigmoid(y4)  # [0, 1] — 물리적 범위 보장
        return y4.view(B, 2, 2, self.out_len)
