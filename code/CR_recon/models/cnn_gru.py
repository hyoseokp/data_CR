"""
MetaSpec_CNNGRU: CNN backbone + GRU (간단한 baseline 모델)
입력: (B, 1, 128, 128) float32
출력: (B, 2, 2, 30) float32 (BGGR spectra)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c, groups=16):
    """GroupNorm with adaptive group count."""
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


class MetaSpec_CNNGRU(nn.Module):
    """
    CNN backbone + GRU for sequential spectrum prediction.
    간단한 비교 모델로, MetaSpec_CNNXAttn의 Transformer decoder를 GRU로 대체.
    """

    def __init__(
        self,
        out_len=30,
        d_model=128,
        gru_layers=2,
        cnn_dropout=0.05,
        head_dropout=0.1,
        use_circular_padding=True,
    ):
        super().__init__()
        self.out_len = out_len
        self.d_model = d_model
        self.use_circular_padding = use_circular_padding

        # Stem: 128 → 64 (5x5 conv with circular padding)
        self.stem_conv = nn.Conv2d(1, 64, 5, stride=2, padding=0, bias=False)
        self.stem_gn = _gn(64)
        self.stem_dropout = nn.Dropout2d(cnn_dropout) if cnn_dropout > 0 else nn.Identity()

        # Stages with circular padding
        self.stage1 = nn.Sequential(
            ResBlockGN(64, 96, stride=2, dropout=cnn_dropout, use_circular=use_circular_padding),
            ResBlockGN(96, 96, stride=1, dropout=cnn_dropout, use_circular=use_circular_padding),
        )
        self.stage2 = nn.Sequential(
            ResBlockGN(96, 128, stride=2, dropout=cnn_dropout, use_circular=use_circular_padding),
            ResBlockGN(128, 128, stride=1, dropout=cnn_dropout, use_circular=use_circular_padding),
        )
        self.stage3 = nn.Sequential(
            ResBlockGN(128, 192, stride=2, dropout=cnn_dropout, use_circular=use_circular_padding),
            ResBlockGN(192, 192, stride=1, dropout=cnn_dropout, use_circular=use_circular_padding),
        )
        self.stage4 = nn.Sequential(
            ResBlockGN(192, 256, stride=2, dropout=cnn_dropout, use_circular=use_circular_padding),
            ResBlockGN(256, 256, stride=1, dropout=cnn_dropout, use_circular=use_circular_padding),
        )

        # Global context
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_z = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(256, d_model),
            nn.SiLU(),
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
        )

        # GRU for spectrum prediction
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=gru_layers,
            batch_first=True,
            dropout=head_dropout if gru_layers > 1 else 0,
        )

        # Output head
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
        # CNN backbone
        # Stem with circular padding (5x5 conv needs padding=2, respects diagonal periodicity)
        if self.use_circular_padding:
            x = circular_pad_2d(x, pad_h=2, pad_w=2)
        else:
            x = F.pad(x, (2, 2, 2, 2), mode='constant', value=0)
        f = self.stem_conv(x)
        f = self.stem_gn(f)
        f = F.silu(f)
        f = self.stem_dropout(f)

        f = self.stage1(f)
        f = self.stage2(f)
        f = self.stage3(f)
        f = self.stage4(f)

        # Global context
        z = self.to_z(self.pool(f))  # (B, d_model)

        # GRU 입력: (B, out_len, d_model)
        # z를 out_len번 반복하여 시퀀스 구성
        z_seq = z.unsqueeze(1).expand(-1, self.out_len, -1)  # (B, out_len, d_model)

        # GRU 처리
        gru_out, _ = self.gru(z_seq)  # (B, out_len, d_model)

        # 각 시간 스텝에 대해 head 적용
        y4 = self.head(gru_out)  # (B, out_len, 4)
        y4 = y4.permute(0, 2, 1) + self.spec_bias  # (B, 4, out_len)
        return y4.view(-1, 2, 2, self.out_len)
