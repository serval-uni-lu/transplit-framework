from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(n_channels // 4, n_channels)
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = torch.log(torch.tensor(10_000.)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).cuda()
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((torch.cos(emb), torch.sin(emb)), axis=1)

        emb = swish(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, context_channels: int, level: int, n_groups: int = 8):
        super().__init__()

        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)
        self.context_emb = nn.Conv1d(
            context_channels, out_channels,
            kernel_size=2**level+4,
            stride=2**level,
            padding=(2**level+3)//2
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        x = x.transpose(1, 2)
        if c is not None:
            c = c.transpose(1, 2)

        h = swish(self.norm1(x))
        h = self.conv1(h)

        t = self.time_emb(t)
        h += t[:, :, None]
        if c is not None:
            c = self.context_emb(c)
            c = torch.mean(c, dim=-1, keepdim=True)
            h += c
        h = swish(self.norm2(h))
        h = self.conv2(h)

        h = h + self.shortcut(x)
        h = h.transpose(1, 2)
        return h


class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 8, causal: bool = True):
        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.in_projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.out_projection = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k
        self.causal = causal

    def forward(self, x: torch.Tensor, t: torch.Tensor = None, c: torch.Tensor = None):
        _, _ = t, c

        batch_size, length, n_channels = x.shape
        qkv = self.in_projection(x)
        qkv = qkv.view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = qkv.split(self.d_k, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        if self.causal:
            ones = torch.ones((length, length)).cuda()
            mask = torch.tril(ones)
            mask = mask[None, :, :, None]
            attn = attn * mask - 1e9 * (1 - mask)
        attn = F.softmax(attn, dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.out_projection(res)

        res += x

        return res


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, context_channels: int, level: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, context_channels, level)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        x = self.res(x, t, c)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, context_channels: int, level: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels, context_channels, level)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        x = self.res(x, t, c)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int, context_channels: int, level: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, context_channels, level)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, context_channels, level)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        x = self.res1(x, t, c)
        x = self.attn(x)
        x = self.res2(x, t, c)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None, c: torch.Tensor = None):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None, c: torch.Tensor = None):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class UNet(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 context_channels: int = 1,
                 output_channels: int = 1,
                 n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 causal: bool = True,
                 n_blocks: int = 2
                 ):
        super().__init__()

        # TODO: causal control

        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv1d(input_channels, n_channels, kernel_size=3, padding=1)

        self.time_emb = TimeEmbedding(n_channels * 4)
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(
                    in_channels, out_channels,
                    time_channels=n_channels * 4, 
                    context_channels=context_channels,
                    level=i,
                    has_attn=is_attn[i]
                ))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(
            out_channels,
            time_channels=n_channels * 4,
            context_channels=context_channels,
            level=i
        )

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(
                    in_channels, out_channels,
                    time_channels=n_channels * 4,
                    context_channels=context_channels,
                    level=i,
                    has_attn=is_attn[i]
                ))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(
                in_channels, out_channels,
                time_channels=n_channels * 4,
                context_channels=context_channels,
                level=i,
                has_attn=is_attn[i]
            ))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, in_channels)
        self.final = nn.Conv1d(in_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None):
        t = self.time_emb(t)

        x = x.transpose(1, 2)
        x = self.image_proj(x)
        x = x.transpose(1, 2)

        h = [x]
        for m in self.down:
            x = m(x, t, c)
            h.append(x)

        x = self.middle(x, t, c)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t, c)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=-1)
                x = m(x, t, c)

        y = x.transpose(1, 2)
        y = swish(self.norm(y))
        y = self.final(y)
        y = y.transpose(1, 2)

        return y


if __name__ == "__main__":
    model = UNet(n_channels=8, ch_mults=(1, 2, 2, 2), is_attn=(False, True, True, True))
    
    batch_size = 32
    seq_length = 168
    x = torch.randn(batch_size, seq_length, 1)
    t = torch.ones(batch_size, dtype=torch.int64)
    y = model(x, t)
