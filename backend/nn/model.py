from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """标准的残差块: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 残差连接：让梯度能传得更深
        out = F.relu(out)
        return out

class AmazonsNet(nn.Module):
    """
    Improved Network:
      - Trunk: Conv + BN + ReLU + [ResBlocks]
      - Heads: More capacity with BN
    """

    def __init__(self, channels: int = 64, num_res_blocks: int = 5):
        super().__init__()
        
        # 1. 初始卷积块 (Input -> 64 channels)
        self.conv_input = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # 2. 残差塔 (ResNet Tower)
        # num_res_blocks 越大越聪明，但也越慢。
        # CPU/轻量级GPU建议 5-6层；强力显卡建议 10-20层。
        self.res_tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # 3. Value Head (估值头)
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), # 压缩通道 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # 输出 [-1, 1]
        )

        # 4. Policy Head (策略头)
        # 输入: Board Features (64) + Action Planes (3) = 67 channels
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels + 3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.p_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Logit
        )

    def forward_features(self, x_board: torch.Tensor) -> torch.Tensor:
        """提取棋盘特征，供后续多次使用"""
        x = self.conv_input(x_board)
        x = self.res_tower(x)
        return x

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        """根据特征评估胜率"""
        return self.v_head(feat).squeeze(-1)

    def policy_logit(self, feat: torch.Tensor, x_action: torch.Tensor) -> torch.Tensor:
        """
        评估某个动作的好坏
        feat: 来自 forward_features 的缓存特征 (B, 64, 10, 10)
        x_action: 动作特征 (B, 3, 10, 10)
        """
        # 拼接棋盘特征和动作特征
        z = torch.cat([feat, x_action], dim=1) 
        z = self.p_conv(z)
        logit = self.p_fc(z)
        return logit.squeeze(-1)