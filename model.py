import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=61):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DetectorEfficiencyTransformer(nn.Module):
    def __init__(
        self, 
        d_model=128, 
        nhead=8, 
        num_layers=4, 
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 特征嵌入层：将单层效率和位置信息映射到 d_model 维度
        self.efficiency_embedding = nn.Linear(1, d_model // 2)
        self.position_embedding = nn.Embedding(61, d_model // 2)
        
        # 动量嵌入层
        self.momentum_embedding = nn.Linear(1, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=62)  # 61 + 1 (momentum token)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 输出层
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, pos_mask, efficiency, momentum):
        """
        Args:
            pos_mask: [batch_size, 61] - 布尔掩码，标记哪些位置有探测器
            efficiency: [batch_size, 61] - 每个位置的单层效率
            momentum: [batch_size, 1] - 粒子动量
        """
        batch_size = pos_mask.size(0)
        
        position_ids = torch.arange(61, device=pos_mask.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.position_embedding(position_ids)

        eff_embed = self.efficiency_embedding(efficiency.unsqueeze(-1))
        
        # 拼接效率和位置嵌入 [batch_size, 61, d_model]
        detector_tokens = torch.cat([eff_embed, pos_embed], dim=-1)
        
        # 创建动量 token [batch_size, 1, d_model]
        momentum_token = self.momentum_embedding(momentum).unsqueeze(1)
        
        # 拼接所有 tokens [batch_size, 62, d_model]
        tokens = torch.cat([momentum_token, detector_tokens], dim=1)
        
        # 添加位置编码
        tokens = self.pos_encoder(tokens)
        tokens = self.layer_norm(tokens)
        
        # 创建注意力掩码：动量token可以关注所有位置，探测器token只在有探测器的位置可见
        src_key_padding_mask = torch.ones(batch_size, 62, dtype=torch.bool, device=pos_mask.device)
        src_key_padding_mask[:, 0] = False
        src_key_padding_mask[:, 1:] = ~pos_mask
        
        encoded = self.transformer_encoder(
            tokens, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 使用动量 token 的输出作为全局表示
        global_repr = encoded[:, 0, :]
        output = self.output_mlp(global_repr)
        
        return output.squeeze(-1)