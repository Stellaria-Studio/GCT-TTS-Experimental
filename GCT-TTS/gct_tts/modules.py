import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel
from .bert_utils import resolve_bert_path   # 新增导入

# ---------- 基础模块 ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---------- 密集图卷积（原版） ----------
class RelationalBatchGCNLayer(nn.Module):
    """关系感知图卷积层，支持多种边类型（密集版）"""
    def __init__(self, hidden_dim, num_relations, num_bases=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.weight_bases = nn.Parameter(torch.empty(num_bases, hidden_dim, hidden_dim))
        self.weight_comp = nn.Parameter(torch.empty(num_relations, num_bases))
        self.bias = nn.Parameter(torch.empty(hidden_dim))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.weight_bases)
        nn.init.xavier_uniform_(self.weight_comp)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj, edge_type):
        B, T, D = x.shape
        out = torch.zeros_like(x)
        for r in range(1, self.num_relations):
            W_r = torch.einsum('b,bij->ij', self.weight_comp[r], self.weight_bases)
            support = torch.matmul(x, W_r)
            mask = (edge_type == r).float()
            adj_r = adj * mask
            out = out + torch.bmm(adj_r, support)
        W_self = torch.mean(self.weight_bases, dim=0)
        out_self = torch.matmul(x, W_self)
        out = out + out_self
        return F.relu(out + self.bias)

# ---------- 稀疏图卷积（新增） ----------
class SparseRelationalBatchGCNLayer(nn.Module):
    """使用稀疏矩阵的关系图卷积层"""
    def __init__(self, hidden_dim, num_relations, num_bases=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.weight_bases = nn.Parameter(torch.empty(num_bases, hidden_dim, hidden_dim))
        self.weight_comp = nn.Parameter(torch.empty(num_relations, num_bases))
        self.bias = nn.Parameter(torch.empty(hidden_dim))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.weight_bases)
        nn.init.xavier_uniform_(self.weight_comp)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj, edge_type):
        B, T, D = x.shape
        out = torch.zeros_like(x)

        # 逐关系处理
        for r in range(1, self.num_relations):
            W_r = torch.einsum('b,bij->ij', self.weight_comp[r], self.weight_bases)
            support = torch.matmul(x, W_r)  # [B, T, D]

            # 对每个样本构造稀疏邻接矩阵并执行稀疏乘法
            for b in range(B):
                mask = (edge_type[b] == r)  # [T, T]
                if not mask.any():
                    continue
                # 获取非零索引
                indices = torch.nonzero(mask, as_tuple=True)  # (row, col)
                values = mask[indices].float()
                size = (T, T)
                sparse_adj = torch.sparse_coo_tensor(torch.stack(indices), values, size, device=x.device)
                # support[b]: [T, D]
                out_b = torch.sparse.mm(sparse_adj, support[b])  # [T, D]
                out[b] += out_b

        # 自连接
        W_self = torch.mean(self.weight_bases, dim=0)
        out_self = torch.matmul(x, W_self)
        out = out + out_self
        return F.relu(out + self.bias)

# ---------- 交叉注意力融合 ----------
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, key_padding_mask=None):
        attn_out, _ = self.cross_attn(x, y, y, key_padding_mask=key_padding_mask)
        return self.norm(x + self.dropout(attn_out))

# ---------- 编码器块 ----------
class NestedGCNTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, num_relations, dropout=0.1, fusion='cross',
                 use_sparse=False, use_adaln=False, checkpointing=False):
        super().__init__()
        self.fusion = fusion
        self.use_sparse = use_sparse
        self.use_adaln = use_adaln
        self.checkpointing = checkpointing

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        if use_sparse:
            self.gcn = SparseRelationalBatchGCNLayer(hidden_dim, num_relations)
        else:
            self.gcn = RelationalBatchGCNLayer(hidden_dim, num_relations)

        self.norm2 = nn.LayerNorm(hidden_dim)

        if fusion == 'cross':
            self.cross_fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)
        else:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, edge_type, padding_mask=None):
        # 自注意力
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # GCN
        gcn_out = self.gcn(x, adj, edge_type)

        # 融合
        if self.fusion == 'cross':
            fused = self.cross_fusion(x, gcn_out, key_padding_mask=padding_mask)
        else:
            gate_value = self.gate(torch.cat([x, gcn_out], dim=-1))
            fused = gate_value * x + (1 - gate_value) * gcn_out
        x = self.norm2(x + self.dropout(fused))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

# ---------- 方差预测器 ----------
class VariancePredictor(nn.Module):
    def __init__(self, hidden_dim, filter_size=512, kernel_size=3, num_layers=6, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        for i in range(num_layers):
            out_dim = filter_size
            conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size//2)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            layers.extend([
                conv,
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        self.conv_layers = nn.Sequential(*layers)
        self.proj = nn.Linear(filter_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        out = self.proj(x).squeeze(-1)
        return out

# ---------- 长度调节器 ----------
class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, durations, max_len=None):
        expanded_batch = []
        for i in range(x.size(0)):
            repeats = durations[i]
            expanded = torch.repeat_interleave(x[i], repeats, dim=0)
            expanded_batch.append(expanded)
        if max_len is None:
            max_len = max(tensor.size(0) for tensor in expanded_batch)
        padded = torch.zeros(x.size(0), max_len, x.size(2)).to(x.device)
        for i, tensor in enumerate(expanded_batch):
            padded[i, :tensor.size(0), :] = tensor
        return padded

# ---------- HiFi-GAN后处理 ----------
class HiFiGANPostNet(nn.Module):
    def __init__(self, n_mel=128, hidden_dim=128, scale=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.block1 = nn.Sequential(
            nn.Conv1d(n_mel, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
        )
        self.out = nn.Conv1d(hidden_dim, n_mel, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        out = self.out(f1 + f2 + f3)
        out = out.transpose(1, 2)
        return out * self.scale

# ---------- 流时长预测器 ----------
class DurationFlow(nn.Module):
    def __init__(self, hidden_dim, n_flows=4):
        super().__init__()
        self.flows = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            ) for _ in range(n_flows)
        ])

    def forward(self, x, log_dur=None, reverse=False):
        if not reverse:
            log_dur = log_dur.unsqueeze(-1)
            log_det_sum = 0
            for flow in self.flows:
                mean, log_std = flow(x).chunk(2, dim=-1)
                z = (log_dur - mean) * torch.exp(-log_std)
                log_dur = z
                log_det_sum += -log_std.squeeze(-1)
            return z.squeeze(-1), log_det_sum
        else:
            z = torch.randn(x.size(0), x.size(1), 1, device=x.device)
            for flow in reversed(self.flows):
                mean, log_std = flow(x).chunk(2, dim=-1)
                z = mean + z * torch.exp(log_std)
            return z.squeeze(-1)

# ---------- 自适应实例归一化 ----------
class AdaIN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(hidden_dim, affine=False, track_running_stats=False)

    def forward(self, x, gamma, beta):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return gamma * x + beta

# ---------- 风格编码器（支持VAE）----------
class StyleEncoder(nn.Module):
    def __init__(self, input_dim, style_dim, use_vae=False):
        super().__init__()
        self.use_vae = use_vae
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        if use_vae:
            self.mean = nn.Linear(128, style_dim)
            self.logvar = nn.Linear(128, style_dim)
        else:
            self.proj = nn.Linear(128, style_dim)

    def forward(self, x):
        # x: [B, T, D] 或 [B, D]
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, D]
        h = self.encoder(x)
        if self.use_vae:
            mu = self.mean(h)
            logvar = self.logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            return z, kl_loss
        else:
            return self.proj(h)

# ---------- BERT编码器封装 ----------
class BertEncoder(nn.Module):
    def __init__(self, model_name, output_dim):
        super().__init__()
        resolved_name = resolve_bert_path(model_name)   # 解析路径
        self.bert = AutoModel.from_pretrained(resolved_name)
        self.bert_dim = self.bert.config.hidden_size
        self.proj = nn.Linear(self.bert_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        return self.dropout(self.proj(last_hidden))

# ---------- 单调对齐模块（简化）----------
class MonotonicAlignment(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)
        # 可学习的位置嵌入作为query
        self.query_pos = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, encoder_out, mel_len=None):
        B, T, D = encoder_out.shape
        queries = self.query_pos.expand(B, -1, -1)  # [B, 1, D]
        attn_output, attn_weights = self.attn(queries, encoder_out, encoder_out)
        # attn_weights: [B, 1, T]
        # 返回原序列和注意力权重（实际应用需按权重扩展）
        return encoder_out, attn_weights

# ---------- 多尺度STFT损失 ----------
class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, scales=[2048, 1024, 512, 256], overlap=0.75):
        super().__init__()
        self.scales = scales

    def forward(self, y_pred, y_true):
        # y_pred, y_true: [B, T, n_mels] 需要转换为波形？这里假设输入已经是mel谱，需要逆变换
        # 实际中需先转换为波形，但此处简化：直接比较幅度谱？不，STFT损失通常用于波形。
        # 这里仅作占位，实际使用时需传入波形或修改。
        # 为完整起见，保留接口。
        loss = 0.0
        for s in self.scales:
            hop = int(s * (1 - overlap))
            win = torch.hann_window(s).to(y_pred.device)
            # 假设y_pred是波形 [B, T]
            spec_pred = torch.stft(y_pred, n_fft=s, hop_length=hop, win_length=s, window=win,
                                    return_complex=True).abs()
            spec_true = torch.stft(y_true, n_fft=s, hop_length=hop, win_length=s, window=win,
                                    return_complex=True).abs()
            loss += F.l1_loss(spec_pred, spec_true)
        return loss / len(self.scales)

# ---------- Mel判别器 ----------
class MelDiscriminator(nn.Module):
    def __init__(self, n_mel=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, (3,3), padding=(1,1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, (3,3), stride=(2,1), padding=(1,1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3,3), stride=(2,1), padding=(1,1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3,3), padding=(1,1)),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, mel):
        mel = mel.unsqueeze(1)  # [B, 1, T, n_mels]
        out = self.convs(mel)   # [B, 128, T', 1?] 需调整
        out = out.mean(dim=[2,3])
        return self.fc(out)