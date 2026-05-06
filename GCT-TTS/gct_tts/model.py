import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from .modules import (
    PositionalEncoding,
    SparseRelationalBatchGCNLayer,
    RelationalBatchGCNLayer,
    NestedGCNTransformerBlock,
    VariancePredictor,
    LengthRegulator,
    HiFiGANPostNet,
    DurationFlow,
    AdaIN,
    StyleEncoder,
    BertEncoder,
    MonotonicAlignment,
)
from .bert_utils import resolve_bert_path   # 新增导入

class GCTTTS(nn.Module):
    def __init__(self, vocab_size, num_speakers=1, hidden_dim=512,
                 num_layers=12, num_relations=1, num_decoder_layers=6,
                 use_flow_duration=False, flow_n_flows=4,
                 num_heads=8, ffn_dim=2048, fusion='cross',
                 use_bert=False, bert_model_name='bert-base-multilingual-cased',
                 use_sparse_gcn=True, use_adaln=True, use_style_emb=True,
                 style_dim=128, use_vae=False, use_implicit_align=False,
                 checkpointing=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_flow_duration = use_flow_duration
        self.use_bert = use_bert
        self.use_sparse_gcn = use_sparse_gcn
        self.use_adaln = use_adaln
        self.use_style_emb = use_style_emb
        self.use_vae = use_vae
        self.use_implicit_align = use_implicit_align
        self.checkpointing = checkpointing

        # 文本编码
        if use_bert:
            self.bert_encoder = BertEncoder(bert_model_name, hidden_dim)
            # 注意：删除了原有的 self.vocab_proj，因为 BertEncoder 已经投影到 hidden_dim
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim)

        # 说话人编码
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, hidden_dim)
        else:
            self.speaker_embedding = None

        # 风格编码
        if use_style_emb:
            self.style_encoder = StyleEncoder(hidden_dim, style_dim, use_vae=use_vae)

        # 编码器层（使用稀疏GCN或密集GCN）
        self.encoder_layers = nn.ModuleList([
            NestedGCNTransformerBlock(
                hidden_dim, num_heads=num_heads, ffn_dim=ffn_dim,
                num_relations=num_relations, dropout=0.1, fusion=fusion,
                use_sparse=use_sparse_gcn, use_adaln=use_adaln,
                checkpointing=checkpointing
            ) for _ in range(num_layers)
        ])

        # 时长预测
        if use_flow_duration:
            self.duration_predictor = DurationFlow(hidden_dim, n_flows=flow_n_flows)
        else:
            self.duration_predictor = VariancePredictor(hidden_dim, num_layers=6, dropout=0.2)

        # 音高预测
        self.pitch_predictor = VariancePredictor(hidden_dim, num_layers=6, dropout=0.2)

        # 长度调节器（扩展文本序列到mel长度）
        self.length_regulator = LengthRegulator()

        # 隐式对齐模块（可选）
        if use_implicit_align:
            self.alignment = MonotonicAlignment(hidden_dim)

        # 解码器输入投影（融合扩展后的文本、音高、风格、说话人）
        style_input_dim = style_dim if use_style_emb else 0
        self.decoder_input_proj = nn.Linear(hidden_dim + 1 + style_input_dim, hidden_dim)

        # Transformer解码器
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            batch_first=True, dropout=0.2, activation='gelu'
        )
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # Mel投影
        self.mel_proj = nn.Linear(hidden_dim, 128)
        self.mel_scale = nn.Parameter(torch.tensor(1.0))
        self.mel_bias = nn.Parameter(torch.tensor(0.0))

        # 后处理网络
        self.postnet = HiFiGANPostNet(n_mel=128, hidden_dim=128, scale=0.1)

        # 自适应层归一化（AdaIN）参数
        if use_adaln and num_speakers > 1:
            self.adaln_gamma = nn.Embedding(num_speakers, hidden_dim)
            self.adaln_beta = nn.Embedding(num_speakers, hidden_dim)
            nn.init.ones_(self.adaln_gamma.weight)
            nn.init.zeros_(self.adaln_beta.weight)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, text_seq, adj_matrix, edge_type, speaker_ids=None,
                target_durations=None, src_padding_mask=None,
                return_attn=False, style_target=None):
        # 文本编码
        if self.use_bert:
            # text_seq 是 input_ids
            attention_mask = ~src_padding_mask if src_padding_mask is not None else None
            x = self.bert_encoder(text_seq, attention_mask=attention_mask)
            # 已投影到 hidden_dim，不再需要额外投影
        else:
            x = self.embedding(text_seq) * math.sqrt(self.hidden_dim)
            x = self.pos_encoder(x)

        B, T, D = x.shape

        # 风格编码
        style = None
        kl_loss = None
        if self.use_style_emb:
            if style_target is not None:
                style = style_target  # 训练时提供风格目标
            else:
                # 推理时从文本编码平均池化提取风格（简单做法）
                style = self.style_encoder(x.mean(dim=1))
            if self.use_vae:
                style, kl_loss = style  # style_encoder返回(style, kl_loss)

        # 说话人自适应（AdaIN）
        if self.use_adaln and speaker_ids is not None and hasattr(self, 'adaln_gamma'):
            gamma = self.adaln_gamma(speaker_ids).unsqueeze(1)  # [B,1,D]
            beta = self.adaln_beta(speaker_ids).unsqueeze(1)
            # 对x进行自适应归一化
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = gamma * (x - mean) / (std + 1e-5) + beta
        elif speaker_ids is not None and self.speaker_embedding is not None:
            spk_emb = self.speaker_embedding(speaker_ids).unsqueeze(1)
            x = x + spk_emb

        # 编码器（使用激活检查点）
        for layer in self.encoder_layers:
            if self.checkpointing and self.training:
                x = checkpoint(layer, x, adj_matrix, edge_type, src_padding_mask, use_reentrant=False)
            else:
                x = layer(x, adj_matrix, edge_type, padding_mask=src_padding_mask)

        # 时长预测
        if self.use_flow_duration:
            if target_durations is not None:
                log_dur_pred, log_det = self.duration_predictor(x, log_dur=torch.log(target_durations.float()+1), reverse=False)
            else:
                log_dur_pred = self.duration_predictor(x, reverse=True)
        else:
            log_dur_pred = self.duration_predictor(x)

        # 音高预测
        pitch_pred = self.pitch_predictor(x)

        # 隐式对齐或显式扩展
        if self.use_implicit_align and target_durations is None:
            # 需要知道目标mel长度，此处简化：假设解码器逐步生成，我们使用注意力机制
            # 实际中可能需要传入mel长度或动态生成
            # 这里演示：直接使用原始序列，不扩展（仅用于占位）
            x_expanded = x
            pitch_expanded = pitch_pred.unsqueeze(-1)
            attn_weights = None
        else:
            if target_durations is not None:
                durations = target_durations
            else:
                dur_float = torch.exp(log_dur_pred) - 1
                durations = torch.clamp(torch.round(dur_float), min=1, max=20).long()
            x_expanded = self.length_regulator(x, durations)
            pitch_expanded = self.length_regulator(pitch_pred.unsqueeze(-1), durations)
            # 合并
            x_expanded = torch.cat([x_expanded, pitch_expanded], dim=-1)

        # 融合风格
        if style is not None:
            style_expanded = style.unsqueeze(1).expand(-1, x_expanded.size(1), -1)
            x_expanded = torch.cat([x_expanded, style_expanded], dim=-1)

        x_expanded = self.decoder_input_proj(x_expanded)
        x_expanded = x_expanded * math.sqrt(self.hidden_dim)
        if not self.use_bert:  # 如果用了BERT，可能已包含位置信息，但也可再加
            x_expanded = self.pos_encoder(x_expanded)
        x_expanded = self.decoder_transformer(x_expanded)

        mel_rough = self.mel_proj(x_expanded) * self.mel_scale + self.mel_bias
        mel_fine = self.postnet(mel_rough)
        mel_final = mel_rough + mel_fine

        outputs = {
            'mel': mel_final,
            'log_dur': log_dur_pred,
            'pitch': pitch_pred,
        }
        if self.use_flow_duration and target_durations is not None:
            outputs['log_det'] = log_det
        if kl_loss is not None:
            outputs['kl_loss'] = kl_loss
        if return_attn and self.use_implicit_align:
            outputs['attn_weights'] = attn_weights
        return outputs