#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCT-TTS 推理脚本（修复：无声音、模型崩塌、外部词表加载）
- 自动读取数据预处理阶段生成的 vocab.pt
- 统一语言模型为英语（针对 LJSpeech）
- 兼容 PyTorch 2.6+ weights_only 问题
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from gct_tts import GCTTTS, TextGraphFrontend


# ---------- 模型加载（增加读取外部 vocab.pt 的逻辑） ----------
def load_model(checkpoint_path, device, vocab_path=None):
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 提取配置
    config = checkpoint.get('config', {})
    if not config:
        print("警告：checkpoint 中无 config 信息，将尝试推断参数...")

    # --- 关键修复：从外部文件加载词表 ---
    vocab_dict = checkpoint.get('vocab', None)
    if vocab_dict is None:
        if vocab_path and os.path.exists(vocab_path):
            print(f"正在从外部路径加载词表: {vocab_path}")
            # 读取保存的词表字典
            vocab_dict = torch.load(vocab_path, map_location='cpu', weights_only=False)
            print(f"成功加载外部词表，包含 {len(vocab_dict)} 个词元。")
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(" 严重警告: 未找到任何词表 (vocab)！")
            print(f" 指定的外部词表路径 ({vocab_path}) 也不存在。")
            print(" 推理将因词表未对齐导致严重杂音或频谱崩塌！")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise FileNotFoundError(f"找不到词表文件: {vocab_path}")
    else:
        print(f"成功从 checkpoint 内部加载词表，包含 {len(vocab_dict)} 个词元。")

    # 获取或推断 vocab_size
    vocab_size = config.get('vocab_size')
    if vocab_size is None and vocab_dict is not None:
        vocab_size = len(vocab_dict)
        print(f"根据加载的词表，推断 vocab_size = {vocab_size}")
    elif vocab_size is None:
        raise ValueError("无法确定 vocab_size，模型无法初始化。")

    hidden_dim = config.get('hidden_dim', 384)
    num_layers = config.get('num_layers', 8)
    num_relations = config.get('num_relations', 50)  # 如果你报错少参数，这里可以根据报错改
    num_decoder_layers = config.get('num_decoder_layers', 4)
    num_heads = config.get('num_heads', 8)
    ffn_dim = config.get('ffn_dim', 1536)
    fusion = config.get('fusion', 'cross')

    # 说话人数量推断
    state_dict = checkpoint['model_state_dict']
    if 'speaker_embedding.weight' in state_dict:
        num_speakers = state_dict['speaker_embedding.weight'].size(0)
    else:
        num_speakers = 1

    # 推断是否使用流时长
    use_flow_duration = any('duration_predictor.flows' in k for k in state_dict.keys())
    flow_n_flows = 4
    if use_flow_duration:
        flow_indices = set()
        for k in state_dict.keys():
            if 'duration_predictor.flows' in k:
                parts = k.split('.')
                for i, part in enumerate(parts):
                    if part == 'flows' and i + 1 < len(parts):
                        try:
                            idx = int(parts[i + 1])
                            flow_indices.add(idx)
                        except:
                            pass
        if flow_indices:
            flow_n_flows = max(flow_indices) + 1
        print(f"检测到流时长预测器，流数量: {flow_n_flows}")

    # 创建模型
    model = GCTTTS(
        vocab_size=vocab_size,
        num_speakers=num_speakers,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_relations=num_relations,
        num_decoder_layers=num_decoder_layers,
        use_flow_duration=use_flow_duration,
        flow_n_flows=flow_n_flows,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        fusion=fusion
    )

    # 严格加载权重
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    mel_stats = checkpoint.get('mel_stats', {'min': -12.0, 'max': 2.0})
    pitch_stats = checkpoint.get('pitch_stats', {'mean': 0.0, 'std': 1.0})

    return model, mel_stats, pitch_stats, vocab_dict


# ---------- 文本处理 ----------
def text_to_input(text, frontend, device):
    seq_ids, adj_matrix, edge_type = frontend.text_to_sequence_and_graph(text)
    seq_ids = seq_ids.unsqueeze(0).to(device)
    adj_matrix = adj_matrix.unsqueeze(0).to(device)
    edge_type = edge_type.unsqueeze(0).to(device)
    return seq_ids, adj_matrix, edge_type


# ---------- 推理 ----------
@torch.no_grad()
def infer(model, seq_ids, adj_matrix, edge_type, device):
    mel_pred, log_dur_pred, pitch_pred = model(
        seq_ids, adj_matrix, edge_type,
        speaker_ids=None,
        target_durations=None,
        src_padding_mask=None
    )
    return mel_pred, log_dur_pred, pitch_pred


# ---------- 保存结果 ----------
def save_mel_plot(mel, path, title="Mel spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel.T, sr=22050, hop_length=256,
                             x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"梅尔谱图已保存: {path}")


def save_mel_npy(mel, path):
    np.save(path, mel)
    print(f"梅尔谱 npy 已保存: {path}")


def generate_audio(mel, mel_stats, sr=22050, hop_length=256, n_iter=64):
    """
    包含反归一化逻辑的 Griffin-Lim 音频生成
    """
    # 1. 反归一化 (Denormalization)
    # 如果你训练时将 mel 缩放到了 [-1, 1]，这里需要还原到真实的 Log-Mel 范围 (通常是 -12 到 2)
    # 以下是一个通用的 Min-Max 反归一化假设：
    mel_min = mel_stats.get('min', -11.5129)
    mel_max = mel_stats.get('max', 2.0)

    # 假设训练时做了归一化： mel_norm = 2 * (mel - mel_min) / (mel_max - mel_min) - 1
    # 那么反归一化就是：
    denorm_mel = (mel + 1.0) / 2.0 * (mel_max - mel_min) + mel_min

    # 2. 将 Log-Mel 转换为线性的幅度/功率 (必须大于0)
    mel_linear = np.exp(denorm_mel)
    mel_linear = np.maximum(mel_linear, 1e-5)  # 防止 0 导致算法崩溃

    # 3. Griffin-Lim 算法
    audio = librosa.feature.inverse.mel_to_audio(
        mel_linear.T, sr=sr, hop_length=hop_length, n_iter=n_iter
    )

    # 4. 音量归一化 (防止爆音)
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = (audio / max_amp) * 0.95

    return audio


# ---------- 主函数 ----------
def main():
    parser = argparse.ArgumentParser(description="GCT-TTS 推理")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/ljspeech/gct_tts_epoch_105.pt",
                        help="模型 checkpoint 路径")

    # 增加独立的 vocab_path 参数，默认指向你提供的位置
    parser.add_argument("--vocab_path", type=str,
                        default="data/features_ljspeech/vocab.pt",
                        help="预处理阶段生成的词表文件路径")

    # 必须使用英文文本
    parser.add_argument("--text", type=str,
                        default="The weather is really nice today. What do you think?",
                        help="输入文本 (LJSpeech 必须使用英文)")

    parser.add_argument("--language", type=str, default="en_core_web_sm",
                        help="spaCy 语言模型")

    parser.add_argument("--output_dir", type=str, default="inference_output",
                        help="输出目录")
    parser.add_argument("--device", type=str, default=None,
                        help="运算设备 (cuda/mps/cpu)")
    parser.add_argument("--save_audio", action="store_true", default=True)
    parser.add_argument("--save_mel_npy", action="store_true", default=True)
    parser.add_argument("--save_mel_plot", action="store_true", default=True)
    parser.add_argument("--griffin_lim_iters", type=int, default=64)
    args = parser.parse_args()

    # 设备选择
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # MPS fallback
    if device.type == 'mps':
        try:
            _ = torch.ones(1, device=device)
        except:
            print("MPS 不可用，回退到 CPU")
            device = torch.device('cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型和词表（这里会读取 args.vocab_path）
    try:
        model, mel_stats, pitch_stats, vocab_dict = load_model(args.checkpoint, device, args.vocab_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)

    # 初始化前端（传入已加载好的词表）
    print("初始化文本前端...")
    try:
        frontend = TextGraphFrontend(language=args.language, vocab_dict=vocab_dict)
    except OSError:
        print(f"缺少语言包。请先运行：python -m spacy download {args.language}")
        sys.exit(1)

    # 处理输入文本
    print(f"输入文本: {args.text}")
    seq_ids, adj_matrix, edge_type = text_to_input(args.text, frontend, device)

    # 【新增调试信息】：检查 Token ID 是否对齐
    print(f"解析后的 token IDs: {seq_ids[0].tolist()}")
    if seq_ids[0].max().item() <= 1:
        print("警告：所有的单词都被解析为了 [UNK] (ID=1) 或 [PAD] (ID=0)！")
        print("这说明你的词表没有涵盖这些单词，或者是大小写/标点符号没对齐。")

    # 推理
    print("推理中...")
    mel_pred, log_dur_pred, pitch_pred = infer(model, seq_ids, adj_matrix, edge_type, device)
    mel = mel_pred.squeeze(0).cpu().numpy()  # [T, 128]
    print(f"生成梅尔谱形状: {mel.shape}, 数值范围:[{mel.min():.3f}, {mel.max():.3f}]")

    base_name = "output"

    if args.save_mel_npy:
        save_mel_npy(mel, os.path.join(args.output_dir, f"{base_name}_mel.npy"))

    if args.save_mel_plot:
        save_mel_plot(mel, os.path.join(args.output_dir, f"{base_name}_mel.png"),
                      title=f"Mel spectrogram: {args.text}")

    if args.save_audio:
        print("生成音频 (Griffin-Lim)...")
        # 注意这里多传了一个 mel_stats
        audio = generate_audio(mel, mel_stats, sr=22050, hop_length=256, n_iter=args.griffin_lim_iters)
        print(f"音频幅度范围:[{audio.min():.3f}, {audio.max():.3f}]")

        audio_path = os.path.join(args.output_dir, f"{base_name}_audio.wav")
        sf.write(audio_path, audio, samplerate=22050)
        print(f"音频已保存: {audio_path}")
    print("推理完成。")


if __name__ == "__main__":
    main()