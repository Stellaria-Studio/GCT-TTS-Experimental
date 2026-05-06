import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import numpy as np

from gct_tts import GCTTTS, GCTDataset, MultiDataset, gct_collate_fn
from gct_tts.frontend import TextGraphFrontend
from gct_tts.modules import MultiScaleSTFTLoss, MelDiscriminator


def noam_scheduler(optimizer, warmup_steps=4000, hidden_dim=256):
    def lr_lambda(step):
        step = max(step, 1)
        return (hidden_dim ** (-0.5)) * min(step ** (-0.5), step * (warmup_steps ** (-1.5)))

    return LambdaLR(optimizer, lr_lambda)


def load_mel_channel_stats(feature_dir):
    """加载预处理阶段保存的Mel频带均值和标准差"""
    stats_path = os.path.join(feature_dir, 'mel_channel_stats.pt')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"未找到Mel通道统计文件 {stats_path}，请先运行预处理并启用统计计算。")
    stats = torch.load(stats_path, weights_only=False)
    mel_mean = stats['mean']  # [n_mels] numpy array
    mel_std = stats['std']
    print(
        f"加载Mel通道统计量: mean范围 [{mel_mean.min():.3f}, {mel_mean.max():.3f}], std范围 [{mel_std.min():.3f}, {mel_std.max():.3f}]")
    return mel_mean, mel_std


def load_pitch_stats(feature_dir):
    stats_path = os.path.join(feature_dir, 'pitch_stats.pt')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"未找到音高统计文件 {stats_path}")
    stats = torch.load(stats_path, weights_only=False)
    print(f"加载音高统计量: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    return stats['mean'], stats['std']


def get_num_relations(dataset):
    max_rel = 0
    for i in range(min(len(dataset), 100)):
        edge_type = dataset[i]['edge_type']
        max_rel = max(max_rel, edge_type.max().item())
    return max_rel + 1


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 自动估算批大小
    if args.batch_size == 'auto':
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            est_sample_mem = args.hidden_dim * 1000  # 粗略估计
            max_batch = int(free_mem * 0.4 / est_sample_mem)
            args.batch_size = max(1, min(max_batch, 64))
            print(f"自动调整批大小为: {args.batch_size}")
        else:
            args.batch_size = 4
    else:
        args.batch_size = int(args.batch_size)

    # ---------- 加载数据集 ----------
    print("加载数据集...")
    if args.multi_dataset:
        dataset = MultiDataset(args.metadata_path, args.feature_dir, args.has_speaker, args.use_hdf5)
    else:
        dataset = GCTDataset(args.metadata_path, args.feature_dir, args.has_speaker, args.use_hdf5)

    if len(dataset) == 0:
        raise RuntimeError("数据集为空，请检查预处理步骤。")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=gct_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )

    # 获取统计信息
    first_feature_dir = args.feature_dir.split(',')[0] if ',' in args.feature_dir else args.feature_dir
    mel_mean, mel_std = load_mel_channel_stats(first_feature_dir)
    pitch_mean, pitch_std = load_pitch_stats(first_feature_dir)

    # 转换为 torch 张量并移至设备
    mel_mean = torch.tensor(mel_mean, device=device).view(1, 1, -1)
    mel_std = torch.tensor(mel_std, device=device).view(1, 1, -1)
    pitch_mean = torch.tensor(pitch_mean, device=device)
    pitch_std = torch.tensor(pitch_std, device=device)

    # 确定关系数
    num_relations = get_num_relations(dataset)
    print(f"数据集中包含 {num_relations} 种依存关系（含自环）")

    # 词表或BERT
    if args.use_bert:
        vocab_size = None  # 将由BERT tokenizer处理
        print("使用BERT作为文本编码器，忽略本地词表")
    else:
        vocab_path = os.path.join(first_feature_dir, 'vocab.pt')
        if os.path.exists(vocab_path):
            vocab = torch.load(vocab_path, weights_only=False)
            vocab_size = len(vocab)
            print(f"加载词表，大小: {vocab_size}")
        else:
            vocab_size = 5000
            print(f"未找到词表文件，使用默认词表大小: {vocab_size}")

    # 说话人数量
    if args.has_speaker:
        speaker_ids = []
        for i in range(min(len(dataset), 1000)):
            item = dataset[i]
            speaker_ids.append(item['speaker_id'])
        num_speakers = len(set(speaker_ids)) if speaker_ids else 1
    else:
        num_speakers = 1

    # 初始化模型
    print("初始化 GCT-TTS 模型（改进版）...")
    model = GCTTTS(
        vocab_size=vocab_size,
        num_speakers=num_speakers,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_relations=num_relations,
        num_decoder_layers=args.num_decoder_layers,
        use_flow_duration=args.use_flow_duration,
        flow_n_flows=args.flow_n_flows,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        fusion=args.fusion,
        use_bert=args.use_bert,
        bert_model_name=args.bert_model,
        use_sparse_gcn=args.use_sparse_gcn,
        use_adaln=args.use_adaln,
        use_style_emb=args.use_style_emb,
        style_dim=args.style_dim,
        use_vae=args.use_vae,
        use_implicit_align=args.use_implicit_align,
        checkpointing=args.checkpointing
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 调度器选择
    if args.use_noam:
        scheduler = noam_scheduler(optimizer, warmup_steps=4000, hidden_dim=args.hidden_dim)
        scheduler_type = 'noam'
    else:
        # 使用 ReduceLROnPlateau，需要监控验证损失（注意：新版PyTorch已移除verbose参数）
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience)
        scheduler_type = 'plateau'

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and torch.cuda.is_available() else None

    # 可选损失
    if args.use_multi_scale_stft:
        criterion_stft = MultiScaleSTFTLoss().to(device)
    if args.use_gan:
        discriminator = MelDiscriminator(n_mel=128).to(device)
        optim_D = AdamW(discriminator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 自适应损失权重（不确定性加权）
    if args.use_uncertainty:
        log_sigma_mel = nn.Parameter(torch.zeros(1, device=device))
        log_sigma_dur = nn.Parameter(torch.zeros(1, device=device))
        log_sigma_pitch = nn.Parameter(torch.zeros(1, device=device))
        log_sigma_stft = nn.Parameter(torch.zeros(1, device=device)) if args.use_multi_scale_stft else None
        uncertainty_params = [log_sigma_mel, log_sigma_dur, log_sigma_pitch]
        if log_sigma_stft is not None:
            uncertainty_params.append(log_sigma_stft)
        optimizer.add_param_group({'params': uncertainty_params})

    # ---------- 频带加权损失配置 ----------
    if args.freq_band_weights:
        band_weights = [float(w) for w in args.freq_band_weights.split(',')]
    else:
        band_weights = [1.0, 1.0, 1.2, 1.5]  # 默认权重，高频更重
    n_bands = len(band_weights)
    n_mels = 128  # 固定
    assert n_mels % n_bands == 0, f"n_mels ({n_mels}) 必须能被频带数 ({n_bands}) 整除"
    band_size = n_mels // n_bands
    band_weights = torch.tensor(band_weights, device=device)
    print(f"使用频带加权损失: 频带数={n_bands}, 权重={band_weights.cpu().tolist()}, 每频带通道数={band_size}")
    # ---------------------------------------

    # 训练循环
    print("开始训练...")
    global_step = 0
    best_val_loss = float('inf')
    early_stop_counter = 0
    saved_checkpoints = []  # 用于管理最多保留的checkpoint文件

    for epoch in range(1, args.epochs + 1):
        # 动态调整时长损失权重
        if epoch < 20:
            current_dur_weight = 0.5
        elif epoch < 50:
            current_dur_weight = 0.2
        else:
            current_dur_weight = 0.1

        model.train()
        total_loss = 0.0
        total_mel = 0.0
        total_dur = 0.0
        total_pitch = 0.0
        total_time = 0.0
        total_range = 0.0
        total_stft = 0.0

        for batch_idx, batch in enumerate(dataloader):
            text_seq = batch["text_seq"].to(device)
            adj_matrix = batch["adj_matrix"].to(device)
            edge_type = batch["edge_type"].to(device)
            mel_target = batch["mel_target"].to(device)
            dur_target = batch["duration_target"].to(device)
            pitch_target = batch["pitch_target"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            mel_mask = batch["mel_padding_mask"].to(device)
            speaker_ids = batch.get("speaker_ids", None)
            if speaker_ids is not None:
                speaker_ids = speaker_ids.to(device)

            # ---------- 归一化（按频带） ----------
            mel_target_norm = (mel_target - mel_mean) / (mel_std + 1e-8)
            pitch_target_norm = (pitch_target - pitch_mean) / pitch_std
            # ---------------------------------------

            # 前向
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        text_seq, adj_matrix, edge_type,
                        speaker_ids=speaker_ids,
                        target_durations=dur_target,
                        src_padding_mask=src_mask
                    )
                    mel_pred = outputs['mel']
                    log_dur_pred = outputs['log_dur']
                    pitch_pred = outputs['pitch']

                    # 对齐长度
                    max_len = mel_target_norm.size(1)
                    if mel_pred.size(1) > max_len:
                        mel_pred = mel_pred[:, :max_len, :]
                    elif mel_pred.size(1) < max_len:
                        pad_size = max_len - mel_pred.size(1)
                        mel_pred = F.pad(mel_pred, (0, 0, 0, pad_size))

                    # ---------- 频带加权 Mel 损失 ----------
                    B, T_mel, D = mel_pred.shape
                    mel_pred_band = mel_pred.view(B, T_mel, n_bands, band_size)
                    mel_target_band = mel_target_norm.view(B, T_mel, n_bands, band_size)
                    band_abs = torch.abs(mel_pred_band - mel_target_band)  # [B,T,n_bands,band_size]
                    band_abs_mean = band_abs.mean(dim=-1)  # [B,T,n_bands]
                    mel_valid = ~mel_mask  # [B,T]
                    mel_valid_expanded = mel_valid.unsqueeze(-1).float()  # [B,T,1]
                    band_abs_mean = band_abs_mean * mel_valid_expanded
                    N = mel_valid.sum(dim=1)  # [B]
                    band_sum = band_abs_mean.sum(dim=1)  # [B,n_bands]
                    band_loss = band_sum / (N.unsqueeze(-1) + 1e-8)  # [B,n_bands]
                    mel_loss = (band_loss * band_weights.unsqueeze(0)).sum(dim=1).mean()  # 标量
                    # ---------------------------------------

                    src_valid = ~src_mask
                    dur_loss = F.mse_loss(log_dur_pred[src_valid], torch.log(dur_target.float() + 1)[src_valid])
                    pitch_loss = F.mse_loss(pitch_pred[src_valid], pitch_target_norm[src_valid])

                    # 时序损失
                    mel_pred_diff = mel_pred[:, 1:, :] - mel_pred[:, :-1, :]
                    mel_target_diff = mel_target_norm[:, 1:, :] - mel_target_norm[:, :-1, :]
                    diff_mask = ~mel_mask[:, 1:]
                    diff_valid = diff_mask.unsqueeze(-1).expand_as(mel_pred_diff)
                    time_loss = F.l1_loss(mel_pred_diff[diff_valid], mel_target_diff[diff_valid])

                    # 范围损失（可选）
                    mel_pred_valid = mel_pred[mel_valid]
                    mel_target_valid = mel_target_norm[mel_valid]
                    if mel_pred_valid.numel() > 0:
                        pred_min = mel_pred_valid.min()
                        pred_max = mel_pred_valid.max()
                        target_min = mel_target_valid.min()
                        target_max = mel_target_valid.max()
                        range_loss = F.smooth_l1_loss(pred_max, target_max) + F.smooth_l1_loss(pred_min, target_min)
                    else:
                        range_loss = torch.tensor(0.0, device=device)

                    # STFT损失（需波形，此处假设未启用）
                    stft_loss = 0.0
                    if args.use_multi_scale_stft:
                        # 需要逆归一化并转为波形，简化起见暂不实现
                        pass

                    # 不确定性加权
                    if args.use_uncertainty:
                        loss = (torch.exp(-log_sigma_mel) * mel_loss + log_sigma_mel +
                                torch.exp(-log_sigma_dur) * dur_loss + log_sigma_dur +
                                torch.exp(-log_sigma_pitch) * pitch_loss + log_sigma_pitch)
                        if args.use_multi_scale_stft:
                            loss += torch.exp(-log_sigma_stft) * stft_loss + log_sigma_stft
                    else:
                        loss = (mel_loss + current_dur_weight * dur_loss + pitch_loss +
                                args.lambda_time * time_loss + args.lambda_range * range_loss)
                        if args.use_multi_scale_stft:
                            loss += args.lambda_stft * stft_loss

                    # VAE KL损失
                    if 'kl_loss' in outputs:
                        loss += args.kl_weight * outputs['kl_loss']

                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if args.use_noam:
                        scheduler.step()
            else:
                outputs = model(
                    text_seq, adj_matrix, edge_type,
                    speaker_ids=speaker_ids,
                    target_durations=dur_target,
                    src_padding_mask=src_mask
                )
                mel_pred = outputs['mel']
                log_dur_pred = outputs['log_dur']
                pitch_pred = outputs['pitch']

                max_len = mel_target_norm.size(1)
                if mel_pred.size(1) > max_len:
                    mel_pred = mel_pred[:, :max_len, :]
                elif mel_pred.size(1) < max_len:
                    pad_size = max_len - mel_pred.size(1)
                    mel_pred = F.pad(mel_pred, (0, 0, 0, pad_size))

                # 频带加权 Mel 损失
                B, T_mel, D = mel_pred.shape
                mel_pred_band = mel_pred.view(B, T_mel, n_bands, band_size)
                mel_target_band = mel_target_norm.view(B, T_mel, n_bands, band_size)
                band_abs = torch.abs(mel_pred_band - mel_target_band)
                band_abs_mean = band_abs.mean(dim=-1)
                mel_valid = ~mel_mask
                mel_valid_expanded = mel_valid.unsqueeze(-1).float()
                band_abs_mean = band_abs_mean * mel_valid_expanded
                N = mel_valid.sum(dim=1)
                band_sum = band_abs_mean.sum(dim=1)
                band_loss = band_sum / (N.unsqueeze(-1) + 1e-8)
                mel_loss = (band_loss * band_weights.unsqueeze(0)).sum(dim=1).mean()

                src_valid = ~src_mask
                dur_loss = F.mse_loss(log_dur_pred[src_valid], torch.log(dur_target.float() + 1)[src_valid])
                pitch_loss = F.mse_loss(pitch_pred[src_valid], pitch_target_norm[src_valid])

                mel_pred_diff = mel_pred[:, 1:, :] - mel_pred[:, :-1, :]
                mel_target_diff = mel_target_norm[:, 1:, :] - mel_target_norm[:, :-1, :]
                diff_mask = ~mel_mask[:, 1:]
                diff_valid = diff_mask.unsqueeze(-1).expand_as(mel_pred_diff)
                time_loss = F.l1_loss(mel_pred_diff[diff_valid], mel_target_diff[diff_valid])

                mel_pred_valid = mel_pred[mel_valid]
                mel_target_valid = mel_target_norm[mel_valid]
                if mel_pred_valid.numel() > 0:
                    pred_min = mel_pred_valid.min()
                    pred_max = mel_pred_valid.max()
                    target_min = mel_target_valid.min()
                    target_max = mel_target_valid.max()
                    range_loss = F.smooth_l1_loss(pred_max, target_max) + F.smooth_l1_loss(pred_min, target_min)
                else:
                    range_loss = torch.tensor(0.0, device=device)

                stft_loss = 0.0
                if args.use_multi_scale_stft:
                    stft_loss = criterion_stft(mel_pred, mel_target_norm)

                if args.use_uncertainty:
                    loss = (torch.exp(-log_sigma_mel) * mel_loss + log_sigma_mel +
                            torch.exp(-log_sigma_dur) * dur_loss + log_sigma_dur +
                            torch.exp(-log_sigma_pitch) * pitch_loss + log_sigma_pitch)
                    if args.use_multi_scale_stft:
                        loss += torch.exp(-log_sigma_stft) * stft_loss + log_sigma_stft
                else:
                    loss = (mel_loss + current_dur_weight * dur_loss + pitch_loss +
                            args.lambda_time * time_loss + args.lambda_range * range_loss)
                    if args.use_multi_scale_stft:
                        loss += args.lambda_stft * stft_loss

                if 'kl_loss' in outputs:
                    loss += args.kl_weight * outputs['kl_loss']

                loss.backward()
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    if args.use_noam:
                        scheduler.step()

            total_loss += loss.item()
            total_mel += mel_loss.item()
            total_dur += dur_loss.item()
            total_pitch += pitch_loss.item()
            total_time += time_loss.item()
            total_range += range_loss.item()
            total_stft += stft_loss.item() if isinstance(stft_loss, torch.Tensor) else stft_loss
            global_step += 1

            if batch_idx % args.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} (Mel: {mel_loss.item():.4f}, Dur: {dur_loss.item():.4f}, "
                      f"Pitch: {pitch_loss.item():.4f}, Time: {time_loss.item():.4f}, Range: {range_loss.item():.4f}, "
                      f"STFT: {stft_loss:.4f}) | LR: {lr:.2e}")

        avg_loss = total_loss / len(dataloader)
        avg_mel = total_mel / len(dataloader)
        avg_dur = total_dur / len(dataloader)
        avg_pitch = total_pitch / len(dataloader)
        avg_time = total_time / len(dataloader)
        avg_range = total_range / len(dataloader)
        avg_stft = total_stft / len(dataloader)
        print(f"==== Epoch {epoch} 结束 | 平均 Loss: {avg_loss:.4f} (Mel: {avg_mel:.4f}, Dur: {avg_dur:.4f}, "
              f"Pitch: {avg_pitch:.4f}, Time: {avg_time:.4f}, Range: {avg_range:.4f}, STFT: {avg_stft:.4f}) ====")

        # 验证
        val_loss = avg_loss  # 用训练损失作为验证损失（简化，实际应使用验证集）
        if epoch % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                test_text = "今天天气真好。"
                frontend = TextGraphFrontend(language="en_core_web_sm", use_bert=args.use_bert,
                                             bert_model=args.bert_model)
                seq_ids, adj, etype = frontend.text_to_sequence_and_graph(test_text)
                seq_ids = seq_ids.unsqueeze(0).to(device)
                adj = adj.unsqueeze(0).to(device)
                etype = etype.unsqueeze(0).to(device)
                src_mask = torch.zeros_like(seq_ids, dtype=torch.bool, device=device)
                outputs = model(
                    seq_ids, adj, etype,
                    speaker_ids=None,
                    target_durations=None,
                    src_padding_mask=src_mask
                )
                mel_pred = outputs['mel']
                log_dur_pred = outputs['log_dur']
                durations_float = torch.exp(log_dur_pred) - 1
                durations_int = torch.clamp(torch.round(durations_float), min=1).long()
                # 逆归一化
                mel_denorm = mel_pred * mel_std + mel_mean
                print(
                    f"验证 '{test_text}': 时长 {durations_int.cpu().tolist()}, Mel范围 [{mel_denorm.min():.2f}, {mel_denorm.max():.2f}]")
                if epoch % (args.val_interval * 2) == 0:
                    try:
                        from scipy.io import wavfile
                        import librosa
                        mel_np = mel_denorm.squeeze(0).cpu().numpy()
                        audio = librosa.feature.inverse.mel_to_audio(mel_np.T, sr=22050, n_iter=32)
                        wavfile.write(os.path.join(args.save_dir, f"val_epoch_{epoch}.wav"), 22050,
                                      (audio * 32767).astype(np.int16))
                        print(f"验证音频已保存")
                    except Exception as e:
                        print(f"保存验证音频失败: {e}")
            model.train()

        # 学习率调度（对于 ReduceLROnPlateau）
        if not args.use_noam:
            scheduler.step(val_loss)

        # 早停检查
        if args.early_stop_patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop_patience:
                    print(f"早停触发：验证损失连续 {args.early_stop_patience} 个epoch未改善，停止训练。")
                    break

        # 保存 checkpoint
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"gct_tts_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if args.use_noam else None,
                'loss': avg_loss,
                'config': {
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_relations': num_relations,
                    'num_decoder_layers': args.num_decoder_layers,
                    'num_heads': args.num_heads,
                    'ffn_dim': args.ffn_dim,
                    'fusion': args.fusion,
                    'use_bert': args.use_bert,
                },
                'mel_stats': {'mean': mel_mean.cpu(), 'std': mel_std.cpu()},
                'pitch_stats': {'mean': pitch_mean.cpu().item(), 'std': pitch_std.cpu().item()}
            }, save_path)
            print(f"模型已保存至: {save_path}")

            # 管理最多保留的checkpoint数量
            saved_checkpoints.append(save_path)
            if args.max_ckpt_keep > 0 and len(saved_checkpoints) > args.max_ckpt_keep:
                oldest_ckpt = saved_checkpoints.pop(0)
                if os.path.exists(oldest_ckpt):
                    os.remove(oldest_ckpt)
                    print(f"删除旧checkpoint: {oldest_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCT-TTS 训练脚本（增强版）")
    parser.add_argument("--metadata_path", type=str, required=True, help="预处理后的元数据文件路径")
    parser.add_argument("--feature_dir", type=str, required=True, help="特征目录（若多数据集用逗号分隔）")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--multi_dataset", action="store_true", help="启用多数据集")
    parser.add_argument("--use_hdf5", action="store_true", help="使用 HDF5 打包特征")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=str, default="auto", help="批大小，可指定数字或'auto'")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--fp16", action="store_true", help="启用混合精度训练")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--has_speaker", action="store_true")
    parser.add_argument("--use_flow_duration", action="store_true")
    parser.add_argument("--flow_n_flows", type=int, default=4)
    parser.add_argument("--use_noam", action="store_true")
    parser.add_argument("--lambda_time", type=float, default=0.3)
    parser.add_argument("--lambda_range", type=float, default=0.3)

    # 模型架构参数
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--fusion", type=str, default="cross", choices=["cross", "gate"])

    # 新特性参数
    parser.add_argument("--use_bert", action="store_true", help="使用BERT作为文本编码器")
    parser.add_argument("--bert_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--use_sparse_gcn", action="store_true", help="使用稀疏图卷积")
    parser.add_argument("--use_adaln", action="store_true", help="使用AdaIN进行说话人自适应")
    parser.add_argument("--use_style_emb", action="store_true", help="使用风格嵌入")
    parser.add_argument("--style_dim", type=int, default=128)
    parser.add_argument("--use_vae", action="store_true", help="使用VAE风格建模")
    parser.add_argument("--use_implicit_align", action="store_true", help="使用隐式对齐替代显式时长")
    parser.add_argument("--checkpointing", action="store_true", help="启用激活检查点以节省显存")
    parser.add_argument("--use_multi_scale_stft", action="store_true", help="使用多尺度STFT损失")
    parser.add_argument("--lambda_stft", type=float, default=0.1)
    parser.add_argument("--use_gan", action="store_true", help="启用对抗训练（实验性）")
    parser.add_argument("--use_uncertainty", action="store_true", help="使用不确定性加权损失")
    parser.add_argument("--kl_weight", type=float, default=0.01, help="VAE KL损失权重")

    # 新增：频带加权损失参数
    parser.add_argument("--freq_band_weights", type=str, default="1.0,1.0,1.2,1.5",
                        help="逗号分隔的频带权重，例如 1.0,1.0,1.2,1.5，数量决定频带数")

    # 新增训练控制参数
    parser.add_argument("--lr_factor", type=float, default=0.5, help="学习率衰减因子（用于ReduceLROnPlateau）")
    parser.add_argument("--lr_patience", type=int, default=5, help="学习率衰减的耐心轮数")
    parser.add_argument("--max_ckpt_keep", type=int, default=5, help="最多保留的checkpoint数量（0表示保留全部）")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="早停耐心轮数（0表示禁用早停）")

    args = parser.parse_args()
    train(args)