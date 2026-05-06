import os
# ========== 强制离线模式 ==========
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# =================================

import argparse
import torch
import torchaudio
import librosa
import numpy as np
import tgt
import pyworld as pw
from tqdm import tqdm
import multiprocessing as mp
import time
import traceback
import h5py
from pathlib import Path

# 注意：不再在这里导入 gct_tts.frontend，而是移到生产者内部

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ---------- 全局变量（生产者进程初始化） ----------
_producer_frontend = None
_producer_sr = 22050
_producer_hop_length = 256
_producer_win_length = 1024
_producer_n_mels = 128
_producer_use_hdf5 = False
_producer_bert_model = "bert-base-multilingual-cased"

def init_producer(sr, hop_length, win_length, n_mels, use_hdf5, use_bert, bert_model):
    """生产者进程初始化：加载前端"""
    # 再次确保子进程使用离线模式
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 在这里导入前端，避免消费者进程导入
    from gct_tts.frontend import TextGraphFrontend

    global _producer_frontend, _producer_sr, _producer_hop_length, _producer_win_length, _producer_n_mels, _producer_use_hdf5, _producer_bert_model
    _producer_sr = sr
    _producer_hop_length = hop_length
    _producer_win_length = win_length
    _producer_n_mels = n_mels
    _producer_use_hdf5 = use_hdf5
    _producer_bert_model = bert_model

    # 如果使用 BERT，检查路径有效性（避免静默回退）
    if use_bert:
        # 打印当前工作目录
        print(f"[进程 {os.getpid()}] 当前工作目录: {os.getcwd()}")
        # 将相对路径转换为绝对路径（如果存在）
        if not os.path.isabs(bert_model) and os.path.exists(bert_model):
            bert_model = os.path.abspath(bert_model)
            print(f"[进程 {os.getpid()}] 转换为绝对路径: {bert_model}")
        # 即使不存在，也保留原值，交给 frontend 处理
        print(f"[进程 {os.getpid()}] 最终传给 frontend 的 bert_model: {bert_model}")

    print(f"[{os.getpid()}] 生产者进程加载前端（use_bert={use_bert}, bert_model={bert_model}）...")
    _producer_frontend = TextGraphFrontend(language="en_core_web_sm", use_bert=use_bert, bert_model=bert_model)

# ---------- 备用音高提取（自相关法） ----------
def compute_f0_autocorr(waveform, sr, hop_length, f0_min=50, f0_max=500):
    frames = librosa.util.frame(waveform, frame_length=hop_length*2, hop_length=hop_length)
    f0 = np.zeros(frames.shape[1])
    for i, frame in enumerate(frames.T):
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        if len(corr) < 2:
            continue
        diff = np.diff(corr)
        peaks = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
        if len(peaks) == 0:
            continue
        min_lag = int(sr / f0_max)
        max_lag = int(sr / f0_min)
        valid_peaks = peaks[(peaks >= min_lag) & (peaks <= max_lag)]
        if len(valid_peaks) == 0:
            continue
        best_lag = valid_peaks[np.argmax(corr[valid_peaks])]
        f0[i] = sr / best_lag
    return f0

# ---------- 音高和时长提取（pyworld） ----------
def get_pitch_and_durations_from_waveform(waveform_np, text_len, mel_len, tg_path=None):
    global _producer_sr, _producer_hop_length
    try:
        # 如果 mel_len < text_len，无法对齐，采用均匀时长
        if mel_len < text_len:
            print(f"  mel_len ({mel_len}) < text_len ({text_len})，使用均匀时长")
            durations = torch.full((text_len,), max(1, mel_len // text_len), dtype=torch.long)
            remainder = mel_len - durations.sum()
            if remainder > 0:
                durations[-1] += remainder
            # 音高全零
            word_pitches = torch.zeros(text_len, dtype=torch.float32)
            return durations, word_pitches

        waveform_f64 = waveform_np.astype(np.float64)
        frame_period = _producer_hop_length / _producer_sr * 1000.0
        f0, t = pw.dio(waveform_f64, _producer_sr, frame_period=frame_period)
        f0 = pw.stonemask(waveform_f64, f0, t, _producer_sr)
        f0 = torch.tensor(f0, dtype=torch.float32)

        if len(f0) > mel_len:
            f0 = f0[:mel_len]
        elif len(f0) < mel_len:
            f0 = torch.nn.functional.pad(f0, (0, mel_len - len(f0)))

        durations = torch.zeros(text_len, dtype=torch.long)

        if tg_path and os.path.exists(tg_path):
            textgrid = tgt.io.read_textgrid(tg_path)
            tier = textgrid.get_tier_by_name("words")
            if len(tier.intervals) != text_len:
                print(f"  TextGrid 单词数 ({len(tier.intervals)}) 与文本长度 ({text_len}) 不符，跳过使用 TextGrid")
                base_frames = mel_len // text_len
                remainder = mel_len % text_len
                durations = torch.full((text_len,), base_frames, dtype=torch.long)
                if remainder > 0:
                    durations[:remainder] += 1
            else:
                for i, interval in enumerate(tier.intervals):
                    start_frame = int(interval.start_time * _producer_sr / _producer_hop_length)
                    end_frame = int(interval.end_time * _producer_sr / _producer_hop_length)
                    durations[i] = end_frame - start_frame
        else:
            base_frames = mel_len // text_len
            remainder = mel_len % text_len
            durations = torch.full((text_len,), base_frames, dtype=torch.long)
            if remainder > 0:
                durations[:remainder] += 1

        dur_sum = durations.sum().item()
        if dur_sum != mel_len:
            diff = mel_len - dur_sum
            durations[-1] += diff

        word_pitches = torch.zeros(text_len, dtype=torch.float32)
        current_frame = 0
        for i in range(text_len):
            dur = durations[i].item()
            if dur > 0:
                word_f0 = f0[current_frame: current_frame + dur]
                valid_f0 = word_f0[word_f0 > 0]
                if len(valid_f0) > 0:
                    word_pitches[i] = valid_f0.mean().float()
            current_frame += dur

        return durations, word_pitches

    except Exception as e:
        print(f"  pyworld 提取音高/时长时出错: {e}")
        # 出错时返回均匀时长和零音高
        if mel_len < text_len:
            durations = torch.full((text_len,), max(1, mel_len // text_len), dtype=torch.long)
            remainder = mel_len - durations.sum()
            if remainder > 0:
                durations[-1] += remainder
        else:
            base_frames = mel_len // text_len
            remainder = mel_len % text_len
            durations = torch.full((text_len,), base_frames, dtype=torch.long)
            if remainder > 0:
                durations[:remainder] += 1
        word_pitches = torch.zeros(text_len, dtype=torch.float32)
        return durations, word_pitches

# ---------- 音高和时长提取（fallback） ----------
def get_pitch_and_durations_fallback(waveform_np, text_len, mel_len, tg_path=None):
    global _producer_sr, _producer_hop_length
    try:
        if mel_len < text_len:
            durations = torch.full((text_len,), max(1, mel_len // text_len), dtype=torch.long)
            remainder = mel_len - durations.sum()
            if remainder > 0:
                durations[-1] += remainder
            word_pitches = torch.zeros(text_len, dtype=torch.float32)
            return durations, word_pitches

        f0 = compute_f0_autocorr(waveform_np, _producer_sr, _producer_hop_length)
        f0 = torch.tensor(f0, dtype=torch.float32)
        if len(f0) > mel_len:
            f0 = f0[:mel_len]
        elif len(f0) < mel_len:
            f0 = torch.nn.functional.pad(f0, (0, mel_len - len(f0)))

        durations = torch.zeros(text_len, dtype=torch.long)

        if tg_path and os.path.exists(tg_path):
            textgrid = tgt.io.read_textgrid(tg_path)
            tier = textgrid.get_tier_by_name("words")
            if len(tier.intervals) != text_len:
                print(f"  TextGrid 单词数 ({len(tier.intervals)}) 与文本长度 ({text_len}) 不符，跳过使用 TextGrid")
                base_frames = mel_len // text_len
                remainder = mel_len % text_len
                durations = torch.full((text_len,), base_frames, dtype=torch.long)
                if remainder > 0:
                    durations[:remainder] += 1
            else:
                for i, interval in enumerate(tier.intervals):
                    start_frame = int(interval.start_time * _producer_sr / _producer_hop_length)
                    end_frame = int(interval.end_time * _producer_sr / _producer_hop_length)
                    durations[i] = end_frame - start_frame
        else:
            base_frames = mel_len // text_len
            remainder = mel_len % text_len
            durations = torch.full((text_len,), base_frames, dtype=torch.long)
            if remainder > 0:
                durations[:remainder] += 1

        dur_sum = durations.sum().item()
        if dur_sum != mel_len:
            diff = mel_len - dur_sum
            durations[-1] += diff

        word_pitches = torch.zeros(text_len, dtype=torch.float32)
        current_frame = 0
        for i in range(text_len):
            dur = durations[i].item()
            if dur > 0:
                word_f0 = f0[current_frame: current_frame + dur]
                valid_f0 = word_f0[word_f0 > 0]
                if len(valid_f0) > 0:
                    word_pitches[i] = valid_f0.mean().float()
            current_frame += dur

        return durations, word_pitches

    except Exception as e:
        print(f"  fallback 提取音高/时长时出错: {e}")
        # 出错时返回均匀时长和零音高
        if mel_len < text_len:
            durations = torch.full((text_len,), max(1, mel_len // text_len), dtype=torch.long)
            remainder = mel_len - durations.sum()
            if remainder > 0:
                durations[-1] += remainder
        else:
            base_frames = mel_len // text_len
            remainder = mel_len % text_len
            durations = torch.full((text_len,), base_frames, dtype=torch.long)
            if remainder > 0:
                durations[:remainder] += 1
        word_pitches = torch.zeros(text_len, dtype=torch.float32)
        return durations, word_pitches

# ---------- 生产者函数 ----------
def producer_process(task_args, result_queue, sr, hop_length, win_length, n_mels, use_hdf5, use_bert, bert_model):
    init_producer(sr, hop_length, win_length, n_mels, use_hdf5, use_bert, bert_model)

    global _producer_frontend, _producer_sr, _producer_hop_length

    for args in task_args:
        wav_id, text, wav_dir, tg_dir, num_augment, out_dir = args
        try:
            wav_path = os.path.join(wav_dir, f"{wav_id}.wav")
            tg_path = os.path.join(tg_dir, f"{wav_id}.TextGrid") if tg_dir else None

            if not os.path.exists(wav_path):
                result_queue.put(('error', (wav_id, f"音频文件不存在 {wav_path}")))
                continue

            waveform_orig, sr_orig = librosa.load(wav_path, sr=_producer_sr, mono=True)

            max_samples = _producer_sr * 20
            if len(waveform_orig) > max_samples:
                result_queue.put(('error', (wav_id, f"音频过长 ({len(waveform_orig)/_producer_sr:.1f}s)，跳过")))
                continue

            seq_ids, adj_matrix, edge_type = _producer_frontend.text_to_sequence_and_graph(text)
            text_len = seq_ids.size(0)

            aug_configs = [('orig', 1.0, 1.0)]
            for i in range(num_augment):
                speed = np.random.uniform(0.85, 1.15)
                vol = np.random.uniform(0.8, 1.2)
                aug_configs.append((f'aug{i}', speed, vol))

            for aug_name, speed, vol in aug_configs:
                try:
                    waveform = waveform_orig.copy()
                    if speed != 1.0:
                        waveform = librosa.effects.time_stretch(waveform, rate=speed)
                    if vol != 1.0:
                        waveform = waveform * vol
                    noise = np.random.normal(0, np.std(waveform) * 0.005, waveform.shape)
                    waveform = waveform + noise

                    mel_len_est = int(len(waveform) / _producer_hop_length) + 1

                    tg_path_to_use = tg_path if aug_name == 'orig' else None

                    try:
                        durations, word_pitches = get_pitch_and_durations_from_waveform(
                            waveform, text_len, mel_len_est, tg_path_to_use
                        )
                    except Exception as e:
                        print(f"[{os.getpid()}] pyworld 失败，切换到自实现音高提取: {e}")
                        durations, word_pitches = get_pitch_and_durations_fallback(
                            waveform, text_len, mel_len_est, tg_path_to_use
                        )

                    # 确保 durations 和 word_pitches 不为 None（函数已保证）
                    new_id = f"{wav_id}_{aug_name}" if aug_name != 'orig' else wav_id

                    sample = {
                        'new_id': new_id,
                        'text': text,
                        'seq': seq_ids.numpy(),
                        'adj': adj_matrix.numpy(),
                        'edge_type': edge_type.numpy(),
                        'dur': durations.numpy(),
                        'pitch': word_pitches.numpy(),
                        'waveform': waveform.astype(np.float32),
                    }

                    result_queue.put(('data', sample))

                except Exception as e:
                    error_msg = f"[{os.getpid()}] 增强样本 {wav_id}_{aug_name} 处理失败: {str(e)}\n{traceback.format_exc()}"
                    result_queue.put(('error', (wav_id, error_msg)))

        except Exception as e:
            error_msg = f"[{os.getpid()}] 生产者处理 {wav_id} 时出错: {str(e)}\n{traceback.format_exc()}"
            result_queue.put(('error', (wav_id, error_msg)))

# ---------- 消费者函数（GPU批处理） ----------
def consumer_process(result_queue, out_dir, use_hdf5, batch_size, device, sr, hop_length, win_length, n_mels):
    # 消费者进程也设置离线模式，防止意外联网
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    print(f"[{os.getpid()}] 消费者进程启动，设备: {device}，初始批大小: {batch_size}")

    # 确保 batch_size 为整数
    if isinstance(batch_size, str):
        if batch_size.lower() == 'auto':
            # 如果还是 'auto'，则估算
            batch_size = estimate_batch_size(default_batch_size=32)
            print(f"[消费者] 将 'auto' 解析为实际批大小: {batch_size}")
        else:
            batch_size = int(batch_size)
            print(f"[消费者] 将字符串批大小转换为整数: {batch_size}")

    if device == 'cuda':
        print("提示：如遇 CUDA 内存碎片，可尝试设置环境变量：export PYTORCH_ALLOC_CONF=expandable_segments:True")

    try:
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"  CUDA 可用，设备名称: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif device == 'cuda':
            device = 'cpu'
            print("  CUDA 不可用，降级到 CPU")
        else:
            print("  使用 CPU")

        device = torch.device(device)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            f_min=0.0,
            f_max=8000.0,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode='reflect',
            norm='slaney',
            mel_scale='htk'
        ).to(device)

        if use_hdf5:
            h5_path = os.path.join(out_dir, 'features.h5')
            h5f = h5py.File(h5_path, 'a')
        else:
            h5f = None

        pbar = tqdm(desc="消费者处理", unit="样本", position=0, leave=True)
        valid_records = []
        all_pitches = []
        all_mel_stats = []
        error_log_path = os.path.join(out_dir, "errors.log")

        # 初始化每个频带的统计量（用于计算全局均值和标准差）
        channel_counts = np.zeros(n_mels, dtype=np.int64)
        channel_means = np.zeros(n_mels, dtype=np.float64)
        channel_m2s = np.zeros(n_mels, dtype=np.float64)

        buffer = []
        producer_count = 0

        # 递归批处理函数（处理OOM）
        def process_batch(samples):
            if len(samples) == 0:
                return
            try:
                waveforms = [torch.from_numpy(s['waveform']).float() for s in samples]
                lengths = [w.size(0) for w in waveforms]
                padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).to(device)

                with torch.no_grad():
                    mels = mel_transform(padded_waveforms)  # [B, n_mels, T']
                    mels = torch.log(torch.clamp(mels, min=1e-5))
                    mels = mels.transpose(1, 2)  # [B, T', n_mels]

                mel_lengths = [max(1, l // hop_length + 1) for l in lengths]

                for i, sample in enumerate(samples):
                    mel = mels[i]
                    actual_len = min(mel.size(0), mel_lengths[i])
                    mel_np = mel[:actual_len].cpu().numpy()

                    # 更新Mel频带统计量（Welford）
                    T = mel_np.shape[0]
                    if T > 0:
                        col_means = np.mean(mel_np, axis=0)  # [n_mels]
                        col_vars = np.var(mel_np, axis=0, ddof=1)  # 样本方差
                        col_vars = np.nan_to_num(col_vars, nan=0.0)
                        for c in range(n_mels):
                            nA = T
                            mA = col_means[c]
                            varA = col_vars[c]
                            M2A = varA * (nA - 1) if nA > 1 else 0.0
                            nB = channel_counts[c]
                            mB = channel_means[c]
                            M2B = channel_m2s[c]
                            n = nA + nB
                            delta = mA - mB
                            m = (nA * mA + nB * mB) / n
                            M2 = M2A + M2B + delta**2 * (nA * nB) / n
                            channel_counts[c] = n
                            channel_means[c] = m
                            channel_m2s[c] = M2

                    if use_hdf5:
                        group_name = sample['new_id']
                        if group_name in h5f:
                            del h5f[group_name]
                        grp = h5f.create_group(group_name)
                        grp.create_dataset('seq', data=sample['seq'])
                        grp.create_dataset('adj', data=sample['adj'])
                        grp.create_dataset('edge_type', data=sample['edge_type'])
                        grp.create_dataset('mel', data=mel_np)
                        grp.create_dataset('dur', data=sample['dur'])
                        grp.create_dataset('pitch', data=sample['pitch'])
                    else:
                        torch.save(torch.tensor(sample['seq']), os.path.join(out_dir, f"{sample['new_id']}_seq.pt"))
                        torch.save(torch.tensor(sample['adj']), os.path.join(out_dir, f"{sample['new_id']}_adj.pt"))
                        torch.save(torch.tensor(sample['edge_type']), os.path.join(out_dir, f"{sample['new_id']}_edge_type.pt"))
                        torch.save(torch.tensor(mel_np), os.path.join(out_dir, f"{sample['new_id']}_mel.pt"))
                        torch.save(torch.tensor(sample['dur']), os.path.join(out_dir, f"{sample['new_id']}_dur.pt"))
                        torch.save(torch.tensor(sample['pitch']), os.path.join(out_dir, f"{sample['new_id']}_pitch.pt"))

                    valid_records.append(f"{sample['new_id']}|{sample['text']}")

                    pitch_array = np.asarray(sample['pitch']).flatten()
                    if pitch_array.size > 0:
                        all_pitches.append(pitch_array)

                    all_mel_stats.append((mel_np.min(), mel_np.max(), mel_np.mean()))
                    pbar.update(1)

            except torch.OutOfMemoryError as e:
                if len(samples) > 1:
                    mid = len(samples) // 2
                    print(f"\nOOM detected, splitting batch of size {len(samples)} into two halves.")
                    process_batch(samples[:mid])
                    process_batch(samples[mid:])
                else:
                    print(f"\n单个样本 {samples[0]['new_id']} 处理时OOM，跳过该样本。")
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"样本 {samples[0]['new_id']} 因CUDA内存不足跳过\n")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"\n处理批次时发生未知错误: {e}")
                for sample in samples:
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"样本 {sample['new_id']} 处理时出错: {traceback.format_exc()}\n")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 主循环
        while True:
            try:
                msg = result_queue.get(timeout=5)
            except mp.queues.Empty:
                if producer_count > 0:
                    break
                continue

            if msg is None:
                producer_count += 1
                continue

            msg_type, *payload = msg
            if msg_type == 'error':
                wav_id, error_msg = payload[0]
                print(f"\n错误: {error_msg}")
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                continue
            elif msg_type == 'data':
                sample = payload[0]
                buffer.append(sample)
                if len(buffer) >= batch_size:
                    process_batch(buffer)
                    buffer = []

        # 处理剩余样本
        if buffer:
            process_batch(buffer)

        pbar.close()
        if use_hdf5:
            h5f.close()

        # 计算并保存Mel通道统计量
        channel_std = np.zeros_like(channel_means)
        for c in range(n_mels):
            if channel_counts[c] > 1:
                channel_std[c] = np.sqrt(channel_m2s[c] / (channel_counts[c] - 1))
            else:
                channel_std[c] = 0.0
        mel_channel_stats = {
            'mean': channel_means.astype(np.float32),
            'std': channel_std.astype(np.float32)
        }
        torch.save(mel_channel_stats, os.path.join(out_dir, 'mel_channel_stats.pt'))
        print(f"Mel通道统计量已保存至 {os.path.join(out_dir, 'mel_channel_stats.pt')}")

        # 合并音高
        if all_pitches:
            all_pitches_array = np.concatenate(all_pitches)
        else:
            all_pitches_array = np.array([])

        stats = {
            'valid_records': valid_records,
            'all_pitches': all_pitches_array,
            'mel_stats': all_mel_stats
        }
        torch.save(stats, os.path.join(out_dir, '.consumer_stats.pt'))
        print(f"[{os.getpid()}] 消费者进程正常退出")

    except Exception as e:
        error_msg = f"消费者进程崩溃: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open(os.path.join(out_dir, "errors.log"), 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")
        with open(os.path.join(out_dir, '.consumer_failed'), 'w') as f:
            f.write("1")
        raise

# ---------- 自动计算批大小（保守版） ----------
def estimate_batch_size(default_batch_size=32, safety_factor=0.4):
    if not torch.cuda.is_available():
        return default_batch_size

    try:
        device = torch.cuda.current_device()
        free_memory, _ = torch.cuda.mem_get_info(device)
        sample_memory = 5 * 1024 * 1024  # 每个样本估计5MB
        usable_memory = free_memory * safety_factor
        estimated_batch = int(usable_memory // sample_memory)
        estimated_batch = max(1, min(estimated_batch, 64))
        print(f"根据GPU显存自动调整批大小: {estimated_batch} (可用显存: {free_memory/1e9:.2f} GB)")
        return estimated_batch
    except Exception as e:
        print(f"自动批大小估算失败，使用默认值 {default_batch_size}: {e}")
        return default_batch_size

# ---------- 异步管道主函数 ----------
def run_async_mode(metadata_path, wav_dir, out_dir, tg_dir, num_augment, num_workers, use_hdf5, batch_size, use_bert, bert_model):
    # 打印 bert_model 以便调试
    print(f"[主进程] run_async_mode 接收到的 bert_model: {bert_model}")

    os.makedirs(out_dir, exist_ok=True)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    items = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2:
            continue
        wav_id, text = parts[0], parts[1]
        items.append((wav_id, text))

    total_items = len(items)
    print(f"异步管道：共 {total_items} 条数据，每条生成 {num_augment+1} 个版本。生产者进程数: {num_workers}，批大小: {batch_size}")

    # ---------- 解析批大小 ----------
    if batch_size is None or batch_size == 'auto':
        batch_size = estimate_batch_size(default_batch_size=32)
        print(f"[主进程] 使用自动估算的批大小: {batch_size}")
    else:
        batch_size = int(batch_size)
        print(f"[主进程] 使用指定的批大小: {batch_size}")
    # ---------------------------------

    chunk_size = max(1, total_items // num_workers)
    task_chunks = [items[i:i+chunk_size] for i in range(0, total_items, chunk_size)]

    result_queue = mp.Queue(maxsize=num_workers * 4)

    producers = []
    sr = 22050
    hop_length = 256
    win_length = 1024
    n_mels = 128
    for chunk in task_chunks:
        args_list = [(wav_id, text, wav_dir, tg_dir, num_augment, out_dir) for wav_id, text in chunk]
        p = mp.Process(target=producer_process, args=(args_list, result_queue, sr, hop_length, win_length, n_mels, use_hdf5, use_bert, bert_model))
        p.start()
        producers.append(p)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    consumer = mp.Process(target=consumer_process, args=(result_queue, out_dir, use_hdf5, batch_size, device, sr, hop_length, win_length, n_mels))
    consumer.start()

    for p in producers:
        p.join()

    for _ in range(len(producers)):
        result_queue.put(None)

    consumer.join(timeout=120)
    success = True
    if consumer.is_alive():
        print("警告：消费者进程未在超时内退出，强制终止")
        consumer.terminate()
        consumer.join()
        success = False
    else:
        if consumer.exitcode != 0:
            print(f"消费者进程异常退出，退出码: {consumer.exitcode}")
            success = False
        fail_flag = os.path.join(out_dir, '.consumer_failed')
        if os.path.exists(fail_flag):
            os.remove(fail_flag)
            success = False

    if not success:
        print("\n异步管道失败，请检查 errors.log。")
        return False

    # ---------- 读取消费者统计信息（简化加载） ----------
    stats_path = os.path.join(out_dir, '.consumer_stats.pt')
    valid_records = []
    all_pitches = []
    all_mel_stats = []

    if os.path.exists(stats_path):
        try:
            # 内部生成的文件，可直接使用 weights_only=False
            stats = torch.load(stats_path, weights_only=False)
            valid_records = stats.get('valid_records', [])
            all_pitches = stats.get('all_pitches', np.array([]))
            all_mel_stats = stats.get('mel_stats', [])
            os.remove(stats_path)
        except Exception as e:
            print(f"加载统计文件失败: {e}，将使用空统计信息继续。")

    # 保存最终元数据
    new_metadata_path = os.path.join(out_dir, "train_metadata.txt")
    with open(new_metadata_path, 'w', encoding='utf-8') as f:
        for record in valid_records:
            f.write(record + "\n")

    # 计算并保存统计量（原有min/max仅用于打印，不再保存）
    if all_mel_stats:
        mins = [s[0] for s in all_mel_stats]
        maxs = [s[1] for s in all_mel_stats]
        means = [s[2] for s in all_mel_stats]
        print(f"\n训练集 Mel 统计（仅参考）：")
        print(f"  最小值范围: [{min(mins):.2f}, {max(mins):.2f}]")
        print(f"  最大值范围: [{min(maxs):.2f}, {max(maxs):.2f}]")
        print(f"  均值范围: [{min(means):.2f}, {max(means):.2f}]")

    if len(all_pitches) > 0:
        if isinstance(all_pitches, list):
            all_pitches = np.array(all_pitches)
        pitch_mean = np.mean(all_pitches)
        pitch_std = np.std(all_pitches) + 1e-5
        torch.save({'mean': pitch_mean, 'std': pitch_std},
                   os.path.join(out_dir, 'pitch_stats.pt'))
        print(f"音高统计量已保存: mean={pitch_mean:.2f}, std={pitch_std:.2f}")

    # 生成词表（如果未使用BERT且不存在）
    if not use_bert:
        vocab_path = os.path.join(out_dir, 'vocab.pt')
        if not os.path.exists(vocab_path):
            print("未找到词表文件，正在生成...")
            # 在这里导入前端（仅用于生成词表）
            from gct_tts.frontend import TextGraphFrontend
            single_frontend = TextGraphFrontend(language="en_core_web_sm")
            for wav_id, text in items:
                single_frontend.text_to_sequence_and_graph(text)
            torch.save(single_frontend.vocab, vocab_path)
            print(f"词表已保存至 {vocab_path}，大小: {len(single_frontend.vocab)}")
        else:
            print(f"词表已存在: {vocab_path}")

    print(f"异步预处理完成！成功生成 {len(valid_records)} 条有效特征，存放于 {out_dir}")
    error_log_path = os.path.join(out_dir, "errors.log")
    if os.path.exists(error_log_path):
        print(f"错误日志已保存至 {error_log_path}")

    return True

# ---------- 主预处理类 ----------
class GCTPreprocessor:
    def __init__(self, sample_rate=22050, hop_length=256, win_length=1024, n_mels=128):
        self.sr = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

    def process_dataset(self, metadata_path, wav_dir, out_dir, tg_dir=None,
                        num_augment=3, num_workers=None, use_hdf5=False, batch_size=None, use_bert=False, bert_model="bert-base-multilingual-cased"):
        # 打印 bert_model 以便调试
        print(f"[GCTPreprocessor.process_dataset] bert_model = {bert_model}")

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        if batch_size is None:
            batch_size = estimate_batch_size(default_batch_size=32)

        print("启用异步 CPU-GPU 管道（自动硬件适配，保守内存策略）...")
        success = run_async_mode(metadata_path, wav_dir, out_dir, tg_dir,
                                 num_augment, num_workers, use_hdf5, batch_size, use_bert, bert_model)
        if not success:
            print("异步管道失败，请检查错误日志。")
            raise RuntimeError("预处理失败，请查看 errors.log 获取详细信息。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--tg_dir", type=str, default=None)
    parser.add_argument("--num_augment", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=None, help="默认自动设置为 CPU 核心数-1")
    parser.add_argument("--use_hdf5", action="store_true")
    parser.add_argument("--batch_size", type=str, default=None, help="批大小，可指定数字或'auto'（自动估算）")
    parser.add_argument("--use_bert", action="store_true", help="使用BERT前端（需在训练时对应启用）")
    parser.add_argument("--bert_model", type=str, default="bert-base-multilingual-cased", help="BERT模型名称或本地路径")
    args = parser.parse_args()

    # ========== 自动纠正 BERT 模型名称常见拼写错误 ==========
    common_bert_mistakes = {
        "bert-base-multilingual-cached": "bert-base-multilingual-cased",
    }
    if args.bert_model in common_bert_mistakes:
        corrected = common_bert_mistakes[args.bert_model]
        print(f"[自动纠正] BERT 模型名称 '{args.bert_model}' 疑似拼写错误，已自动更正为 '{corrected}'")
        args.bert_model = corrected
    # =====================================================

    # 打印命令行解析结果以便调试
    print(f"[__main__] 命令行解析结果:")
    print(f"  metadata = {args.metadata}")
    print(f"  wav_dir = {args.wav_dir}")
    print(f"  out_dir = {args.out_dir}")
    print(f"  use_bert = {args.use_bert}")
    print(f"  bert_model = {args.bert_model}")

    preprocessor = GCTPreprocessor()
    preprocessor.process_dataset(
        metadata_path=args.metadata,
        wav_dir=args.wav_dir,
        out_dir=args.out_dir,
        tg_dir=args.tg_dir,
        num_augment=args.num_augment,
        num_workers=args.num_workers,
        use_hdf5=args.use_hdf5,
        batch_size=args.batch_size,
        use_bert=args.use_bert,
        bert_model=args.bert_model
    )