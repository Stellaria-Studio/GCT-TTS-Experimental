import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class GCTDataset(Dataset):
    def __init__(self, metadata_path, feature_dir, has_speaker=False, use_hdf5=False):
        self.feature_dir = feature_dir
        self.has_speaker = has_speaker
        self.use_hdf5 = use_hdf5
        self.data_list = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    self.data_list.append(parts)
        self.h5_file = None  # 占位符，不会在 pickling 中使用

    def __getitem__(self, idx):
        parts = self.data_list[idx]
        wav_id = parts[0]
        speaker_id = int(parts[2]) if self.has_speaker and len(parts) >= 3 else 0

        if self.use_hdf5:
            with h5py.File(os.path.join(self.feature_dir, 'features.h5'), 'r') as h5f:
                grp = h5f[wav_id]
                text_seq = torch.tensor(grp['seq'][:], dtype=torch.long)
                adj_matrix = torch.tensor(grp['adj'][:], dtype=torch.float32)
                edge_type = torch.tensor(grp['edge_type'][:], dtype=torch.long)
                duration = torch.tensor(grp['dur'][:], dtype=torch.long)
                pitch = torch.tensor(grp['pitch'][:], dtype=torch.float32)
                mel = torch.tensor(grp['mel'][:], dtype=torch.float32)
        else:
            text_seq = torch.load(os.path.join(self.feature_dir, f"{wav_id}_seq.pt"))
            adj_matrix = torch.load(os.path.join(self.feature_dir, f"{wav_id}_adj.pt"))
            edge_type = torch.load(os.path.join(self.feature_dir, f"{wav_id}_edge_type.pt"))
            duration = torch.load(os.path.join(self.feature_dir, f"{wav_id}_dur.pt"))
            pitch = torch.load(os.path.join(self.feature_dir, f"{wav_id}_pitch.pt"))
            mel = torch.load(os.path.join(self.feature_dir, f"{wav_id}_mel.pt"))

        return {
            "text_seq": text_seq,
            "adj_matrix": adj_matrix,
            "edge_type": edge_type,
            "duration": duration,
            "pitch": pitch,
            "mel": mel,
            "speaker_id": speaker_id
        }

    def __len__(self):
        return len(self.data_list)

class MultiDataset(Dataset):
    """支持多个数据集合并，并自动重映射说话人 ID 为全局连续 ID"""
    def __init__(self, metadata_paths, feature_dirs, has_speaker=False, use_hdf5=False):
        self.has_speaker = has_speaker
        self.datasets = []
        self.speaker_map = {}
        self.global_speaker_counter = 0
        self.cumulative_len = [0]

        meta_list = metadata_paths.split(',') if isinstance(metadata_paths, str) else metadata_paths
        feat_list = feature_dirs.split(',') if isinstance(feature_dirs, str) else feature_dirs

        for meta_path, feat_dir in zip(meta_list, feat_list):
            meta_path = meta_path.strip()
            feat_dir = feat_dir.strip()
            ds = GCTDataset(meta_path, feat_dir, has_speaker, use_hdf5)
            self.datasets.append(ds)
            ds_len = len(ds)
            self.cumulative_len.append(self.cumulative_len[-1] + ds_len)

            if has_speaker:
                for i in range(ds_len):
                    item = ds[i]
                    orig_id = item['speaker_id']
                    if orig_id not in self.speaker_map:
                        self.speaker_map[orig_id] = self.global_speaker_counter
                        self.global_speaker_counter += 1

    def __len__(self):
        return self.cumulative_len[-1]

    def __getitem__(self, idx):
        for i in range(len(self.datasets)):
            if idx < self.cumulative_len[i+1]:
                item = self.datasets[i][idx - self.cumulative_len[i]]
                if self.has_speaker:
                    item['speaker_id'] = self.speaker_map[item['speaker_id']]
                return item
        raise IndexError

def gct_collate_fn(batch):
    text_lens = [b["text_seq"].size(0) for b in batch]
    mel_lens = [b["mel"].size(0) for b in batch]
    max_text_len = max(text_lens)
    max_mel_len = max(mel_lens)
    B = len(batch)
    mel_dim = batch[0]["mel"].size(1)

    texts_padded = torch.zeros((B, max_text_len), dtype=torch.long)
    adjs_padded = torch.zeros((B, max_text_len, max_text_len), dtype=torch.float32)
    edge_types_padded = torch.zeros((B, max_text_len, max_text_len), dtype=torch.long)
    durations_padded = torch.zeros((B, max_text_len), dtype=torch.long)
    pitches_padded = torch.zeros((B, max_text_len), dtype=torch.float32)
    mels_padded = torch.zeros((B, max_mel_len, mel_dim), dtype=torch.float32)
    speaker_ids = torch.zeros(B, dtype=torch.long)

    src_padding_mask = torch.ones((B, max_text_len), dtype=torch.bool)
    mel_padding_mask = torch.ones((B, max_mel_len), dtype=torch.bool)

    for i, b in enumerate(batch):
        t_len = text_lens[i]
        m_len = mel_lens[i]
        texts_padded[i, :t_len] = b["text_seq"]
        adjs_padded[i, :t_len, :t_len] = b["adj_matrix"]
        edge_types_padded[i, :t_len, :t_len] = b["edge_type"]
        durations_padded[i, :t_len] = b["duration"]
        pitches_padded[i, :t_len] = b["pitch"]
        mels_padded[i, :m_len, :] = b["mel"]
        src_padding_mask[i, :t_len] = False
        mel_padding_mask[i, :m_len] = False
        speaker_ids[i] = b["speaker_id"]

    return {
        "text_seq": texts_padded,
        "adj_matrix": adjs_padded,
        "edge_type": edge_types_padded,
        "duration_target": durations_padded,
        "pitch_target": pitches_padded,
        "mel_target": mels_padded,
        "src_padding_mask": src_padding_mask,
        "mel_padding_mask": mel_padding_mask,
        "speaker_ids": speaker_ids,
        "text_lens": text_lens,
        "mel_lens": mel_lens
    }