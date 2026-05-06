import os
import torch
import torchaudio
import soundfile as sf
from faster_whisper import WhisperModel
from tqdm import tqdm


class AutoDatasetBuilder:
    def __init__(self, model_size="base", device="cuda", compute_type="float16"):
        print(f"正在加载 Faster-Whisper ({model_size}) 模型，请稍候...")
        self.model = WhisperModel(
            model_size,
            device=device if torch.cuda.is_available() else "cpu",
            compute_type=compute_type if torch.cuda.is_available() else "int8"
        )
        print("ASR 模型加载完毕！")

    def build_dataset(self, raw_audio_path, output_wav_dir, metadata_path, target_sr=22050):
        os.makedirs(output_wav_dir, exist_ok=True)

        print(f"正在读取原始音频: {raw_audio_path}")
        # 使用 soundfile 读取音频（保持原始采样率）
        waveform_np, sr = sf.read(raw_audio_path, dtype='float32')
        # 转为单声道（如果多声道则平均）
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=1)

        print("开始智能 VAD 断句与语音转文字...")
        segments, info = self.model.transcribe(
            raw_audio_path,
            beam_size=5,
            language="zh",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        valid_records = []
        segment_idx = 1

        for segment in tqdm(segments, desc="处理进度"):
            text = segment.text.strip()
            if len(text) < 2 or text.isascii():
                continue
            start_time = segment.start
            end_time = segment.end
            duration = end_time - start_time
            if duration < 1.5 or duration > 12.0:
                continue

            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)

            # 切片
            sliced_waveform_np = waveform_np[start_frame:end_frame]

            # 重采样到目标采样率（如果需要）
            if sr != target_sr:
                # 转换为 tensor 并使用 torchaudio 重采样（不依赖 FFmpeg）
                sliced_tensor = torch.from_numpy(sliced_waveform_np).unsqueeze(0).float()
                resampled_tensor = torchaudio.functional.resample(sliced_tensor, sr, target_sr)
                sliced_waveform_np = resampled_tensor.squeeze(0).numpy()

            wav_id = f"myvoice_{segment_idx:04d}"
            save_path = os.path.join(output_wav_dir, f"{wav_id}.wav")
            # 使用 soundfile 保存
            sf.write(save_path, sliced_waveform_np, target_sr)

            valid_records.append(f"{wav_id}|{text}")
            segment_idx += 1

        with open(metadata_path, "w", encoding="utf-8") as f:
            for record in valid_records:
                f.write(record + "\n")

        print(f"\n🎉 数据集构建完成！")
        print(f"共生成 {len(valid_records)} 条有效语音数据。")
        print(f"音频存放于: {output_wav_dir}")
        print(f"文本标注存放于: {metadata_path}")


if __name__ == "__main__":
    builder = AutoDatasetBuilder(model_size="small")
    raw_audio_file = "raw_record.wav"
    if os.path.exists(raw_audio_file):
        builder.build_dataset(
            raw_audio_path=raw_audio_file,
            output_wav_dir="data/wavs",
            metadata_path="data/metadata_auto.txt",
            target_sr=22050
        )
    else:
        print(f"请在当前目录准备一个名为 {raw_audio_file} 的长录音文件以供测试。")