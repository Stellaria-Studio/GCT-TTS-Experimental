import torch
import torchaudio
import numpy as np

# 生成 1kHz 正弦波（1秒）
sr = 22050
duration = 1.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
sine_wave = 0.5 * np.sin(2 * np.pi * 1000 * t)
wav_tensor = torch.from_numpy(sine_wave).unsqueeze(0).float()  # [1, T]

# 提取 Mel 谱（与预处理参数一致）
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    f_min=0.0,
    f_max=8000.0,
    n_mels=128,
    power=2.0,
    normalized=False,
    center=True,
    pad_mode='reflect',
    norm='slaney'
)
mel = mel_transform(wav_tensor)               # [1, n_mels, T]
mel = torch.log(torch.clamp(mel, min=1e-5))   # 取对数

# 加载声码器（修正版）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
try:
    vocoder = torch.hub.load('bshall/hifigan:main', 'hifigan', trust_repo=True)
    vocoder = vocoder.to(device)
except Exception as e:
    print(f"加载声码器失败: {e}")
    print("尝试在 CPU 上运行...")
    device = torch.device('cpu')
    vocoder = torch.hub.load('bshall/hifigan:main', 'hifigan', trust_repo=True)
    vocoder = vocoder.to(device)

vocoder.eval()

# 将输入移到相同设备
mel = mel.to(device)

# 合成
with torch.no_grad():
    wav_reconstructed = vocoder(mel)

# 保存音频（移到CPU保存）
wav_reconstructed = wav_reconstructed.cpu()
torchaudio.save("sine_reconstructed.wav", wav_reconstructed.squeeze(0), sr)
print("正弦波重建完成，请听 sine_reconstructed.wav")