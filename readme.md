# GCT-TTS: Graph Convolutional Transformer for Text-to-Speech

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-red)]()

**GCT-TTS** is a non‑autoregressive text‑to‑speech model that combines **Graph Convolutional Networks (GCN)** with **Transformers**.  
It explicitly models syntactic dependency graphs to improve the naturalness and controllability of long‑sentence prosody.

> ⚠️ **Project Status: Experimental / On hold**  
> The model architecture is fully implemented, but the generated acoustic features **do not yet yield audible speech** after vocoding.  
> We are actively debugging issues related to mel normalization, dynamic range, and vocoder input mismatch.  
> Community contributions are highly welcome!

---

## Features

- **Graph‑based Text Encoding** – spaCy dependency trees or BERT, with relation‑aware GCN layers.
- **Nested GCN‑Transformer Blocks** – Self‑attention + graph convolution + cross‑attention/gate fusion.
- **Sparse GCN Option** – Memory‑friendly sparse adjacency matrix support.
- **Multi‑Speaker & Style Modelling** – AdaIN speaker adaptation, VAE style encoder, optional normalising flows for duration.
- **Hybrid Loss** – Band‑weighted mel loss, duration/pitch MSE, temporal difference & range regularisation, optional STFT & GAN losses.
- **Asynchronous Preprocessing** – CPU producer / GPU consumer pipeline with speed/volume augmentation, HDF5 packing.
- **Multi‑Dataset Training** – Automatic speaker‑ID remapping, supports LibriTTS, LJSpeech, VCTK, AISHELL‑3 and more.
- **One‑Click Workflow** – `run_all.py` covers transcription, preprocessing and training end‑to‑end.

---

## Installation

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0 (CUDA 11.8+ recommended)
- Other dependencies (see code for full list)

### Setup
```bash
git clone https://github.com/StellariaStudio/gct-tts.git
cd gct-tts

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy librosa soundfile pyworld tgt h5py tqdm faster-whisper

# (Optional) For BERT frontend
pip install transformers
python -m spacy download en_core_web_sm
Quick Start

1) Single‑speaker personal voice

bash
cp your_recording.wav data/raw_record.wav
python run_all.py --use_bert --num_workers 4
2) Multi‑dataset large‑scale training

bash
python run_all.py --datasets libritts,ljspeech \
    --use_bert --has_speaker --epochs 300 --batch_size auto
Usage Details

Preprocessing

bash
python preprocess.py \
    --metadata data/metadata.txt \
    --wav_dir data/wavs \
    --out_dir data/features \
    --num_workers 4 --use_bert --bert_model bert-base-multilingual-cased
Training

bash
python train.py \
    --metadata_path data/features/train_metadata.txt \
    --feature_dir data/features \
    --save_dir checkpoints \
    --epochs 200 --batch_size 32 --fp16 \
    --use_bert --use_sparse_gcn --use_style_emb
Inference (experimental)

bash
python inference.py \
    --checkpoint checkpoints/gct_tts_epoch_105.pt \
    --vocab_path data/features/vocab.pt \
    --text "The weather is really nice today." \
    --output_dir inference_output \
    --save_audio --save_mel_plot
Known Issue – The generated mel‑spectrogram currently fails to produce intelligible speech when passed to a vocoder (Griffin‑Lim / HiFi‑GAN). Long‑term root causes may include:

Incorrect mel normalization / denormalization
Dynamic range mismatch
Vocoder input expectation
See Issues for details.
Project Structure

text
gct-tts/
├── gct_tts/                  # Core package
│   ├── __init__.py
│   ├── model.py
│   ├── modules.py
│   ├── frontend.py
│   ├── dataset.py
│   └── bert_utils.py
├── train.py
├── inference.py
├── preprocess.py
├── finetune.py
├── auto_transcribe.py
├── run_all.py
├── data/                     # Audio, metadata, features
├── checkpoints/
└── README.md
Technical Overview

Text Frontend – spaCy (dependency parsing) or BERT (self‑loop graph).
Encoder – Stacked NestedGCNTransformerBlock layers.
Prosody Predictors – CNN / normalising flow for duration, CNN for pitch.
Length Regulator & Decoder – Expand tokens to mel frames, fuse pitch/style/speaker, pass through Transformer decoder + HiFi‑GAN‑like post‑net.
Losses – Band‑weighted mel error (configurable high‑frequency emphasis), log‑duration MSE, pitch MSE, temporal difference and range regularisation.
Contributing

We warmly welcome contributions! Please read CONTRIBUTING.md for guidelines on how to report issues, suggest features, and submit pull requests.

Acknowledgements

FastSpeech 2
Grad‑TTS
GCN in Speech Synthesis
spaCy, Hugging Face Transformers
License

This project is licensed under the MIT License – see the LICENSE file for details.
Copyright © 2026 Stellaria Studio