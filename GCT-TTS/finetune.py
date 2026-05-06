# finetune.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from gct_tts import GCTTTS, GCTDataset, gct_collate_fn
# 复用我们在 train.py 中写好的损失函数
from train import compute_loss


def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 GCT-TTS 个人音色微调任务 (设备: {device})")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 初始化模型
    model = GCTTTS(vocab_size=5000, hidden_dim=256, num_layers=6).to(device)

    # 2. 加载预训练底模 (Base Model)
    if not os.path.exists(args.base_model):
        print(f"❌ 找不到底模文件: {args.base_model}")
        print("请确保您提供了一个在大规模开源数据集上训练好的底模权重！")
        return

    print(f"加载底模权重: {args.base_model}")
    checkpoint = torch.load(args.base_model, map_location=device, weights_only=True)
    # strict=False 允许加载可能因词表大小不同而轻微不匹配的权重
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # 3. 核心策略：冻结 (Freeze) 网络层
    # 我们冻结文本 Embedding 和 嵌套 GCN-Transformer 编码器的前 4 层
    # 这样可以保留底模强大的“普通话发音能力”和“句法情感理解能力”，防止灾难性遗忘
    print("🔒 正在冻结底层语义与图网络权重...")
    for param in model.embedding.parameters():
        param.requires_grad = False

    for i in range(len(model.encoder_layers) - 2):  # 假设一共6层，冻结前4层
        for param in model.encoder_layers[i].parameters():
            param.requires_grad = False

    print("🔓 解锁韵律预测器与声学解码器，主攻音色与语速学习...")
    # 剩下的层 (如 duration_predictor, pitch_predictor, mel_decoder) 默认 requires_grad=True

    # 4. 加载个人微调数据
    dataset = GCTDataset(metadata_path=args.metadata_path, feature_dir=args.feature_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=gct_collate_fn, num_workers=2
    )

    # 5. 配置优化器 (使用极小的学习率)
    # 由于是微调，学习率通常比从零训练小 5 到 10 倍
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化未冻结的参数
        lr=args.learning_rate
    )

    model.train()

    # 6. 开始微调循环
    print(f"🎧 开始微调，共 {args.epochs} 轮 (由于冻结了网络，显存占用会极低)...")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            text_seq = batch["text_seq"].to(device)
            adj_matrix = batch["adj_matrix"].to(device)
            mel_target = batch["mel_target"].to(device)
            dur_target = batch["duration_target"].to(device)
            pitch_target = batch["pitch_target"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            mel_mask = batch["mel_padding_mask"].to(device)

            mel_pred, log_dur_pred, pitch_pred = model(text_seq, adj_matrix, target_durations=dur_target)

            # 对齐 Mel 长度
            max_len = mel_target.size(1)
            if mel_pred.size(1) > max_len:
                mel_pred = mel_pred[:, :max_len, :]
            elif mel_pred.size(1) < max_len:
                mel_pred = torch.nn.functional.pad(mel_pred, (0, 0, 0, max_len - mel_pred.size(1)))

            # 计算损失
            loss, mel_l, dur_l, pitch_l = compute_loss(
                mel_pred, mel_target, log_dur_pred, dur_target, pitch_pred, pitch_target, src_mask, mel_mask
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"微调 Epoch[{epoch}/{args.epochs}] | Loss: {loss.item():.4f} (Mel: {mel_l.item():.4f})")

        # 保存微调后的专属模型
        save_path = os.path.join(args.save_dir, f"gct_tts_myvoice_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, save_path)
        print(f"✅ 专属音色模型已保存至: {save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCT-TTS 个人音色极速微调脚本")
    parser.add_argument("--base_model", type=str, required=True, help="预训练底模的权重路径")
    parser.add_argument("--metadata_path", type=str, default="data/features/train_metadata.txt")
    parser.add_argument("--feature_dir", type=str, default="data/features")
    parser.add_argument("--save_dir", type=str, default="checkpoints_finetuned")
    parser.add_argument("--batch_size", type=int, default=8, help="微调 Batch Size (可设置得小一点)")
    parser.add_argument("--epochs", type=int, default=20, help="微调轮数通常 20-50 轮即可")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="微调学习率极小")

    args = parser.parse_args()
    finetune(args)