#!/usr/bin/env python3
"""
GCT-TTS 全流程自动化脚本（增强版）
支持多数据集、BERT、稀疏GCN、风格建模等新特性。
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

# 添加 utils 目录到系统路径（假设 utils 与项目根目录同级）
PROJECT_ROOT = Path(__file__).parent.absolute()
UTILS_PATH = PROJECT_ROOT.parent / "utils"
if UTILS_PATH.exists():
    sys.path.insert(0, str(UTILS_PATH))
else:
    print("警告：未找到 utils 目录，下载功能将不可用。")

# ==================== 路径配置 ====================
DATA_DIR = PROJECT_ROOT / "data"
WAVS_DIR = DATA_DIR / "wavs"
FEATURES_DIR = DATA_DIR / "features"
TRANSCRIBE_META = DATA_DIR / "metadata_auto.txt"
FINAL_META = DATA_DIR / "metadata.txt"
PREPROCESS_META = FEATURES_DIR / "train_metadata.txt"
BASE_MODEL = PROJECT_ROOT / "checkpoints" / "base.pt"

STEP1_FLAG = DATA_DIR / ".step1_done"
STEP2_FLAG = FEATURES_DIR / ".step2_done"
STEP3_FLAG = PROJECT_ROOT / "checkpoints" / ".step3_done"
STEP4_FLAG = PROJECT_ROOT / "checkpoints" / ".step4_done"

# 数据集 URL 映射
DATASET_URLS = {
    'libritts': 'https://openslr.elda.org/resources/60/train-clean-100.tar.gz',
    'ljspeech': 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
    'vctk': 'https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip',
    'aishell3': 'https://www.openslr.org/resources/93/data_aishell3.tgz'
}

# 解压根目录映射
EXTRACT_ROOTS = {
    'libritts': 'train-clean-100',
    'ljspeech': 'LJSpeech-1.1',
    'vctk': 'VCTK-Corpus-0.92',
    'aishell3': 'data_aishell3'
}

# ==================== 工具函数 ====================
def run_cmd(cmd, desc):
    print(f"\n=== {desc} ===")
    print(f"执行: {cmd}")
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    try:
        for line in process.stdout:
            print(line, end='')
    except Exception as e:
        print(f"读取输出时出错: {e}", file=sys.stderr)
    ret = process.wait()
    if ret != 0:
        print(f"错误: {desc} 失败，退出码 {ret}", file=sys.stderr)
        sys.exit(ret)
    print(f"完成: {desc}")

def should_skip_step(flag_file, required_items, description):
    all_ok = True
    for item in required_items:
        if callable(item):
            if not item():
                all_ok = False
                break
        elif isinstance(item, Path):
            if not item.exists():
                all_ok = False
                break
            if item.is_dir() and not any(item.iterdir()):
                all_ok = False
                break
    if all_ok:
        if not flag_file.exists():
            flag_file.touch()
            print(f"✓ 检测到 {description} 已完成（输出完好），自动创建标志。")
        else:
            print(f"✓ {description} 已完成，跳过。")
        return True
    else:
        if flag_file.exists():
            flag_file.unlink()
            print(f"! {description} 标志存在但输出缺失，将重新运行。")
        return False

def check_pt_files_excluding_base(ckpt_dir):
    if not ckpt_dir.exists():
        return False
    pt_files = list(ckpt_dir.glob("*.pt"))
    return any(f.name != "base.pt" for f in pt_files)

def check_finetune_pt():
    return (PROJECT_ROOT / "checkpoints" / "finetune.pt").exists()

def check_feature_completeness(feature_dir, meta_file, use_hdf5=False):
    if not meta_file.exists():
        return False
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            ids = [line.split('|')[0] for line in f if line.strip()]
    except:
        return False
    if not ids:
        return False

    if use_hdf5:
        h5_path = feature_dir / 'features.h5'
        if not h5_path.exists():
            return False
        try:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                sample_id = ids[0]
                if sample_id not in f:
                    return False
        except:
            return False
    else:
        sample_size = min(5, len(ids))
        for fid in ids[:sample_size]:
            for suffix in ['_seq.pt', '_adj.pt', '_edge_type.pt', '_mel.pt', '_dur.pt', '_pitch.pt']:
                if not (feature_dir / f"{fid}{suffix}").exists():
                    return False
        total_expected = len(ids) * 6
        total_present = sum(1 for _ in feature_dir.glob("*.pt"))
        if total_present < total_expected:
            return False
    return True

def check_download_complete(dataset_name):
    """严格检查数据集是否下载并解压完整"""
    raw_meta = DATA_DIR / f"metadata_{dataset_name}.txt"
    wav_dir = DATA_DIR / "wavs"
    if not raw_meta.exists() or raw_meta.stat().st_size == 0:
        return False
    if not wav_dir.exists() or not any(wav_dir.glob("*.wav")):
        return False
    # 抽样检查前5个音频
    try:
        with open(raw_meta, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                wav_id = line.strip().split('|')[0]
                if not (wav_dir / f"{wav_id}.wav").exists():
                    return False
    except:
        return False
    return True

# ==================== 单录音模式 ====================
def run_single_recording_mode(args):
    raw_audio = DATA_DIR / "raw_record.wav"
    if not raw_audio.exists():
        print(f"错误: 未找到 {raw_audio}，请将原始录音放在 data/raw_record.wav")
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    WAVS_DIR.mkdir(exist_ok=True)
    FEATURES_DIR.mkdir(exist_ok=True)
    (PROJECT_ROOT / "checkpoints").mkdir(exist_ok=True)

    step1_required = [
        TRANSCRIBE_META,
        lambda: WAVS_DIR.exists() and any(WAVS_DIR.iterdir()),
        FINAL_META
    ]
    if not should_skip_step(STEP1_FLAG, step1_required, "音频转写与切分"):
        target_wav = PROJECT_ROOT / "raw_record.wav"
        if target_wav.exists():
            target_wav.unlink()
        try:
            os.symlink(raw_audio, target_wav)
        except OSError:
            shutil.copy2(raw_audio, target_wav)

        run_cmd("python auto_transcribe.py", "音频转写与切分")

        if target_wav.exists():
            target_wav.unlink()

        if not TRANSCRIBE_META.exists():
            print(f"错误: 转写后未生成 {TRANSCRIBE_META}")
            sys.exit(1)

        shutil.copy2(TRANSCRIBE_META, FINAL_META)
        print(f"已使用转写结果作为最终标注: {FINAL_META}")
        STEP1_FLAG.touch()
    else:
        if not FINAL_META.exists() and TRANSCRIBE_META.exists():
            shutil.copy2(TRANSCRIBE_META, FINAL_META)
            print(f"已补充 {FINAL_META}（从 {TRANSCRIBE_META} 复制）")

    step2_required = [
        PREPROCESS_META,
        lambda: check_feature_completeness(FEATURES_DIR, PREPROCESS_META, use_hdf5=args.use_hdf5)
    ]
    if not should_skip_step(STEP2_FLAG, step2_required, "特征预处理"):
        tg_dir = DATA_DIR / "textgrids"
        tg_arg = f"--tg_dir {tg_dir}" if tg_dir.exists() else ""
        workers_arg = f"--num_workers {args.num_workers}" if args.num_workers else ""
        hdf5_arg = "--use_hdf5" if args.use_hdf5 else ""
        batch_arg = f"--batch_size {args.batch_size}" if args.batch_size else ""
        use_bert_arg = "--use_bert" if args.use_bert else ""
        bert_model_arg = f"--bert_model {args.bert_model}" if args.use_bert else ""
        cmd_pre = (f"python preprocess.py --metadata {FINAL_META} --wav_dir {WAVS_DIR} --out_dir {FEATURES_DIR} "
                   f"{tg_arg} {workers_arg} {hdf5_arg} {batch_arg} {use_bert_arg} {bert_model_arg}")
        run_cmd(cmd_pre, "特征预处理")
        STEP2_FLAG.touch()
    else:
        print("特征预处理已完成，跳过。")

    step3_required = [lambda: check_pt_files_excluding_base(PROJECT_ROOT / "checkpoints")]
    if not should_skip_step(STEP3_FLAG, step3_required, "模型训练"):
        cmd_train = (f"python train.py --metadata_path {PREPROCESS_META} --feature_dir {FEATURES_DIR} "
                     f"--save_dir {PROJECT_ROOT / 'checkpoints'} --epochs {args.epochs} "
                     f"--batch_size {args.batch_size} ")
        if args.fp16:
            cmd_train += "--fp16 "
        if args.use_bert:
            cmd_train += " --use_bert"
        if args.bert_model:
            cmd_train += f" --bert_model {args.bert_model}"
        if args.use_sparse_gcn:
            cmd_train += " --use_sparse_gcn"
        if args.use_adaln:
            cmd_train += " --use_adaln"
        if args.use_style_emb:
            cmd_train += " --use_style_emb"
        if args.use_vae:
            cmd_train += " --use_vae"
        if args.use_implicit_align:
            cmd_train += " --use_implicit_align"
        if args.checkpointing:
            cmd_train += " --checkpointing"
        if args.use_multi_scale_stft:
            cmd_train += " --use_multi_scale_stft"
        if args.use_uncertainty:
            cmd_train += " --use_uncertainty"
        if args.use_flow_duration:
            cmd_train += " --use_flow_duration"
        if args.use_noam:
            cmd_train += " --use_noam"
        if args.lambda_time:
            cmd_train += f" --lambda_time {args.lambda_time}"
        if args.lambda_range:
            cmd_train += f" --lambda_range {args.lambda_range}"
        if args.hidden_dim:
            cmd_train += f" --hidden_dim {args.hidden_dim}"
        if args.num_layers:
            cmd_train += f" --num_layers {args.num_layers}"
        if args.num_heads:
            cmd_train += f" --num_heads {args.num_heads}"
        if args.ffn_dim:
            cmd_train += f" --ffn_dim {args.ffn_dim}"
        if args.num_decoder_layers:
            cmd_train += f" --num_decoder_layers {args.num_decoder_layers}"
        if args.grad_clip:
            cmd_train += f" --grad_clip {args.grad_clip}"
        if args.lr_factor:
            cmd_train += f" --lr_factor {args.lr_factor}"
        if args.lr_patience:
            cmd_train += f" --lr_patience {args.lr_patience}"
        if args.max_ckpt_keep:
            cmd_train += f" --max_ckpt_keep {args.max_ckpt_keep}"
        if args.early_stop_patience:
            cmd_train += f" --early_stop_patience {args.early_stop_patience}"

        run_cmd(cmd_train, "模型训练")
        STEP3_FLAG.touch()
    else:
        print("模型训练已完成，跳过。")

    if BASE_MODEL.exists():
        finetune_script = PROJECT_ROOT / "finetune.py"
        if not finetune_script.exists():
            print("未找到 finetune.py，跳过微调步骤。")
        else:
            step4_required = [check_finetune_pt]
            if not should_skip_step(STEP4_FLAG, step4_required, "模型微调"):
                cmd_finetune = f"python finetune.py --base_model {BASE_MODEL} --metadata_path {PREPROCESS_META} --feature_dir {FEATURES_DIR} --epochs 20"
                run_cmd(cmd_finetune, "模型微调")
                STEP4_FLAG.touch()
            else:
                print("模型微调已完成，跳过。")
    else:
        print(f"未找到底模 {BASE_MODEL}，跳过微调。")

    print("\n🎉 单录音模式完成！模型保存在 checkpoints/ 目录。")

# ==================== 多数据集模式 ====================
def run_multi_dataset_mode(args):
    datasets = [ds.strip() for ds in args.datasets.split(',') if ds.strip()]
    if not datasets:
        print("错误: --datasets 参数无效")
        sys.exit(1)

    print(f"将处理多个数据集: {datasets}")
    print(f"[DEBUG] args.bert_model in run_multi_dataset_mode: {args.bert_model}")  # 调试输出

    meta_paths = []
    feature_dirs = []

    for ds in datasets:
        print(f"\n===== 处理数据集: {ds} =====")

        raw_meta = DATA_DIR / f"metadata_{ds}.txt"
        wav_dir = DATA_DIR / "wavs"
        feature_dir = DATA_DIR / f"features_{ds}"
        train_meta = feature_dir / "train_metadata.txt"
        step1_flag = DATA_DIR / f".{ds}_download_done"

        # ---------- 下载 ----------
        def check_download():
            return check_download_complete(ds)

        step1_required = [check_download]
        if not should_skip_step(step1_flag, step1_required, f"下载数据集 {ds}"):
            # 增强清理：删除所有相关文件，确保全新下载
            print(f"清理数据集 {ds} 的不完整文件...")

            # 1. 删除元数据
            if raw_meta.exists():
                raw_meta.unlink()
                print(f"删除元数据: {raw_meta}")

            # 2. 删除 wavs 目录下属于该数据集的音频文件
            if wav_dir.exists():
                # 安全做法：删除所有以数据集前缀开头的 wav 文件
                for wav_file in wav_dir.glob(f"{ds}_*.wav"):
                    wav_file.unlink()
                    print(f"删除音频: {wav_file}")

            # 3. 删除解压根目录
            extract_root_name = EXTRACT_ROOTS.get(ds)
            if extract_root_name:
                extract_root = DATA_DIR / extract_root_name
                if extract_root.exists():
                    shutil.rmtree(extract_root)
                    print(f"删除解压根目录: {extract_root}")

            # 4. 删除压缩包
            url = DATASET_URLS.get(ds)
            if url:
                archive_name = url.split('/')[-1]
                archive_path = DATA_DIR / archive_name
                if archive_path.exists():
                    archive_path.unlink()
                    print(f"删除压缩包: {archive_path}")

            # 5. 删除标志文件
            if step1_flag.exists():
                step1_flag.unlink()

            # 确保 data 目录存在
            DATA_DIR.mkdir(exist_ok=True)

            # 调用下载函数
            try:
                import download_dataset
                download_dataset.download_and_prepare(ds, str(DATA_DIR))
            except ImportError:
                print("错误：无法导入 download_dataset 模块。请确保 utils/download_dataset.py 存在。")
                sys.exit(1)
            step1_flag.touch()
        else:
            print(f"数据集 {ds} 已下载，跳过。")

        # ---------- 预处理 ----------
        step2_flag = DATA_DIR / f".{ds}_preprocess_done"

        def check_preprocess():
            return check_feature_completeness(feature_dir, train_meta, use_hdf5=args.use_hdf5)

        step2_required = [check_preprocess]
        if not should_skip_step(step2_flag, step2_required, f"特征预处理 {ds}"):
            # 如果特征目录已存在，先删除（确保重新生成）
            if feature_dir.exists():
                shutil.rmtree(feature_dir)
                print(f"已删除旧特征目录: {feature_dir}")
            feature_dir.mkdir(exist_ok=True, parents=True)
            workers_arg = f"--num_workers {args.num_workers}" if args.num_workers else ""
            hdf5_arg = "--use_hdf5" if args.use_hdf5 else ""
            batch_arg = f"--batch_size {args.batch_size}" if args.batch_size else ""
            use_bert_arg = "--use_bert" if args.use_bert else ""
            bert_model_arg = f"--bert_model {args.bert_model}" if args.use_bert else ""
            num_augment_arg = f"--num_augment {args.num_augment}" if args.num_augment else ""
            cmd_pre = (f"python preprocess.py --metadata {raw_meta} --wav_dir {wav_dir} "
                       f"--out_dir {feature_dir} {num_augment_arg} {workers_arg} {hdf5_arg} {batch_arg} {use_bert_arg} {bert_model_arg}")
            run_cmd(cmd_pre, f"特征预处理 {ds}")
            step2_flag.touch()
        else:
            print(f"特征预处理 {ds} 已完成，跳过。")

        meta_paths.append(str(train_meta))
        feature_dirs.append(str(feature_dir))

    # ---------- 合并训练 ----------
    combined_name = "_".join(datasets)
    ckpt_dir = PROJECT_ROOT / "checkpoints" / combined_name
    step3_flag = DATA_DIR / f".{combined_name}_train_done"

    def check_train_done():
        return check_pt_files_excluding_base(ckpt_dir)

    if not should_skip_step(step3_flag, [check_train_done], f"合并训练 {combined_name}"):
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        cmd_train = (f"python train.py "
                     f"--multi_dataset "
                     f"--metadata_path {','.join(meta_paths)} "
                     f"--feature_dir {','.join(feature_dirs)} "
                     f"--has_speaker "
                     f"--save_dir {ckpt_dir} "
                     f"--epochs {args.epochs or 200} "
                     f"--batch_size {args.batch_size or 'auto'} ")
        if args.fp16:
            cmd_train += "--fp16 "
        if args.accumulation_steps:
            cmd_train += f"--accumulation_steps {args.accumulation_steps} "
        if args.use_hdf5:
            cmd_train += "--use_hdf5 "
        if args.use_bert:
            cmd_train += "--use_bert "
        if args.bert_model:
            cmd_train += f"--bert_model {args.bert_model} "
        if args.use_sparse_gcn:
            cmd_train += "--use_sparse_gcn "
        if args.use_adaln:
            cmd_train += "--use_adaln "
        if args.use_style_emb:
            cmd_train += "--use_style_emb "
        if args.use_vae:
            cmd_train += "--use_vae "
        if args.use_implicit_align:
            cmd_train += "--use_implicit_align "
        if args.checkpointing:
            cmd_train += "--checkpointing "
        if args.use_multi_scale_stft:
            cmd_train += "--use_multi_scale_stft "
        if args.use_uncertainty:
            cmd_train += "--use_uncertainty "
        if args.use_flow_duration:
            cmd_train += "--use_flow_duration "
        if args.use_noam:
            cmd_train += "--use_noam "
        if args.lambda_time:
            cmd_train += f"--lambda_time {args.lambda_time} "
        if args.lambda_range:
            cmd_train += f"--lambda_range {args.lambda_range} "
        if args.hidden_dim:
            cmd_train += f"--hidden_dim {args.hidden_dim} "
        if args.num_layers:
            cmd_train += f"--num_layers {args.num_layers} "
        if args.num_heads:
            cmd_train += f"--num_heads {args.num_heads} "
        if args.ffn_dim:
            cmd_train += f"--ffn_dim {args.ffn_dim} "
        if args.num_decoder_layers:
            cmd_train += f"--num_decoder_layers {args.num_decoder_layers} "
        if args.grad_clip:
            cmd_train += f"--grad_clip {args.grad_clip} "
        if args.lr_factor:
            cmd_train += f"--lr_factor {args.lr_factor} "
        if args.lr_patience:
            cmd_train += f"--lr_patience {args.lr_patience} "
        if args.max_ckpt_keep:
            cmd_train += f"--max_ckpt_keep {args.max_ckpt_keep} "
        if args.early_stop_patience:
            cmd_train += f"--early_stop_patience {args.early_stop_patience} "

        run_cmd(cmd_train, f"合并训练 {combined_name}")
        step3_flag.touch()
    else:
        print(f"合并训练 {combined_name} 已完成，跳过。")

    print(f"\n🎉 多数据集 {combined_name} 全流程完成！模型保存在 {ckpt_dir}")

# ==================== 主入口 ====================
def main():
    parser = argparse.ArgumentParser(description="GCT-TTS 一键全流程脚本（支持多数据集）")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集名称，例如 libritts,ljspeech（若指定则进入多数据集模式）")
    # 通用参数
    parser.add_argument("--num_workers", type=int, default=None, help="指定预处理使用的进程数（核心数）")
    parser.add_argument("--use_hdf5", action="store_true", help="启用 HDF5 打包")
    parser.add_argument("--batch_size", type=str, default="32", help="训练/预处理批大小，可指定数字或'auto'")
    parser.add_argument("--num_augment", type=int, default=3, help="数据增强次数")
    parser.add_argument("--fp16", action="store_true", help="启用混合精度训练")

    # 新特性开关
    parser.add_argument("--use_bert", action="store_true", help="使用BERT作为文本编码器")
    parser.add_argument("--bert_model", type=str, default="bert-base-multilingual-cased", help="BERT模型名称或本地路径")
    parser.add_argument("--use_sparse_gcn", action="store_true", help="使用稀疏图卷积")
    parser.add_argument("--use_adaln", action="store_true", help="使用AdaIN进行说话人自适应")
    parser.add_argument("--use_style_emb", action="store_true", help="使用风格嵌入")
    parser.add_argument("--use_vae", action="store_true", help="使用VAE风格建模")
    parser.add_argument("--use_implicit_align", action="store_true", help="使用隐式对齐替代显式时长")
    parser.add_argument("--checkpointing", action="store_true", help="启用激活检查点以节省显存")
    parser.add_argument("--use_multi_scale_stft", action="store_true", help="使用多尺度STFT损失")
    parser.add_argument("--use_uncertainty", action="store_true", help="使用不确定性加权损失")

    # 模型架构参数
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--num_decoder_layers", type=int, default=6)

    # 多数据集模式参数
    parser.add_argument("--has_speaker", action="store_true", default=True,
                        help="数据集是否包含说话人ID（多数据集默认 True）")
    parser.add_argument("--use_flow_duration", action="store_true", help="使用流时长预测器")
    parser.add_argument("--use_noam", action="store_true", help="使用Noam调度器")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--lambda_time", type=float, default=0.3)
    parser.add_argument("--lambda_range", type=float, default=0.3)

    # 新增训练控制参数
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="学习率衰减因子（用于ReduceLROnPlateau）")
    parser.add_argument("--lr_patience", type=int, default=5, help="学习率衰减的耐心轮数")
    parser.add_argument("--max_ckpt_keep", type=int, default=5, help="最多保留的checkpoint数量")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="早停耐心轮数（0表示禁用早停）")

    args = parser.parse_args()

    # ========== 自动纠正 BERT 模型名称常见拼写错误 ==========
    common_bert_mistakes = {
        "bert-base-multilingual-cached": "bert-base-multilingual-cased",
        # 如有其他常见错误可继续添加
    }
    if args.bert_model in common_bert_mistakes:
        corrected = common_bert_mistakes[args.bert_model]
        print(f"[自动纠正] BERT 模型名称 '{args.bert_model}' 疑似拼写错误，已自动更正为 '{corrected}'")
        args.bert_model = corrected
    # =====================================================

    # 打印所有参数以便调试
    print("[DEBUG] 命令行参数解析结果:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # 确保必要的目录存在
    DATA_DIR.mkdir(exist_ok=True)
    (PROJECT_ROOT / "checkpoints").mkdir(exist_ok=True)

    if args.datasets:
        run_multi_dataset_mode(args)
    else:
        run_single_recording_mode(args)

if __name__ == "__main__":
    main()