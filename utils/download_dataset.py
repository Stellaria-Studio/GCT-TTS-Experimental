import os
import shutil
import tarfile
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def extract_archive(archive_path, extract_dir):
    try:
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(extract_dir)
            print(f"Extracted tar archive: {archive_path}")
            return
    except (tarfile.TarError, EOFError):
        pass
    try:
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(extract_dir)
            print(f"Extracted zip archive: {archive_path}")
            return
    except zipfile.BadZipFile:
        pass
    raise ValueError(f"Unsupported or corrupted archive format: {archive_path}")

# ---------- 数据集有效性验证函数 ----------
def is_valid_libritts_root(root):
    """检查目录是否为有效的LibriTTS根目录（包含至少一个说话人目录及其音频）"""
    root = Path(root)
    # 应该包含以数字命名的子目录（说话人ID）
    for item in root.iterdir():
        if item.is_dir() and item.name.isdigit():
            for chapter in item.iterdir():
                if chapter.is_dir() and any(chapter.glob('*.wav')):
                    return True
    return False

def is_valid_ljspeech_root(root):
    root = Path(root)
    # 应包含 metadata.csv 和 wavs 目录，且 wavs 下至少有一个音频
    if (root / 'metadata.csv').exists() and (root / 'wavs').exists():
        if any((root / 'wavs').glob('*.wav')):
            return True
    return False

def is_valid_vctk_root(root):
    root = Path(root)
    # 应包含 txt 和 wav48_silence_trimmed（或 wav48）目录，且至少有一个音频
    txt_dir = root / 'txt'
    wav_dir = root / 'wav48_silence_trimmed'
    if not wav_dir.exists():
        wav_dir = root / 'wav48'
    if txt_dir.exists() and wav_dir.exists():
        if any(wav_dir.glob('*/*.wav')):
            return True
    return False

def is_valid_aishell3_root(root):
    root = Path(root)
    # 应包含 train 目录，且 train/wav 下有音频
    train_dir = root / 'train'
    if train_dir.exists():
        wav_dir = train_dir / 'wav'
        if wav_dir.exists() and any(wav_dir.glob('*.wav')):
            return True
    return False

# ---------- 查找数据根目录 ----------
def find_data_root(extract_dir, dataset_name):
    """查找数据集根目录，并验证有效性，若无效返回 None"""
    extract_dir = Path(extract_dir)
    if dataset_name == 'ljspeech':
        # 常见情况：解压后得到 LJSpeech-1.1 目录
        for item in extract_dir.iterdir():
            if item.is_dir() and is_valid_ljspeech_root(item):
                return item
        if is_valid_ljspeech_root(extract_dir):
            return extract_dir
    elif dataset_name == 'libritts':
        candidates = ['LibriTTS', 'train-clean-100', 'train-clean-360', 'train-other-500',
                      'dev-clean', 'dev-other', 'test-clean', 'test-other']
        for cand in candidates:
            cand_path = extract_dir / cand
            if cand_path.exists() and is_valid_libritts_root(cand_path):
                return cand_path
        if is_valid_libritts_root(extract_dir):
            return extract_dir
    elif dataset_name == 'vctk':
        for item in extract_dir.iterdir():
            if item.is_dir() and is_valid_vctk_root(item):
                return item
        if is_valid_vctk_root(extract_dir):
            return extract_dir
    elif dataset_name == 'aishell3':
        for item in extract_dir.iterdir():
            if item.is_dir() and is_valid_aishell3_root(item):
                return item
        if is_valid_aishell3_root(extract_dir):
            return extract_dir
    return None

# ---------- 数据准备函数（各数据集） ----------
def prepare_libritts(data_root, dataset_prefix):
    wav_dir = data_root.parent / 'wavs'
    wav_dir.mkdir(exist_ok=True)
    meta_lines = []

    # 如果 data_root 直接是说话人目录的集合，遍历其子目录
    speaker_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()]
    if not speaker_dirs:
        # 可能 data_root 下还有一级（例如 train-clean-100 下是说话人目录）
        for sub in data_root.iterdir():
            if sub.is_dir():
                speaker_dirs.extend([d for d in sub.iterdir() if d.is_dir() and d.name.isdigit()])
    if not speaker_dirs:
        raise RuntimeError(f"在 {data_root} 下未找到任何说话人目录")

    for spk_dir in speaker_dirs:
        speaker_id = spk_dir.name
        for chapter_dir in spk_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            wav_files = list(chapter_dir.glob('*.wav'))
            print(f"  在 {chapter_dir} 中找到 {len(wav_files)} 个 wav 文件")
            for wav_file in wav_files:
                orig_wav_id = wav_file.stem
                wav_id = f"{dataset_prefix}_{orig_wav_id}"
                # 查找对应的文本文件（.normalized.txt 或 .txt）
                txt_file = wav_file.with_suffix('.normalized.txt')
                if not txt_file.exists():
                    txt_file = wav_file.with_suffix('.txt')
                if txt_file.exists():
                    text = txt_file.read_text(encoding='utf-8').strip()
                    dest_path = wav_dir / f"{wav_id}.wav"
                    shutil.copy(wav_file, dest_path)
                    meta_lines.append(f"{wav_id}|{text}|{speaker_id}")
                else:
                    print(f"警告：未找到 {wav_file} 对应的文本文件")

    if len(meta_lines) == 0:
        raise RuntimeError(f"LibriTTS 数据准备失败：在 {data_root} 下未找到任何有效音频-文本对")
    return meta_lines

def prepare_ljspeech(data_root, dataset_prefix):
    wav_dir = data_root.parent / 'wavs'
    wav_dir.mkdir(exist_ok=True)
    meta_lines = []

    metadata_file = data_root / 'metadata.csv'
    wav_source_dir = data_root / 'wavs'
    if not metadata_file.exists():
        # 尝试在子目录中查找
        for sub in data_root.iterdir():
            if sub.is_dir() and (sub / 'metadata.csv').exists():
                metadata_file = sub / 'metadata.csv'
                wav_source_dir = sub / 'wavs'
                break
    if not metadata_file.exists():
        raise FileNotFoundError(f"无法找到 metadata.csv，请检查数据集结构：{data_root}")

    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                orig_wav_id, _, text = parts[:3]
                wav_id = f"{dataset_prefix}_{orig_wav_id}"
                src = wav_source_dir / f"{orig_wav_id}.wav"
                if not src.exists():
                    print(f"警告：音频文件 {src} 不存在，跳过")
                    continue
                dst = wav_dir / f"{wav_id}.wav"
                shutil.copy(src, dst)
                meta_lines.append(f"{wav_id}|{text}|0")

    if len(meta_lines) == 0:
        raise RuntimeError(f"LJSpeech 数据准备失败：在 {data_root} 下未找到任何有效音频-文本对")
    return meta_lines

def prepare_vctk(data_root, dataset_prefix):
    wav_dir = data_root.parent / 'wavs'
    wav_dir.mkdir(exist_ok=True)
    meta_lines = []

    txt_dir = data_root / 'txt'
    wav_source_dir = data_root / 'wav48_silence_trimmed'
    if not wav_source_dir.exists():
        wav_source_dir = data_root / 'wav48'

    if not txt_dir.exists():
        raise FileNotFoundError(f"未找到 txt 目录：{txt_dir}")

    for wav_file in wav_source_dir.glob('*/*.wav'):
        orig_speaker_id = wav_file.parent.name
        orig_wav_id = wav_file.stem
        wav_id = f"{dataset_prefix}_{orig_speaker_id}_{orig_wav_id}"
        txt_file = txt_dir / orig_speaker_id / f"{orig_wav_id}.txt"
        if txt_file.exists():
            text = txt_file.read_text(encoding='utf-8').strip()
            dst = wav_dir / f"{wav_id}.wav"
            shutil.copy(wav_file, dst)
            meta_lines.append(f"{wav_id}|{text}|{orig_speaker_id}")
        else:
            print(f"警告：未找到 {wav_file} 对应的文本文件 {txt_file}")

    if len(meta_lines) == 0:
        raise RuntimeError(f"VCTK 数据准备失败：在 {data_root} 下未找到任何有效音频-文本对")
    return meta_lines

def prepare_aishell3(data_root, dataset_prefix):
    wav_dir = data_root.parent / 'wavs'
    wav_dir.mkdir(exist_ok=True)
    meta_lines = []

    train_dir = data_root / 'train'
    if not train_dir.exists():
        train_dir = data_root
    content_file = train_dir / 'content.txt'
    if not content_file.exists():
        raise FileNotFoundError(f"无法找到 content.txt，请检查数据集结构：{train_dir}")

    with open(content_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue
            filename = parts[0]
            text = parts[1]
            wav_path = train_dir / 'wav' / f"{filename}.wav"
            if wav_path.exists():
                speaker_id = filename.split('/')[0]
                orig_wav_id = filename.replace('/', '_')
                wav_id = f"{dataset_prefix}_{orig_wav_id}"
                dst = wav_dir / f"{wav_id}.wav"
                shutil.copy(wav_path, dst)
                meta_lines.append(f"{wav_id}|{text}|{speaker_id}")
            else:
                print(f"警告：音频文件 {wav_path} 不存在，跳过")

    if len(meta_lines) == 0:
        raise RuntimeError(f"AISHELL-3 数据准备失败：在 {data_root} 下未找到任何有效音频-文本对")
    return meta_lines

# ---------- 主接口 ----------
def download_and_prepare(dataset_name, target_dir):
    target_dir = Path(target_dir)
    dataset_name = dataset_name.lower()
    urls = {
        'libritts': 'https://openslr.elda.org/resources/60/train-clean-100.tar.gz',
        'ljspeech': 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
        'vctk': 'https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip',
        'aishell3': 'https://www.openslr.org/resources/93/data_aishell3.tgz'
    }
    if dataset_name not in urls:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 先尝试查找已有有效数据
    data_root = find_data_root(target_dir, dataset_name)
    if data_root is not None:
        print(f"找到有效的数据根目录: {data_root}，跳过下载和解压。")
    else:
        # 需要下载
        url = urls[dataset_name]
        archive_name = url.split('/')[-1]
        archive_path = target_dir / archive_name
        if not archive_path.exists():
            print(f"Downloading {dataset_name}...")
            download_file(url, archive_path)
        print("Extracting...")
        extract_archive(archive_path, target_dir)
        print("Locating data root...")
        data_root = find_data_root(target_dir, dataset_name)
        if data_root is None:
            raise RuntimeError(f"无法找到 {dataset_name} 的有效数据根目录，解压可能失败或目录结构不匹配。")

    print(f"Data root found: {data_root}")
    print("Preparing metadata...")
    dataset_prefix = dataset_name
    if dataset_name == 'libritts':
        meta_lines = prepare_libritts(data_root, dataset_prefix)
    elif dataset_name == 'ljspeech':
        meta_lines = prepare_ljspeech(data_root, dataset_prefix)
    elif dataset_name == 'vctk':
        meta_lines = prepare_vctk(data_root, dataset_prefix)
    elif dataset_name == 'aishell3':
        meta_lines = prepare_aishell3(data_root, dataset_prefix)
    else:
        meta_lines = []

    meta_path = target_dir / f'metadata_{dataset_name}.txt'
    with open(meta_path, 'w', encoding='utf-8') as f:
        for line in meta_lines:
            f.write(line + '\n')
    print(f"Metadata saved to {meta_path}")
    wavs_dir = target_dir / 'wavs'
    return meta_path, wavs_dir