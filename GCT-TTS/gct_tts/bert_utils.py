import os
from pathlib import Path

def get_project_root():
    """自动定位项目根目录（包含 run_all.py 的目录）"""
    current_file = Path(__file__).resolve()  # bert_utils.py 的绝对路径
    for parent in current_file.parents:
        if (parent / "run_all.py").exists():
            return parent
    # 若找不到，回退到当前工作目录
    return Path.cwd()

def resolve_bert_path(bert_input):
    """
    智能解析 BERT 模型路径。
    1. 如果 bert_input 本身是一个存在的目录，返回其绝对路径。
    2. 否则，如果 bert_input 看起来像 Hugging Face 模型名（不包含路径分隔符且不是路径），直接返回原字符串。
    3. 否则在以下候选目录中查找（转换为绝对路径）：
       - 当前工作目录下的 bert_input
       - 当前工作目录下的 pretrained_models/bert_input
       - 项目根目录下的 bert_input
       - 项目根目录下的 pretrained_models/bert_input
    4. 如果都找不到，返回原字符串（由 from_pretrained 处理）。
    """
    # 如果已经是有效路径，直接返回绝对路径
    if os.path.exists(bert_input):
        return os.path.abspath(bert_input)

    # 如果输入不包含路径分隔符，且不是路径，可能是 Hugging Face 模型名，直接返回（跳过本地搜索）
    if '/' not in bert_input and '\\' not in bert_input and not bert_input.startswith('.'):
        # 但仍可先检查是否存在于预设的本地目录中
        root = get_project_root()
        cwd = Path.cwd()
        candidates = [
            cwd / bert_input,
            cwd / "pretrained_models" / bert_input,
            root / bert_input,
            root / "pretrained_models" / bert_input,
        ]
        for cand in candidates:
            abs_cand = str(cand.resolve())
            if os.path.exists(abs_cand):
                print(f"在本地找到 BERT 模型目录: {abs_cand}")
                return abs_cand
        # 未找到本地目录，返回原字符串（HuggingFace 加载）
        return bert_input

    root = get_project_root()
    cwd = Path.cwd()

    candidates = [
        cwd / bert_input,
        cwd / "pretrained_models" / bert_input,
        root / bert_input,
        root / "pretrained_models" / bert_input,
    ]

    for cand in candidates:
        abs_cand = str(cand.resolve())
        if os.path.exists(abs_cand):
            print(f"在本地找到 BERT 模型目录: {abs_cand}")
            return abs_cand

    print(f"未找到本地 BERT 模型，将使用 Hugging Face 加载 {bert_input}（如有缓存则使用缓存）")
    return bert_input