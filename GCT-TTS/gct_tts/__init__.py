# gct_tts/__init__.py

from .model import GCTTTS
from .frontend import TextGraphFrontend
from .dataset import GCTDataset, gct_collate_fn
from .dataset import MultiDataset   # 假设 MultiDataset 定义在 dataset.py 中

__all__ =[
    "GCTTTS",
    "TextGraphFrontend",
    "GCTDataset",
    "gct_collate_fn",
    "MultiDataset"
]