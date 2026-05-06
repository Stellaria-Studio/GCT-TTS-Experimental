import torch
import spacy
import numpy as np
from transformers import AutoTokenizer
import os
from .bert_utils import resolve_bert_path

class TextGraphFrontend:
    def __init__(self, language="en_core_web_sm", vocab_dict=None, use_bert=False,
                 bert_model="bert-base-multilingual-cased", bert_model_path=None):
        """
        文本前端：将文本转换为 token 序列和依存关系图。

        参数:
            language: spaCy 语言模型名称（仅在 use_bert=False 时使用）
            vocab_dict: 预定义词表（非BERT模式）
            use_bert: 是否使用 BERT tokenizer
            bert_model: BERT 模型名称或本地路径（仅在 use_bert=True 时有效）
            bert_model_path: 备用的 BERT 路径（优先级高于 bert_model）
        """
        self.use_bert = use_bert
        self.language = language
        self.vocab_dict = vocab_dict
        self.bert_model = bert_model
        self.bert_model_path = bert_model_path

        if use_bert:
            # ---------- BERT 模式 ----------
            raw_bert = bert_model_path if bert_model_path is not None else bert_model
            resolved_path = resolve_bert_path(raw_bert)
            print(f"[TextGraphFrontend] 解析后的 BERT 路径: {resolved_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
            self.vocab_size = self.tokenizer.vocab_size

            # 尝试加载spacy（可选，但不再强制用于图构建）
            try:
                self.nlp = spacy.load(language)
                print(f"[TextGraphFrontend] spaCy 模型 {language} 加载成功，将用于可选依存解析（需对齐处理）")
            except OSError:
                print(f"[TextGraphFrontend] spaCy 模型 {language} 未找到，将仅使用自环图。")
                self.nlp = None

            # 依存关系数量：如果 nlp 可用，预留所有标签；否则仅自环
            if self.nlp is not None:
                self.dep_labels = list(self.nlp.get_pipe("parser").labels) if hasattr(self.nlp, "get_pipe") else []
                self.dep2id = {"[SELF]": 0, "[UNK]": 1}
                for label in self.dep_labels:
                    if label not in self.dep2id:
                        self.dep2id[label] = len(self.dep2id)
                self.num_relations = len(self.dep2id)
            else:
                self.num_relations = 1   # 仅自环
                self.dep2id = {"[SELF]": 0}
        else:
            # ---------- 非 BERT 模式（使用 spaCy） ----------
            try:
                self.nlp = spacy.load(language)
            except OSError:
                print(f"请先下载语言模型: python -m spacy download {language}")
                raise

            self.dep_labels = list(self.nlp.get_pipe("parser").labels) if hasattr(self.nlp, "get_pipe") else []
            self.dep2id = {"[SELF]": 0, "[UNK]": 1}
            for label in self.dep_labels:
                if label not in self.dep2id:
                    self.dep2id[label] = len(self.dep2id)
            self.num_relations = len(self.dep2id)

            if vocab_dict is not None:
                self.vocab = vocab_dict
                self.vocab_size = len(vocab_dict)
                self.dynamic = False
            else:
                self.vocab = {"[PAD]": 0, "[UNK]": 1}
                self.vocab_size = 2
                self.dynamic = True

    def text_to_sequence_and_graph(self, text):
        """
        将文本转换为 token 序列和依存关系图。

        返回:
            seq_ids: torch.LongTensor, shape (L,) - token ID 序列
            adj_matrix: torch.FloatTensor, shape (L, L) - 邻接矩阵（无向）
            edge_type: torch.LongTensor, shape (L, L) - 边类型（0为自环，>0为依存类型）
        """
        if self.use_bert:
            # ---------- BERT 模式 ----------
            encoding = self.tokenizer(text, return_tensors='pt', add_special_tokens=True)
            seq_ids = encoding['input_ids'].squeeze(0)   # [L]
            L = seq_ids.size(0)

            # 若 nlp 可用，可尝试构建依存图（但需对齐 token，此处简化，仅返回自环）
            if self.nlp is not None:
                # 警告：BERT tokenizer 与 spaCy 分词结果可能不一致，因此不自动构建依存图
                # 若需要高级图，可在此实现对齐逻辑，当前版本仅使用自环。
                pass

            # 返回自环图（全1对角线）
            adj_matrix = torch.eye(L, dtype=torch.float32)
            edge_type = torch.zeros(L, L, dtype=torch.long)
            return seq_ids, adj_matrix, edge_type

        else:
            # ---------- 非 BERT 模式（spaCy 分词 + 依存解析） ----------
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            L = len(tokens)

            # 构建序列 ID（动态或固定词表）
            if self.dynamic:
                seq = []
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        self.vocab_size = len(self.vocab)
                    seq.append(self.vocab[token])
                seq_ids = torch.tensor(seq, dtype=torch.long)
            else:
                seq_ids = torch.zeros(L, dtype=torch.long)
                for i, token in enumerate(tokens):
                    seq_ids[i] = self.vocab.get(token, self.vocab["[UNK]"])

            # 初始化邻接矩阵和边类型（自环）
            adj_matrix = torch.eye(L, dtype=torch.float32)
            edge_type = torch.zeros(L, L, dtype=torch.long)

            # 遍历依存关系构建图
            for token in doc:
                i = token.i
                # 处理子节点
                for child in token.children:
                    j = child.i
                    dep = child.dep_
                    dep_id = self.dep2id.get(dep, self.dep2id["[UNK]"])
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
                    edge_type[i, j] = dep_id
                    edge_type[j, i] = dep_id
                # 处理头结点（已经包含在子节点关系中，但为确保全覆盖可保留）
                head = token.head
                if head.i != i:
                    j = head.i
                    dep = token.dep_
                    dep_id = self.dep2id.get(dep, self.dep2id["[UNK]"])
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
                    edge_type[i, j] = dep_id
                    edge_type[j, i] = dep_id

            return seq_ids, adj_matrix, edge_type