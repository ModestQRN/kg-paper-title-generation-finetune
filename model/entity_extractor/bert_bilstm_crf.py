# bert_bilstm_crf.py
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer, BertConfig
from torchcrf import CRF
import logging
from typing import List, Dict, Tuple, Set, Optional, Any

logger = logging.getLogger(__name__)


class EntityTripleExtractor:
    """
    使用Bert+BiLSTM+CRF模型进行实体关系抽取
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化抽取器
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # 1. 实体标签必须先定义，才能计算 num_tags
        # 标签定义必须与您的训练数据标签一致
        self.entity_labels = [
            "[CLS]",  # 索引 0 (通常用于模型输入，但在解析时需跳过)
            "[SEP]",  # 索引 1 (通常用于模型输入，但在解析时需跳过)
            "O",  # 索引 2
            "B-MODEL", "I-MODEL",
            "B-METRIC", "I-METRIC",
            "B-DATASET", "I-DATASET",
            "B-METHOD", "I-METHOD",
            "B-TASK", "I-TASK",
            "B-FRAMEWORK", "I-FRAMEWORK"
        ]
        self.tag_to_index = {tag: i for i, tag in enumerate(self.entity_labels)}
        self.index_to_tag = {i: tag for i, tag in enumerate(self.entity_labels)}
        num_tags = len(self.entity_labels)

        # 2. 【核心修复 1】：加载分词器
        # 由于您的训练数据是英文的，强制使用英文 BERT 分词器，以防止本地路径加载到错误的（如中文）配置
        english_bert_name = 'bert-base-uncased'
        try:
            # 尝试从本地路径加载（如果路径中包含分词器文件），否则使用英文默认名
            tokenizer_path = model_path if os.path.exists(os.path.join(model_path, 'vocab.txt')) else english_bert_name
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Using tokenizer from: {self.tokenizer.name_or_path}")
        except Exception as e:
            # 如果加载失败，则退回到远程英文模型名称
            self.tokenizer = BertTokenizer.from_pretrained(english_bert_name)
            logger.warning(f"本地分词器加载失败，改用 {english_bert_name}。请确保模型权重与此匹配。")

        # 3. 只调用一次模型初始化，并正确传递 num_tags
        self.model = BERT_BiLSTM_CRF.from_pretrained(model_path, num_tags=num_tags)
        self.model.to(self.device)
        self.model.eval()

        # 4. 定义关系类型 (如果构建知识图谱需要)
        self.relationship_types: Set[Tuple[str, str, str]] = {
            ("MODEL", "EVALUATED_ON", "DATASET"),
            ("MODEL", "ACHIEVES", "METRIC"),
            ("METHOD", "APPLIES_TO", "TASK"),
            ("FRAMEWORK", "INCLUDES", "MODEL"),
            # 根据您的实际关系类型进行补充
        }

    def _get_entity_from_sequence(self, tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
        """
        根据 B-I-O 标签序列解析出实体
        """
        entities = []
        current_entity = None

        # 注意：这里的 tokens 和 tags 应该与模型输入序列对齐（通常跳过 [CLS] 和 [SEP]）
        # 但在 predict_entities 中我们处理了完整的序列。因此，这里处理的是 BERT tokens。

        # 跳过 [CLS] 和 [SEP] 标签对应的 token
        start_index = 1  # 跳过 [CLS]
        end_index = len(tokens) - 1  # 跳过 [SEP]

        # 确保标签序列长度大于等于 tokens 长度
        if len(tags) < len(tokens):
            logger.warning("标签序列长度与 Token 长度不匹配，可能导致解析错误。")
            end_index = len(tags)  # 以较短的为准

        for i in range(start_index, end_index):
            token = tokens[i]
            tag = tags[i]

            # 忽略 Sub-Word 之后的 ## 标记，将其合并
            if token.startswith("##"):
                if current_entity:
                    current_entity["text"] += token.replace("##", "")
                continue

            # 处理 B-I-O 标签
            tag_prefix, *tag_suffix = tag.split('-')
            entity_type = tag_suffix[0] if tag_suffix else None

            if tag_prefix == 'B':
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "type": entity_type,
                    "start": i,
                    "end": i
                }
            elif tag_prefix == 'I':
                if current_entity and current_entity["type"] == entity_type:
                    current_entity["text"] += token
                    current_entity["end"] = i
                else:
                    # I 标签没有对应的 B 标签，作为 O 标签处理
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = None
            elif tag_prefix == 'O':
                if current_entity:
                    entities.append(current_entity)
                current_entity = None

        # 处理最后一个实体
        if current_entity:
            entities.append(current_entity)

        return entities

    def predict_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中预测实体
        """
        # 1. 分词并转换为模型输入
        tokens = self.tokenizer.tokenize(text)
        # 为 BERT 输入添加 [CLS] 和 [SEP]
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 转换为 PyTorch 张量
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([[1] * len(input_ids)], dtype=torch.long).to(self.device)

        inputs = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask
        }

        # 2. 执行预测
        with torch.no_grad():
            # self.model(**inputs) 返回 List[List[int]] (批次的标签索引序列)
            predictions_batch = self.model(**inputs)

        # 3. 【核心修复 2】：修正 CRF 解码结果的解析
        predictions_sequence = []
        if predictions_batch and predictions_batch[0] is not None:
            # predictions_batch[0] 是批次中的第一个（也是唯一的）序列
            seq = predictions_batch[0]
            if isinstance(seq, list):
                predictions_sequence = seq
            elif isinstance(seq, torch.Tensor):
                predictions_sequence = seq.tolist()
            else:
                logger.error("模型预测结果格式不正确，预期为 List[List[int]] 或 List[Tensor]。")
                return []

        if not predictions_sequence:
            logger.error("模型预测结果为空。")
            return []

        # 4. 索引转标签
        predicted_tags = [self.index_to_tag.get(i, "O") for i in predictions_sequence]

        # 5. 从标签序列中提取实体
        entities = self._get_entity_from_sequence(tokens, predicted_tags)

        return entities

    def extract_triples(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取知识三元组（实体和关系）
        """
        # 1. 提取实体
        entities = self.predict_entities(text)

        logger.info(f"成功抽取的实体数量: {len(entities)}")
        if len(entities) > 0:
            logger.info(f"抽取的实体示例: {entities[:5]}")

        # 2. 关系抽取（此处简化为基于启发式规则或关系定义的匹配）
        # 实际关系抽取模型通常需要额外的模型（如 CasRel, OneIE, R-BERT 等）
        triples = []

        # 简化的关系匹配：遍历所有实体对，查找是否存在预定义的关系
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue

                entity1 = entities[i]
                entity2 = entities[j]

                # 检查是否存在预定义的关系
                # 关系类型为 (主体类型, 关系名, 客体类型)
                possible_relation = None

                # 示例启发式规则（您需要根据您的数据和训练目标来定义）
                if (entity1["type"], "EVALUATED_ON", entity2["type"]) in self.relationship_types:
                    possible_relation = "EVALUATED_ON"
                elif (entity1["type"], "ACHIEVES", entity2["type"]) in self.relationship_types:
                    possible_relation = "ACHIEVES"
                # ... 添加更多关系判断

                if possible_relation:
                    triples.append({
                        "head": entity1["text"],
                        "head_type": entity1["type"],
                        "relation": possible_relation,
                        "tail": entity2["text"],
                        "tail_type": entity2["type"],
                    })

        # 关系去重
        unique_triples = []
        triple_set = set()
        for triple in triples:
            t = (triple["head"], triple["relation"], triple["tail"])
            if t not in triple_set:
                triple_set.add(t)
                unique_triples.append(triple)

        logger.info(f"成功生成的三元组数量: {len(unique_triples)}")
        if len(unique_triples) > 0:
            logger.info(f"生成的三元组示例: {unique_triples[:5]}")
        return unique_triples


class BERT_BiLSTM_CRF(nn.Module):
    """
    BERT + BiLSTM + CRF 模型结构
    """

    def __init__(self, bert_model_path: str, num_tags: int, **kwargs):
        super(BERT_BiLSTM_CRF, self).__init__()

        # 1. 加载 BERT 模型
        self.bert = BertModel.from_pretrained(bert_model_path, **kwargs)
        self.bert_hidden_size = self.bert.config.hidden_size

        # 2. BiLSTM 层
        self.bilstm_hidden_size = 128 * 2   # 可调参数
        self.bilstm_num_layers = 2  # 可调参数
        self.bilstm = nn.LSTM(
            input_size=self.bert_hidden_size,
            hidden_size=self.bilstm_hidden_size,
            num_layers=self.bilstm_num_layers,
            bidirectional=True,
            batch_first=True
        )

        # 3. 线性层，将 BiLSTM 输出映射到标签空间
        # BiLSTM 输出维度是 2 * bilstm_hidden_size
        self.hidden2tag = nn.Linear(self.bilstm_hidden_size * 2, num_tags)

        # 4. CRF 层
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # 1. BERT 输出
        # output.last_hidden_state: (batch_size, seq_len, bert_hidden_size)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state

        # 2. BiLSTM 层
        # BiLSTM_output: (batch_size, seq_len, 2 * bilstm_hidden_size)
        bilstm_output, _ = self.bilstm(sequence_output)

        # 3. 线性层
        # logits: (batch_size, seq_len, num_tags)
        logits = self.hidden2tag(bilstm_output)

        # 4. CRF 损失或解码
        if labels is not None:
            # 训练模式: 返回负对数似然损失 (Neg Log Likelihood Loss)
            # mask: attention_mask 传入 CRF 作为 mask
            loss = self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
            # 返回负损失，因为 torch.optimizer 最小化损失
            return -loss
        else:
            # 预测模式: 返回最佳路径（标签索引序列）
            # decode 返回 List[List[int]]，其中包含批次中每个样本的最佳标签路径
            return self.crf.decode(logits, mask=attention_mask.bool())

    @classmethod
    def from_pretrained(cls, bert_model_path, *model_args, **kwargs):
        """
        从预训练模型加载模型权重
        """
        # 1. 弹出自定义模型的必需参数
        num_tags = kwargs.pop('num_tags')

        # 【修复 3: 清理不属于自定义模型 __init__ 的通用参数】
        # 确保只传递给 BERT 基模型的参数不会传递给 BERT_BiLSTM_CRF 的 __init__
        keys_to_remove = [
            'batch_first', 'use_cache', 'return_dict', 'output_attentions', 'output_hidden_states',
            'is_decoder', 'cross_attention_config', 'pruned_heads', 'tie_weights',
            'torchscript', 'output_loading_info', 'model_max_length', 'config'  # 移除 config
        ]

        safe_kwargs = kwargs.copy()
        for key in keys_to_remove:
            if key in safe_kwargs:
                safe_kwargs.pop(key)

        # 2. 初始化模型实例
        # 只传递经过清理的 safe_kwargs
        model = cls(bert_model_path, num_tags=num_tags, **safe_kwargs)

        # 3. 显式地加载自定义模型（BERT + BiLSTM + CRF）的权重
        custom_model_weights_path = os.path.join(bert_model_path, "pytorch_model.bin")

        if not os.path.exists(custom_model_weights_path):
            logger.warning(f"WARN: 未找到自定义模型权重文件: {custom_model_weights_path}。尝试继续加载。")

        try:
            state_dict = torch.load(custom_model_weights_path, map_location=torch.device('cpu'), weights_only= True)

            # 由于可能存在 key 名称不匹配（例如旧版本保存的权重），需要进行映射和过滤
            # 假设保存的权重是完整的 BERT_BiLSTM_CRF 模型的 state_dict
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"成功从 {custom_model_weights_path} 加载模型权重。")
        except Exception as e:
            logger.error(f"加载自定义模型权重失败: {e}。模型将使用随机或默认权重。")

        return model