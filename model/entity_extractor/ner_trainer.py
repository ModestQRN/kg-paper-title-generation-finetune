# ner_trainer.py (增加评估和绘图逻辑)
import os
import logging
import torch
import numpy as np # 新增导入
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer 
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from typing import Dict, List, Any, Optional, Tuple

# 新增导入用于评估和绘图
from seqeval.metrics import classification_report # 用于计算NER指标
import matplotlib.pyplot as plt # 用于绘图

# 假设 bert_bilstm_crf.py 在同一目录
from bert_bilstm_crf import BERT_BiLSTM_CRF 

logger = logging.getLogger(__name__)

class NERDataset(Dataset):
    """
    NER数据集类，处理 Spacy/JSON 格式数据并转换为 BERT + BIO 格式
    ... (保持原 NERDataset 类的代码不变)
    """
    # 将 tokenizer 的类型提示改为 AutoTokenizer
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 128):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 标签映射：必须与 bert_bilstm_crf.py 中 BERT_BiLSTM_CRF 的 num_tags 保持一致
        # 注意：[CLS] 和 [SEP] 必须有自己的标签ID
        self.label_map = {
            # 特殊标签
            "[CLS]": 0, "[SEP]": 1, "O": 2, 
            # 实体标签
            "B-MODEL": 3, "I-MODEL": 4,
            "B-METRIC": 5, "I-METRIC": 6,
            "B-DATASET": 7, "I-DATASET": 8,
            "B-METHOD": 9, "I-METHOD": 10,
            "B-TASK": 11, "I-TASK": 12,
            "B-FRAMEWORK": 13, "I-FRAMEWORK": 14
        }
        self.num_tags = len(self.label_map)
        
        # 反向映射，用于解码CRF输出 (新增)
        self.id_to_label = {v: k for k, v in self.label_map.items()} # 新增
        
        # 处理原始数据
        self.data = self._process_data(data)

    def _convert_spacy_to_bio(self, text: str, entities: List[List[Any]]) -> List[str]:
        """
        核心数据处理逻辑：将字符偏移量标注转换为词元级别的 BIO 标签序列。
        """
        
        # 1. 使用分词器获取词元序列及其在原始文本中的字符偏移量
        encoded = self.tokenizer.encode_plus(
            text, 
            truncation=True, 
            max_length=self.max_length,
            return_offsets_mapping=True, 
            add_special_tokens=True # 包含 [CLS], [SEP]
        )
        
        offset_mapping = encoded["offset_mapping"] 
        input_ids = encoded["input_ids"]
        
        # 初始化所有标签为 'O'，长度为 input_ids 的实际长度（包括 [CLS], [SEP]）
        bio_labels = ["O"] * len(input_ids)
        
        # 2. 遍历所有实体标注
        for char_start, char_end, label_type in entities:
            
            # 3. 遍历每个词元的偏移量进行对齐
            for token_idx, (token_char_start, token_char_end) in enumerate(offset_mapping):
                
                # 跳过特殊词元 ([CLS]和[SEP]的偏移量通常是(0, 0))
                if token_char_start == 0 and token_char_end == 0:
                    # 将特殊词元的标签设置为对应的特殊标签
                    if token_idx == 0 and self.tokenizer.cls_token_id == input_ids[token_idx]:
                        bio_labels[token_idx] = "[CLS]"
                    elif self.tokenizer.sep_token_id == input_ids[token_idx] or token_idx == len(input_ids) - 1:
                         bio_labels[token_idx] = "[SEP]"
                    continue
                
                # 检查词元是否与实体重叠
                
                # 情况 A: 词元的起始点落在实体范围内 -> B 或 I
                if char_start <= token_char_start < char_end:
                    # 如果词元起点等于实体起点，则是 B-
                    if token_char_start == char_start:
                        bio_labels[token_idx] = f"B-{label_type}"
                    # 如果词元起点晚于实体起点，则是 I-
                    else:
                        bio_labels[token_idx] = f"I-{label_type}"

                # 情况 B: 词元完全包含在实体中，且不是实体开头 -> I
                elif char_start < token_char_start and token_char_end <= char_end:
                    # 确保前一个词元是 B 或 I，避免断开的 I (这个逻辑在 B- 处已经处理，这里可简化)
                    if bio_labels[token_idx] == "O": # 只有当它还未被标记时才更新
                        bio_labels[token_idx] = f"I-{label_type}"

        # 确保标签序列长度不超过 max_length
        return bio_labels[:self.max_length]


    def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理数据，生成 token_ids, attention_mask 和 label_ids
        """
        processed_data = []
        for item in tqdm(data, desc="Processing data"):
            text = item["text"]
            entities = item["annotations"].get("entities", [])
            
            # 1. 字符偏移量 -> BIO 标签序列
            bio_labels = self._convert_spacy_to_bio(text, entities)
            
            # 2. 分词器编码 (再次编码以获取 padding 和 mask)
            encoded = self.tokenizer.encode_plus(
                text, 
                truncation=True, 
                max_length=self.max_length,
                padding="max_length",
                return_tensors='pt',
                return_attention_mask=True
            )
            
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            
            # 3. 将 BIO 标签转换为 ID 并进行 padding
            label_ids = [self.label_map.get(label, self.label_map["O"]) for label in bio_labels]
            
            # 填充标签序列到最大长度
            padding_length = self.max_length - len(label_ids)
            if padding_length > 0:
                # 使用 'O' 的 ID 来填充剩余部分
                label_ids += [self.label_map["O"]] * padding_length
            
            # 确保标签长度一致
            label_ids = torch.tensor(label_ids[:self.max_length], dtype=torch.long)
            
            processed_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label_ids
            })
            
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- NERTrainer 类 (完整定义 - 增加评估和绘图) ---
class NERTrainer:
    """
    NER 模型训练器
    """
    
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, device, 
                 learning_rate, train_batch_size, eval_batch_size, 
                 num_epochs, warmup_ratio, weight_decay, output_dir: str = "./output_ner_model"): # 新增 output_dir
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.output_dir = output_dir # 新增
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)
        
        # 优化器和调度器
        t_total = len(self.train_dataloader) * num_epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(t_total * warmup_ratio), num_training_steps=t_total
        )
        
        # 损失记录列表 (新增)
        self.train_losses = []
        self.eval_losses = []
        
    def train(self):
        self.model.train()
        global_step = 0
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            for step, batch in tqdm(enumerate(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.num_epochs}", total=len(self.train_dataloader)):
                
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                loss = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                
                total_loss += loss.item()
                global_step += 1
                
                if global_step % 50 == 0:
                    avg_loss = total_loss / (step + 1)
                    logger.info(f"Step {global_step} - Train Loss: {avg_loss:.4f}")
            
            avg_train_loss = total_loss / len(self.train_dataloader)
            self.train_losses.append(avg_train_loss) # 记录训练损失 (新增)
            
            avg_eval_loss, metrics = self._evaluate() # 修改：同时返回损失和指标
            self.eval_losses.append(avg_eval_loss) # 记录验证损失 (新增)
            
            logger.info(f"Epoch {epoch+1} Complete - Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}")
            logger.info("\n" + metrics) # 打印评估指标 (新增)
            
        # 训练完成后，绘制损失曲线并保存 (新增)
        self._plot_loss()
        logger.info(f"损失函数图像已保存到: {os.path.join(self.output_dir, 'loss_curve.png')}")

    def _evaluate(self, dataloader: Optional[DataLoader] = None) -> Tuple[float, str]: # 修改返回类型
        """
        评估模型并计算NER指标 (P, R, F1)
        """
        self.model.eval()
        eval_loss = 0
        all_preds = [] # 存储所有预测标签
        all_labels = [] # 存储所有真实标签
        dataloader = dataloader or self.eval_dataloader
        
        # 获取ID到Label的映射，用于解码
        id_to_label = self.train_dataset.id_to_label 
        # 过滤掉特殊标签和 'O'，只保留实体标签用于评估 (seqeval会自动处理B/I)
        label_list = [id_to_label[i] for i in sorted(id_to_label.keys()) if i > 2 and id_to_label[i] != 'O']
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"): # 增加进度条
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 计算损失
                loss = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                eval_loss += loss.item()

                # 获取预测结果
                # decode返回的是一个list of list of ints
                predictions = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None # 不计算损失，只进行解码
                )

                # 将预测和真实标签转换为BIO字符串格式 (排除特殊token和padding)
                for i in range(len(batch["labels"])): # 遍历batch中的每个样本
                    true_labels = []
                    predicted_labels = []
                    
                    # 长度取决于attention_mask中为1的部分 (即非padding部分)
                    for token_idx in range(batch["labels"].size(1)):
                        if batch["attention_mask"][i][token_idx].item() == 1:
                            true_id = batch["labels"][i][token_idx].item()
                            pred_id = predictions[i][token_idx]
                            
                            true_label = id_to_label[true_id]
                            pred_label = id_to_label[pred_id]
                            
                            # 过滤掉特殊标签 [CLS], [SEP]
                            if true_label not in ["[CLS]", "[SEP]"]:
                                # seqeval要求输入是 list of list of strings
                                true_labels.append(true_label)
                                predicted_labels.append(pred_label)
                    
                    # 避免空序列加入
                    if true_labels:
                        all_labels.append(true_labels)
                        all_preds.append(predicted_labels)
        
        avg_eval_loss = eval_loss / len(dataloader)
        self.model.train()
        
        # 计算seqeval指标
        # 过滤掉O标签，因为我们只需要评估实体 (B/I)
        report = classification_report(all_labels, all_preds, digits=4)
        
        return avg_eval_loss, report
    
    def _plot_loss(self):
        """
        绘制训练和验证损失曲线并保存
        """
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_epochs + 1), self.train_losses, label='Training Loss', marker='o')
        plt.plot(range(1, self.num_epochs + 1), self.eval_losses, label='Validation Loss', marker='x')
        
        plt.title('Training and Validation Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        plt.savefig(os.path.join(self.output_dir, 'loss_curve.png'))
        plt.close() # 关闭图形，释放内存

    def save_model(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        self.model.bert.config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"模型和分词器已保存到: {save_path}")