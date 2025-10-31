# main_train.py (仅修改 NERTrainer 初始化部分)
import os
import json
import torch
import random
import logging
from sklearn.model_selection import train_test_split

# 确保能正确导入模型和训练器
try:
    from bert_bilstm_crf import BERT_BiLSTM_CRF
    from ner_trainer import NERDataset, NERTrainer 
except ImportError as e:
    # 如果导入失败，可能是文件路径问题，提示用户检查
    print(f"导入模型模块失败，请确保 bert_bilstm_crf.py 和 ner_trainer.py 在当前目录下：{e}")
    exit()

from transformers import BertTokenizerFast # 保持 BertTokenizer 导入

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 训练配置 ---
class TrainingConfig:
    # 路径配置
    DATA_PATH = "spacy_ner_training_data_annotated_final.json"
    OUTPUT_DIR = "./output_ner_model"  # 模型保存目录
    
    # BERT模型配置：请根据您的Autodl环境和需求选择合适的预训练模型
    # 推荐使用中文预训练模型，如 'hfl/chinese-bert-wwm-ext'
    BERT_MODEL_PATH = "bert-base-chinese" 
    
    # 训练参数 (可根据您的 Autodl 显存和性能进行调整)
    MAX_LENGTH = 128            # 序列最大长度
    LSTM_HIDDEN_DIM = 256       # BiLSTM隐藏层维度
    TRAIN_BATCH_SIZE = 32       # 训练批次大小
    EVAL_BATCH_SIZE = 64        # 评估批次大小
    LEARNING_RATE = 2e-5        # 学习率
    NUM_EPOCHS = 10             # 训练轮次
    WARMUP_RATIO = 0.1          # 学习率预热比例
    WEIGHT_DECAY = 0.01
    
    # 数据划分
    TEST_SIZE = 0.1             # 10%用于验证集
    RANDOM_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_path):
    """加载JSON格式的NER数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 您的数据已经是列表格式，直接加载
        data = json.load(f)
    return data

def train_ner_model():
    """NER模型训练主函数"""
    config = TrainingConfig()
    
    # 设置随机种子
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    logger.info(f"设备: {config.DEVICE}")
    logger.info(f"加载预训练BERT模型: {config.BERT_MODEL_PATH}")

    # 1. 加载数据
    all_data = load_data(config.DATA_PATH)
    logger.info(f"原始数据条数: {len(all_data)}")
    
    # 2. 划分训练集和验证集
    train_data, eval_data = train_test_split(
        all_data, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED
    )
    logger.info(f"训练集条数: {len(train_data)}, 验证集条数: {len(eval_data)}")

    # 3. 初始化分词器、数据集
    # 注意：这里使用 BertTokenizerFast，但在 ner_trainer 中我们使用了 AutoTokenizer，
    # 实际运行时 AutoTokenizer 会自动识别并加载 BertTokenizerFast。
    tokenizer = BertTokenizerFast.from_pretrained(config.BERT_MODEL_PATH)
    
    # 初始化数据集，数据处理（字符对齐）将在这一步完成
    train_dataset = NERDataset(train_data, tokenizer, config.MAX_LENGTH)
    eval_dataset = NERDataset(eval_data, tokenizer, config.MAX_LENGTH)
    
    # 从数据集获取标签数量
    num_tags = train_dataset.num_tags 
    logger.info(f"实体标签总数 (包括 B/I/O/[CLS]/[SEP]): {num_tags}")

    # 4. 初始化模型
    model = BERT_BiLSTM_CRF.from_pretrained(
        config.BERT_MODEL_PATH, 
        num_tags=num_tags,
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM
    ).to(config.DEVICE)

    # 5. 初始化训练器 (新增 output_dir 参数)
    trainer = NERTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        device=config.DEVICE,
        learning_rate=config.LEARNING_RATE,
        train_batch_size=config.TRAIN_BATCH_SIZE,
        eval_batch_size=config.EVAL_BATCH_SIZE,
        num_epochs=config.NUM_EPOCHS,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        output_dir=config.OUTPUT_DIR # 传递保存路径
    )

    # 6. 启动训练
    logger.info("开始模型训练...")
    trainer.train()
    logger.info("模型训练完成。")

    # 7. 保存模型
    trainer.save_model(config.OUTPUT_DIR)
    logger.info(f"模型和分词器已保存到: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    train_ner_model()