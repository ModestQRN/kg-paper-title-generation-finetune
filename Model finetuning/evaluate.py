import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import evaluate

# 基础模型路径
BASE_MODEL_PATH = "/home/ubuntu/dennis/AR_Bench/Llama-3.1-8B-Instruct"

# LoRA适配器路径
ADAPTER_PATH = "/home/ubuntu/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/train_2025-10-26-10-36-13/checkpoint-4221"

# 评估数据集的路径
TEST_DATASET_PATH = "/home/ubuntu/LLaMA-Factory/data/subset_004_percent.json"

# 结果保存路径
OUTPUT_FILE_PATH = "evaluation_results.jsonl"


# 配置模型加载参数
LOAD_IN_8BIT = False
LOAD_IN_4BIT = False

# 加载模型和分词器
print("=" * 20)
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    load_in_8bit=LOAD_IN_8BIT,
    load_in_4bit=LOAD_IN_4BIT,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

print("Model loaded successfully!")
print("=" * 20)

# 准备评估数据集
prompts = []
references = []
original_data_list = []

with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)
    for item in tqdm(dataset, desc="Preparing Prompts"):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        reference_output = item.get("output", "")

        # 构建提示词
        prompt_text = (
            "<s>[INST] <<SYS>>\n"
            "You are a helpful assistant. \n"
            "<</SYS>>\n\n"
            f"{instruction}\n{input_text} [/INST]"
        )

        prompts.append(prompt_text)
        references.append(reference_output)
        original_data_list.append(item)

print(f"Loaded and prepared {len(prompts)} samples.")
print("=" * 20)

# 运行推理生成预测
predictions = []
batch_size = 4  # 根据GPU显存调整
print(f"Starting generation with batch size: {batch_size}")

for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Outputs"):
    batch_prompts = prompts[i:i + batch_size]

    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id
        )

    input_lengths = inputs.input_ids.shape[1]
    batch_outputs = tokenizer.batch_decode(
        outputs[:, input_lengths:],
        skip_special_tokens=True
    )

    predictions.extend(batch_outputs)

# 清理显存
del model
torch.cuda.empty_cache()
print("Generation finished.")
print("=" * 20)

# 计算BERTScore
print("Calculating BERTScore...")
bertscore = evaluate.load('./bertscore')
local_bertscore_model_path = "/home/ubuntu/LLaMA-Factory/roberta-large"

bertscore_results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en",
    model_type="roberta-large"
)
# 计算平均分数
avg_bertscore = {
    "precision": sum(bertscore_results['precision']) / len(bertscore_results['precision']),
    "recall": sum(bertscore_results['recall']) / len(bertscore_results['recall']),
    "f1": sum(bertscore_results['f1']) / len(bertscore_results['f1']),
}

# 打印BERTScore结果
print("\n--- BERTScore Results ---")
print(f"Precision: {avg_bertscore['precision']:.4f}")
print(f"Recall: {avg_bertscore['recall']:.4f}")
print(f"F1 Score: {avg_bertscore['f1']:.4f}")
print("--------------------------\n")

# 保存详细结果
with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
    for i in range(len(prompts)):
        result_item = {
            "instruction": original_data_list[i]["instruction"],
            "input": original_data_list[i]["input"],
            "reference": references[i],
            "generated": predictions[i].strip(),
            "bertscore_precision": bertscore_results['precision'][i],
            "bertscore_recall": bertscore_results['recall'][i],
            "bertscore_f1": bertscore_results['f1'][i]
        }
        f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

print(f"Detailed results with BERTScore saved to {OUTPUT_FILE_PATH}")
