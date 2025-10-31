import json
import logging
import re
from typing import Dict, List, Any
from zai import ZhipuAiClient

logger = logging.getLogger(__name__)

# 定义目标实体类型 (与知识图谱的节点类型保持一致)
ENTITY_TYPES = ["MODEL", "METHOD", "TASK", "METRIC", "DATASET"]

# 用于指导LLM输出的Prompt模板 (中文指令非常重要)
EXTRACTION_PROMPT_TEMPLATE = """
你是一名知识抽取专家，专门从学术论文中提取关键实体。
请严格按照要求，从提供的文本中抽取以下类型的实体，并以JSON格式返回。

实体类型说明：
1. MODEL (模型/架构)：如 BERT, Transformer, LLaMA-2, Qwen等。
2. METHOD (方法/算法)：如 Retrieval-Augmented Generation, DPO, Fine-tuning, LoRA等。
3. TASK (任务)：如 Question Answering, Sentiment Analysis, Data Scraping等。
4. METRIC (指标)：如 F1-score, Accuracy, BLEU, ROUGE等。
5. DATASET (数据集)：如 COCO, ImageNet, Piazza等。

要求：
1. 仅抽取文本中明确提到的实体。
2. 实体名称保持原样，不要修改或缩写（除非文本中本身就是缩写）。
3. 如果某种实体在文本中不存在，该类型的列表应为空 `[]`。
4. 严格只返回一个JSON对象，不要包含任何解释、markdown格式（例如：```json）或其他文字。

文本：
---
{text_to_extract}
---

请返回JSON结果：
"""


class LLMEntityExtractor:
    """
    使用 zai.ZhipuAiClient（GLM 模型）进行实体抽取的模块。
    """

    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化 ZhipuAiClient 客户端。
        Args:
            llm_config: 包含 "api_key" 和 "model_name" 的字典。
        """
        # 从配置中获取 Key 和模型名
        self.api_key = llm_config.get("api_key")
        # 【统一模型名】使用您测试可用的 glm-4.6
        self.model_name = llm_config.get("model_name", "glm-4.5v")

        if not self.api_key:
            logger.error("Zhipu AI API Key 未提供！无法调用 API。")
            raise ValueError("API Key is required for ZhipuAI client.")

        try:
            # 使用 ZhipuAiClient 初始化
            self.client = ZhipuAiClient(api_key=self.api_key)
            logger.info(f"LLMEntityExtractor 初始化成功，目标模型: {self.model_name}")
        except Exception as e:
            logger.error(f"初始化 ZhipuAiClient 客户端失败: {e}")
            raise

    def _call_llm_api(self, prompt: str, model_name: str, temperature: float) -> str:
        """
        调用 Zhipu AI API，并强制要求 JSON 格式输出。
        """
        try:
            # 调用 client.chat.completions.create
            response: Any = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                # 实体抽取需要稳定结果，使用较低温度
                temperature=temperature,
                # 【移除response_format】由于 zai 客户端可能不支持 response_format，先移除，依靠 prompt 引导 JSON 输出
                # response_format={"type": "json_object"}
            )

            # 返回模型的文本输出
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Zhipu AI API 调用失败: {e}")
            return json.dumps({etype: [] for etype in ENTITY_TYPES})

    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        从单个文本块中抽取实体，包含输出清洗和错误处理。
        """
        text_to_extract = text[:8000]

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text_to_extract=text_to_extract)

        try:
            raw_response = self._call_llm_api(prompt,"glm-4.6", 0.3 )

            # 1. 输出清洗：移除可能残留的```json```格式标签
            cleaned_response = raw_response.strip()
            if not cleaned_response.startswith('{'):
                cleaned_response = re.sub(r"^\s*`{3}(json|JSON)?\s*", "", cleaned_response, flags=re.IGNORECASE)
                cleaned_response = re.sub(r"\s*`{3}\s*$", "", cleaned_response)

            # 2. JSON解析 (准确性保障)
            entities: Dict[str, List[str]] = json.loads(cleaned_response)

            # 3. 结果校验
            valid_entities = {}
            for etype in ENTITY_TYPES:
                if etype in entities and isinstance(entities[etype], list):
                    valid_entities[etype] = list(set(e.strip() for e in entities[etype] if e and isinstance(e, str)))
                else:
                    valid_entities[etype] = []

            return valid_entities

        except json.JSONDecodeError as e:
            logger.error(f"LLM响应JSON格式错误: {e}. 响应内容: {raw_response[:200]}...")
            return {etype: [] for etype in ENTITY_TYPES}

        except Exception as e:
            logger.error(f"LLM实体抽取过程中发生未知错误: {e}")
            return {etype: [] for etype in ENTITY_TYPES}

    def generate_text(self, prompt: str, temperature: float = 0.5) -> str:
        """
        从LLM生成一段文本答案。
        """
        try:
            # 调用底层 API
            raw_response = self._call_llm_api(prompt, self.model_name, temperature)

            # **文本生成不需要JSON解析和复杂校验，只需简单的Markdown清理**
            cleaned_response = raw_response.strip()

            # 移除可能存在的 Markdown 格式标签
            if cleaned_response.startswith('```'):
                cleaned_response = re.sub(r"^\s*`{3}(text|TEXT)?\s*", "", cleaned_response, flags=re.IGNORECASE)
                cleaned_response = re.sub(r"\s*`{3}\s*$", "", cleaned_response)

            return cleaned_response.strip()

        except Exception as e:
            # 遵循实体抽取中的错误处理方式：记录并抛出或返回默认值
            logger.error(f"LLM 文本生成失败: {e}")
            # 在 data_preprocessor 中会捕获此错误并返回友好的默认答案
            raise

    # ----------------------------------------------------
    # 新增核心功能 3: 列表生成 (用于 FAQ 相似问法生成)
    # ----------------------------------------------------
    def generate_list(self, prompt: str, temperature: float = 0.7) -> List[str]:
        """
        从LLM生成一个字符串列表 (用于相似问法)。

        注意：此方法会强制要求LLM以JSON数组格式返回，并执行实体抽取中的JSON清洗和解析流程。
        """
        # 完整的Prompt应该在 data_preprocessor 中构建，这里只执行调用和解析
        try:
            # 调用底层 API
            raw_response = self._call_llm_api(prompt, self.model_name, temperature)

            # 1. 输出清洗：移除可能残留的```json```格式标签 (与抽取逻辑一致)
            cleaned_response = raw_response.strip()

            # 1. 尝试移除 Markdown 格式（如 ```json ... ```）
            # 这一步可以保留，但需确保能处理多行内容 (re.DOTALL)
            cleaned_response = re.sub(r"^\\s*`{3}(json|JSON)?\\s*", "", cleaned_response,
                                      flags=re.IGNORECASE | re.DOTALL)
            cleaned_response = re.sub(r"\\s*`{3}\\s*$", "", cleaned_response)
            cleaned_response = cleaned_response.strip()

            # 2. 查找并提取第一个完整的 JSON 数组 ([ ... ])
            start_index = cleaned_response.find('[')
            end_index = cleaned_response.rfind(']')

            if start_index != -1 and end_index != -1 and end_index > start_index:
                # 提取 [ 和 ] 之间的内容
                json_content = cleaned_response[start_index: end_index + 1]
                # 尝试解析
                result_list: List[Any] = json.loads(json_content)
            else:
                # 如果没有找到有效的 [ ... ] 结构，记录警告并返回空列表
                logger.warning(f"无法在响应中找到有效的JSON数组结构: {cleaned_response[:100]}...")
                return []

            # 3. 结果校验：确保是列表并进行清洗
            if isinstance(result_list, list):
                # 过滤空字符串，并去除首尾空格，确保元素是字符串
                return [item.strip() for item in result_list if item and isinstance(item, str)]
            else:
                logger.warning(f"LLM 列表生成返回的不是列表 (类型: {type(result_list)})。")
                return []

        except json.JSONDecodeError as e:
            # 遵循实体抽取中的错误处理
            logger.error(f"LLM 列表生成返回的JSON格式错误: {e}. 原始响应: {raw_response[:100]}")
            return []
        except Exception as e:
            # 遵循实体抽取中的意外错误处理
            logger.error(f"LLM 列表生成发生意外错误: {e}")
            raise  # 向上抛出异常，让上层函数处理错误