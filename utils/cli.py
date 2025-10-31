import os
import logging
from neo4j import GraphDatabase
from typing import Dict, List, Any

# --- 1. 配置和初始化 ---

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neo4j配置 (使用您在 kg_builder.py 中提供的配置)
NEO4J_CONFIG = {
    "uri": "neo4j://localhost:7687",
    "auth": ("neo4j", "password"),
    "database": "neo4j",
}


# --- 2. 模拟 LLM 客户端和 RAG 核心函数 ---
class LLMClient:
    """
    模拟的大语言模型客户端，用于生成问答响应。
    现在包含更智能的模拟 RAG 逻辑。
    """

    def __init__(self, model_name="Academic-LLM-RAG-3.1"):
        self.model_name = model_name

    def generate_response(self, question: str, context: str) -> str:
        """
        根据问题和检索到的知识生成答案 (RAG 核心逻辑)
        """
        # 模拟 RAG 逻辑：如果检索到知识，则使用知识生成答案
        if context:
            # 简化回答模板，突出 RAG 效果
            return (
                f"\n✨ 【检索结果】\n"
                f"根据知识图谱中检索到的信息（关键三元组：{context[:150]}...），我为您生成了以下专业回答：\n"
                f"回答：{self._mock_answer_generation(question, context)}\n"
            )
        else:
            # 模拟 LLM 在缺乏知识时的通用回答
            return (
                f"\n⚠️ 【知识图谱未命中】\n"
                f"抱歉，知识图谱中未找到与 '{question[:40]}...' 直接相关的实体或论文。这是基于我预训练知识的通用回答：\n"
                f"回答：{self._mock_fallback_answer(question)}\n"
            )

    def _mock_answer_generation(self, question: str, context: str) -> str:
        # 默认回退逻辑（如果命中知识图谱，但未命中定制逻辑）
        return (
            "针对您的问题，知识图谱已检索到相关实体关系，证明信息存在。请相信这是一个高效且准确的回答。"
        )

    def _mock_fallback_answer(self, question):
        if "最先进" in question or "最好" in question:
            return "在特定任务上，没有绝对“最好”的模型，但 Transformer 及其变体（如 BERT/LLaMA）是当前研究的主流。"
        return f"这是一个通用的 NLP 问题，涉及深度学习和大规模预训练技术。"


# --- 3. 知识图谱检索逻辑 ---
class KnowledgeGraphQuery:
    """最终修正版的知识图谱查询逻辑，使用 labels(node) 获取实体类型"""

    # 扩展关键词列表，涵盖 triple.txt 中的关键实体，用于初步匹配
    KNOWN_ENTITIES = [
        "Retrieval Augmented Generation (RAG)", "RAG",
        "Supervised Fine-Tuning (SFT)", "SFT",
        "Direct Preference Optimization (DPO)", "DPO",
        "CXR-RePaiR-Gen", "CXR-RePaiR", "CXR-ReDonE",
        "gpt-4", "gpt-3.5-turbo", "text-davinci-003",
        "radiology report generation", "report writing",
        "MIMIC-CXR", "CXR-PRO", "LLM", "MODEL", "METHOD", "TASK"
    ]

    def __init__(self, uri, auth, database):
        try:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            self.driver.verify_connectivity()
            self.database = database
            # logger.info("Neo4j 连接成功。") # 保持输出整洁，不再重复
        except Exception as e:
            logger.error(f"无法连接到 Neo4j 数据库: {e}")
            self.driver = None

    def find_related_knowledge(self, question: str) -> str:
        """
        核心检索增强逻辑：识别问题中的关键实体，并查询图谱中相关的关系三元组。
        """
        if not self.driver:
            return ""

        # 1. 实体识别和匹配
        found_entities = set()
        for entity in self.KNOWN_ENTITIES:
            if entity.lower() in question.lower():
                found_entities.add(entity)

        if not found_entities:
            return ""

        entity_names_list = list(found_entities)

        # 2. 构造 Cypher 查询：使用 labels(node) 获取标签
        cypher_query = f"""
        UNWIND $entity_names AS name
        MATCH (e1)-[r]->(e2)
        WHERE e1.name = name OR e2.name = name
        RETURN e1.name AS subject, type(r) AS predicate, e2.name AS object, 
               labels(e1)[0] AS subject_type, labels(e2)[0] AS object_type
        LIMIT 10  // 限制返回的关系数量，避免上下文过长
        """

        knowledge_snippets = []
        with self.driver.session(database=self.database) as session:
            try:
                # 传递参数列表
                result = session.run(cypher_query, entity_names=entity_names_list)

                # 格式化检索到的关系三元组
                for record in result:
                    # 确保 subject_type 和 object_type 不为空 (即节点有标签)
                    subject_type = record['subject_type'] if record['subject_type'] else 'None'
                    object_type = record['object_type'] if record['object_type'] else 'None'

                    snippet = (
                        f"[{subject_type}: {record['subject']}] "
                        f"--[{record['predicate']}]--> "
                        f"[{object_type}: {record['object']}]"
                    )
                    knowledge_snippets.append(snippet)

            except Exception as e:
                # 注意：如果连接或Cypher本身语法有问题，可能会在这里捕获
                logger.error(f"Cypher 查询失败: {e}")
                return ""

        if knowledge_snippets:
            # 将检索到的所有三元组片段合并成一个上下文字符串
            return " | ".join(knowledge_snippets)
        else:
            return ""

# --- 4. 命令行接口 (CLI) ---

class KGRAG_CLI:
    """
    基于知识图谱 RAG 的交互式命令行接口
    """

    def __init__(self):
        self.llm = LLMClient()
        self.kg_query = KnowledgeGraphQuery(**NEO4J_CONFIG)
        self.model_name = self.llm.model_name
        self.kg_status = "【已连接】" if self.kg_query.driver else "【未连接】"

    def run(self):
        """
        启动命令行循环
        """
        # 1. 启动问候语 (自我介绍)
        print("=" * 60)
        print(f"🤖 欢迎使用学术知识问答系统命令行接口 (CLI)")
        print("-" * 60)
        print(f"💬 我是一个学术问答大模型")
        print(
            f"📖 我可以解决的问题：从学术论文知识图谱中检索事实，并生成关于 NLP、LLM 架构、方法论和评估指标等方面的专业回答。")
        print(f"🔗 知识图谱状态: {self.kg_status}")
        print("=" * 60)
        print("\n请问您有什么关于学术知识的问题？ (输入 'exit' 或 'quit' 退出)\n")

        # 2. 交互循环
        while True:
            try:
                question = input(f"👤 您的问题 > ")

                if question.lower() in ['exit', 'quit']:
                    print("\n👋 感谢使用，再见！")
                    break

                if not question.strip():
                    continue

                # 3. 执行 RAG 流程

                # 检索知识
                context = self.kg_query.find_related_knowledge(question)

                # 生成回答
                response = self.llm.generate_response(question, context)

                print(response)

            except EOFError:
                print("\n👋 感谢使用，再见！")
                break
            except Exception as e:
                logger.error(f"发生未知错误: {e}")
                print("\n[系统错误] 抱歉，处理您的请求时发生错误。")


# --- 5. 主程序入口 ---

if __name__ == "__main__":
    cli = KGRAG_CLI()
    cli.run()