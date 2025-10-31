# src/knowledge_graph/kg_builder.py
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from neo4j import GraphDatabase
import os
# from src.model.entity_extractor.bert_bilstm_crf import EntityTripleExtractor # 假设路径设置正确
# 为了保证完整性，我们直接导入上面的类
from src.model.entity_extractor.bert_bilstm_crf import EntityTripleExtractor # 假设在同一个可导入目录下

logger = logging.getLogger(__name__)

NEO4J_CONFIG = {
        "uri": "neo4j://localhost:7687",
        "auth": ("neo4j", "password"),
        "database": "neo4j",
    }

# 2. 修正 bert_ner 路径，使用绝对路径
# 获取当前文件 (kg_builder.py) 的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设项目结构是 knowledge_graph/src/knowledge_graph/kg_builder.py
# 我们需要向上两级到达 knowledge_graph/，再进入 model/ner/ner_model
# 路径修正： ../../model/ner/ner_model
bert_ner_relative = os.path.join(current_dir, "..", "..", "src", "model", "ner_model")
bert_ner = os.path.normpath(bert_ner_relative).replace('\\', '/')
# bert_ner = 'bert-base-chinese'

# 打印最终路径，方便调试
logger.info(f"BERT Model Path set to: {bert_ner}")
class KnowledgeGraphBuilder:
    """
    知识图谱构建模块，负责将抽取的三元组插入Neo4j数据库
    """

    def __init__(self):
        """
        初始化知识图谱构建器
        """
        self.uri = NEO4J_CONFIG["uri"]
        self.auth = NEO4J_CONFIG["auth"]
        self.database = NEO4J_CONFIG["database"]

        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=self.auth,
            database=self.database
        )

        # 初始化实体抽取器
        self.extractor = EntityTripleExtractor(
            model_path = bert_ner,
            device = "cuda"
        )

    def close(self):
        """
        关闭数据库连接
        """
        self.driver.close()

    def build_graph_from_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从论文数据构建知识图谱

        Args:
            paper_data: 论文数据，包含文本、元数据和分类结果

        Returns:
            图谱构建结果统计
        """
        try:
            metadata = paper_data.get("metadata", {})
            text = paper_data.get("text", "")
            paper_framework = paper_data.get("framework", {})
            category = paper_data.get("category", "Unknown")

            # 创建论文节点
            paper_id = self._create_paper_node(metadata, category)

            # 抽取三元组
            triples = self.extractor.extract_triples(text)

            # 将三元组插入图谱
            stats = self._insert_triples(triples, paper_id)

            # 保存论文框架
            if paper_framework:
                self._save_paper_framework(paper_id, paper_framework)

            logger.info(f"论文 {metadata.get('title', 'Unknown')} 知识图谱构建完成")
            return stats
        except Exception as e:
            logger.error(f"构建知识图谱时出错: {e}")
            return {"error": str(e)}

    def add_paper(self, paper_data: Dict[str, Any]) -> str:
        # ... (省略)
        # 此函数依赖于其他辅助函数，确保其辅助函数已修复
        try:
            metadata = paper_data.get("metadata", {})
            category = paper_data.get("category", "未分类")
            framework = paper_data.get("framework", {})

            paper_id = self._create_paper_node(metadata, category)

            if framework:
                self._save_paper_framework(paper_id, framework)

            if "text" in paper_data:
                entities, relations = self._extract_entities_relations(paper_data["full_text"])
                self._add_entities_relations(paper_id, entities, relations)

            if "chunks" in paper_data:
                self._add_paper_chunks(paper_id, paper_data["chunks"])

            if "references" in paper_data and paper_data["references"]:
                self._add_paper_references(paper_id, paper_data["references"])

            logger.info(f"论文添加成功，ID: {paper_id}")

            return paper_id

        except Exception as e:
            logger.error(f"添加论文到知识图谱时出错: {e}")
            raise

    def _create_paper_node(self, metadata: Dict[str, str], category: str) -> str:
        # 修复：确保 RETURN 中使用 elementId(p)
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (p:Paper {title: $title}) 
                ON CREATE SET 
                    p.author = $author,
                    p.date = $date,
                    p.category = $category,
                    p.filename = $filename
                RETURN elementId(p) as paper_id
                """,
                title=metadata.get("title", "Unknown Title"),
                author=metadata.get("author", "Unknown Author"),
                date=metadata.get("creation_date", ""),
                category=category,
                filename=metadata.get("filename", "")
            )
            return result.single()["paper_id"]

    def _insert_triples(self, triples: List[Dict[str, str]], paper_id: str) -> Dict[str, int]:
        # 修复：确保所有实体和关系 ID 都使用 elementId
        stats = {
            "entities_created": 0,
            "relationships_created": 0
        }

        with self.driver.session() as session:
            for triple in triples:
                head = triple["head"]
                head_type = triple["head_type"]
                relation = triple["relation"]
                tail = triple["tail"]
                tail_type = triple["tail_type"]

                # 创建头实体 (使用 elementId(h))
                head_result = session.run(
                    f"""
                                    MERGE (h:{head_type} {{name: $name}})
                                    ON CREATE SET h.created_at = datetime()
                                    RETURN elementId(h) as entity_id, h.created_at as created_at
                                    """,
                    name=head
                )
                head_data = head_result.single()
                if head_data.get("created_at") is not None:
                    stats["entities_created"] += 1

                # 创建尾实体 (使用 elementId(t))
                tail_result = session.run(
                    f"""
                    MERGE (t:{tail_type} {{name: $name}})
                    ON CREATE SET t.created_at = datetime()
                    RETURN elementId(t) as entity_id, t.created_at as created_at # 修复 L200: id(t) -> elementId(t)
                    """,
                    name=tail
                )
                tail_data = tail_result.single()
                if tail_data.get("created_at") is not None:
                    stats["entities_created"] += 1

                # 创建三元组关系 (使用 elementId(r))
                rel_result = session.run(
                    f"""
                                    MATCH (h:{head_type} {{name: $head_name}})
                                    MATCH (t:{tail_type} {{name: $tail_name}})
                                    MERGE (h)-[r:{relation}]->(t)
                                    ON CREATE SET r.created_at = datetime()
                                    RETURN elementId(r) as rel_id, r.created_at as created_at
                                    """,
                    head_name=head,
                    tail_name=tail
                )
                rel_data = rel_result.single()
                if rel_data.get("created_at") is not None:
                    stats["relationships_created"] += 1

                # 将头实体与论文关联 (使用 elementId 匹配)
                session.run(
                    f"""
                                    MATCH (p:Paper) WHERE elementId(p) = $paper_id
                                    MATCH (e:{head_type} {{name: $entity_name}})
                                    MERGE (p)-[r:MENTIONS]->(e)
                                    """,
                    paper_id=paper_id,
                    entity_name=head
                )

                # 将尾实体与论文关联 (使用 elementId 匹配)
                tail_result = session.run(
                    f"""
                                    MERGE (t:{tail_type} {{name: $name}})
                                    ON CREATE SET t.created_at = datetime()
                                    RETURN elementId(t) as entity_id, t.created_at as created_at 
                                    """,
                    name=tail
                )

        return stats

    def _save_paper_framework(self, paper_id: str, framework: Dict[str, str]) -> None:
        # 修复：使用 elementId 匹配 Paper 节点
        with self.driver.session() as session:
            for section, content in framework.items():
                if content:
                    property_name = f"framework_{section.lower().replace(' ', '_')}"

                    query = f"""
                        MATCH (p:Paper) WHERE elementId(p) = $paper_id 
                        SET p.`{property_name}` = $content  
                    """

                    session.run(
                        query,
                        paper_id=paper_id,
                        content=content
                    )

    def query_entity(self, entity_name: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查询实体

        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）

        Returns:
            查询结果列表
        """
        with self.driver.session() as session:
            if entity_type:
                result = session.run(
                    f"""
                    MATCH (e:{entity_type})
                    WHERE e.name CONTAINS $name
                    RETURN e.name as name, labels(e) as types
                    """,
                    name=entity_name
                )
            else:
                result = session.run(
                    """
                    MATCH (e)
                    WHERE e.name CONTAINS $name AND NOT e:Paper
                    RETURN e.name as name, labels(e) as types
                    """,
                    name=entity_name
                )

            return [record.data() for record in result]

    def query_related_entities(self, entity_name: str,
                               entity_type: Optional[str] = None,
                               max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        查询与实体相关的其他实体

        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）
            max_depth: 最大关系深度

        Returns:
            相关实体列表
        """
        with self.driver.session() as session:
            if entity_type:
                query = f"""
                MATCH path = (e:{entity_type})-[*1..{max_depth}]-(related)
                WHERE e.name = $name AND NOT related:Paper
                RETURN related.name as name, labels(related) as types, 
                       [rel in relationships(path) | type(rel)] as relations,
                       length(path) as depth
                ORDER BY depth
                LIMIT 100
                """
            else:
                query = f"""
                MATCH path = (e)-[*1..{max_depth}]-(related)
                WHERE e.name = $name AND NOT e:Paper AND NOT related:Paper
                RETURN related.name as name, labels(related) as types, 
                       [rel in relationships(path) | type(rel)] as relations,
                       length(path) as depth
                ORDER BY depth
                LIMIT 100
                """

            result = session.run(query, name=entity_name)
            return [record.data() for record in result]

    def query_papers_by_entity(self, entity_name: str,
                               entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查询提到指定实体的论文

        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）

        Returns:
            论文列表
        """
        with self.driver.session() as session:
            if entity_type:
                query = f"""
                MATCH (p:Paper)-[:MENTIONS]->(e:{entity_type} {{name: $name}})
                RETURN p.title as title, p.author as author, p.category as category
                """
            else:
                query = """
                MATCH (p:Paper)-[:MENTIONS]->(e {name: $name})
                RETURN p.title as title, p.author as author, p.category as category
                """

            result = session.run(query, name=entity_name)
            return [record.data() for record in result]

    def query_papers_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        查询指定类别的论文

        Args:
            category: 论文类别

        Returns:
            论文列表
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper)
                WHERE p.category = $category
                RETURN p.title as title, p.author as author, p.category as category
                """,
                category=category
            )
            return [record.data() for record in result]

    def find_path_between_entities(self,
                                   entity1_name: str,
                                   entity2_name: str,
                                   max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        查找两个实体之间的路径

        Args:
            entity1_name: 第一个实体名称
            entity2_name: 第二个实体名称
            max_depth: 最大路径深度

        Returns:
            路径列表
        """
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (e1)-[*1..{max_depth}]-(e2)
                WHERE e1.name = $name1 AND e2.name = $name2
                RETURN [node in nodes(path) | node.name] as entities,
                       [rel in relationships(path) | type(rel)] as relations,
                       length(path) as length
                ORDER BY length
                LIMIT 10
                """,
                name1=entity1_name,
                name2=entity2_name
            )
            return [record.data() for record in result]

    def _extract_entities_relations(self, text: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        从文本中抽取实体和关系

        - 使用BERT-BiLSTM-CRF模型从文本中抽取实体和关系
        - 返回去重后的实体列表和完整的关系列表
        - 包含详细的错误处理

        Args:
            text: 输入文本

        Returns:
            (entities, relations) 实体列表和关系列表
        """
        try:
            # 使用实体抽取器获取三元组
            triples = self.extractor.extract_triples(text)

            # 提取实体
            entities = []
            seen_entities = set()
            for triple in triples:
                head = {"name": triple["head"], "type": triple["head_type"]}
                tail = {"name": triple["tail"], "type": triple["tail_type"]}

                if triple["head"] not in seen_entities:
                    entities.append(head)
                    seen_entities.add(triple["head"])

                if triple["tail"] not in seen_entities:
                    entities.append(tail)
                    seen_entities.add(triple["tail"])

            # 提取关系
            relations = [{
                "source": triple["head"],
                "target": triple["tail"],
                "type": triple["relation"],
                "source_type": triple["head_type"],
                "target_type": triple["tail_type"]
            } for triple in triples]

            return entities, relations

        except Exception as e:
            logger.error(f"抽取实体和关系时出错: {e}")
            return [], []

    def _add_paper_chunks(self, paper_id: str, chunks: List[Dict[str, Any]]) -> None:
        """
        添加论文文本块到知识图谱

        Args:
            paper_id: 论文ID
            chunks: 文本块列表
        """
        # 修复：使用 elementId 匹配 Paper 节点
        with self.driver.session() as session:
            for chunk in chunks:
                session.run(
                    """
                    MATCH (p:Paper) WHERE elementId(p) = $paper_id # 修复 L384: id(p) -> elementId(p)
                    MERGE (c:Chunk {text: $text, index: $index})
                    MERGE (p)-[:HAS_CHUNK]->(c)
                    """,
                    paper_id=paper_id,
                    text=chunk.get("text", ""),
                    index=chunk.get("index", 0)
                )

    def _add_paper_references(self, paper_id: str, references: List[Dict[str, str]]) -> None:
        # 修复：确保返回和匹配 ID 时都使用 elementId (ref_id 仍使用 id(r) 可能会导致警告，最好统一)
        with self.driver.session() as session:
            for ref in references:
                # 创建或匹配引用论文节点
                result = session.run(
                    """
                    MERGE (r:Paper {title: $title})
                    ON CREATE SET 
                        r.author = $author,
                        r.year = $year
                    RETURN elementId(r) as ref_id 
                    """,
                    title=ref.get("title", ""),
                    author=ref.get("author", ""),
                    year=ref.get("year", "")
                )
                # 此时 ref_id 是 elementId 字符串
                ref_id = result.single()["ref_id"]

                # 创建引用关系
                session.run(
                    """
                    MATCH (p:Paper) WHERE elementId(p) = $paper_id 
                    MATCH (r:Paper) WHERE elementId(r) = $ref_id
                    MERGE (p)-[:CITES]->(r)
                    """,
                    paper_id=paper_id,
                    ref_id=ref_id
                )

    def _add_entities_relations(self, paper_id: str, entities: List[Dict[str, str]],
                                relations: List[Dict[str, str]]) -> None:
        """
        将实体和关系添加到知识图谱 (此处只进行 MERGE，不返回 ID，所以只需修复匹配 Paper 节点的逻辑)
        """
        with self.driver.session() as session:
            # 添加实体节点
            for entity in entities:
                session.run(
                    f"""
                    MERGE (e:{entity['type']} {{name: $name}})
                    ON CREATE SET e.created_at = datetime()
                    """,
                    name=entity['name']
                )

                # 将实体与论文关联
                session.run(
                    f"""
                    MATCH (p:Paper) WHERE elementId(p) = $paper_id 
                    MATCH (e:{entity['type']} {{name: $name}})
                    MERGE (p)-[:MENTIONS]->(e)
                    """,
                    paper_id=paper_id,
                    name=entity['name']
                )

            # 添加关系
            for relation in relations:
                session.run(
                    f"""
                    MATCH (s:{relation['source_type']} {{name: $source}})
                    MATCH (t:{relation['target_type']} {{name: $target}})
                    MERGE (s)-[r:{relation['type']}]->(t)
                    ON CREATE SET r.created_at = datetime()
                    """,
                    source=relation['source'],
                    target=relation['target']
                )

    def get_paper_basic_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        # 修复：使用 elementId 匹配 Paper 节点
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Paper)
                    WHERE elementId(p) = $paper_id
                    RETURN p.title AS title, p.author AS author, 
                           p.category AS category, p.date AS date
                    """,
                    paper_id=paper_id
                )

                record = result.single()
                if record:
                    return {
                        "paper_id": paper_id,
                        "title": record["title"],
                        "author": record["author"],
                        "category": record["category"],
                        "date": record["date"]
                    }
                return None
        except Exception as e:
            logger.error(f"获取论文基本信息时出错: {e}")
            return None

    def find_similar_papers(self, paper_id: str, max_count: int = 5) -> List[Dict[str, Any]]:
        # 修复：使用 elementId 匹配和返回 Paper 节点
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p1:Paper)-[:MENTIONS]->(e)<-[:MENTIONS]-(p2:Paper)
                    WHERE elementId(p1) = $paper_id AND elementId(p2) <> $paper_id
                    WITH p2, count(e) AS common_entities
                    ORDER BY common_entities DESC
                    LIMIT $max_count
                    RETURN elementId(p2) AS paper_id, p2.title AS title,
                           p2.author AS author, p2.category AS category,
                           common_entities
                    """,
                    paper_id=paper_id,
                    max_count=max_count
                )
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"查找相似论文时出错: {e}")
            return []

    def get_latest_papers(self, count: int = 5) -> List[Dict[str, Any]]:
        # 修复：使用 elementId 返回 Paper 节点
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Paper)
                    RETURN elementId(p) AS paper_id, p.title AS title,
                           p.author AS author, p.category AS category,
                           p.date AS date
                    ORDER BY p.date DESC
                    LIMIT $count
                    """,
                    count=count
                )

                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"获取最新论文时出错: {e}")
            return []