import requests
import xml.etree.ElementTree as ET
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置 ---
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
# 假设您的项目结构，数据将保存到 src/data/src/raw
RAW_DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILENAME = "arxiv_metadata.jsonl"
PDF_BASE_URL = "https://arxiv.org/pdf/"

# 确保输出目录存在
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- 核心函数 (parse_arxiv_xml 保持不变) ---
def parse_arxiv_xml(xml_content: str) -> List[Dict[str, Any]]:
    """
    解析arXiv API返回的Atom XML格式内容，提取元数据。

    Args:
        xml_content: arXiv API响应的XML字符串。

    Returns:
        包含论文元数据的字典列表。
    """
    # 定义Atom XML命名空间
    NAMESPACE = {'atom': 'http://www.w3.org/2005/Atom',
                 'arxiv': 'http://arxiv.org/schemas/atom'}

    root = ET.fromstring(xml_content)
    papers = []

    # 遍历所有 <entry> 标签，每个标签代表一篇论文
    for entry in root.findall('atom:entry', NAMESPACE):
        try:
            # 提取 arXiv ID。ID通常在 <id> 标签中，需要处理URL
            id_tag = entry.find('atom:id', NAMESPACE)
            arxiv_id = id_tag.text.split('/')[-1] if id_tag is not None else None

            # 提取主要元数据
            title = entry.find('atom:title', NAMESPACE).text.strip() if entry.find('atom:title',
                                                                                   NAMESPACE) is not None else ""
            abstract = entry.find('atom:summary', NAMESPACE).text.strip() if entry.find('atom:summary',
                                                                                        NAMESPACE) is not None else ""

            # 提取作者
            authors_list = []
            for author_entry in entry.findall('atom:author', NAMESPACE):
                name = author_entry.find('atom:name', NAMESPACE).text
                authors_list.append(name)
            authors = ", ".join(authors_list)

            # 提取分类
            category_tag = entry.find('atom:category', NAMESPACE)
            primary_category = category_tag.attrib[
                'term'] if category_tag is not None and 'term' in category_tag.attrib else ""

            # 构建最终数据结构
            paper_data = {
                "id": arxiv_id,
                "submitter": authors_list[0] if authors_list else "",
                "authors": authors,
                "title": title,
                "abstract": abstract,
                "categories": primary_category,
                "pdf_url": f"{PDF_BASE_URL}{arxiv_id}.pdf" if arxiv_id else None,
            }
            papers.append(paper_data)

        except Exception as e:
            logger.warning(f"解析论文条目时出错: {e}. 跳过该条目.")
            continue

    return papers


def fetch_and_save_metadata(query: str, max_results: int = 50):
    """
    调用arXiv API获取元数据并保存到文件。
    """
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results
    }

    # 构造完整的URL并打印出来
    request_url = requests.Request('GET', ARXIV_API_BASE, params=params).prepare().url
    logger.info(f"正在向 arXiv API 发送请求...")
    logger.info(f"  查询关键词: '{query}'")
    logger.info(f"  完整请求URL: {request_url}")

    try:
        response = requests.get(ARXIV_API_BASE, params=params, timeout=30)
        response.raise_for_status()

        # 打印原始响应内容
        response_text = response.text
        logger.debug("--- 原始 XML 响应内容 ---\n" + response_text[:2000] + "\n...")

        # 解析XML
        papers_data = parse_arxiv_xml(response_text)

        if not papers_data:
            logger.warning("未找到任何符合条件的论文。")

            # 进一步诊断：检查 XML 中是否有错误消息
            if "No search results" in response_text or "Error" in response_text:
                logger.warning("【诊断提示】API 响应中可能包含错误信息或明确提示无结果，请检查查询语法。")
                logger.warning("【解决建议】请尝试使用更简单的查询（例如只使用 'RAG'）。")
            return

        # 保存为 JSONL 文件
        output_path = RAW_DATA_DIR / OUTPUT_FILENAME
        with open(output_path, 'w', encoding='utf-8') as f:
            for paper in papers_data:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')

        logger.info(f"成功获取并保存 {len(papers_data)} 条论文元数据到 {output_path}")

    except requests.exceptions.RequestException as e:
        logger.error(f"请求 arXiv API 时出错: {e}")
    except Exception as e:
        logger.error(f"处理数据时发生意外错误: {e}")


# --- 运行示例 ---
if __name__ == "__main__":
    search_query = "(cat:cs.CL OR cat:cs.AI OR cat:cs.LG) AND abs:RAG"

    fetch_and_save_metadata(
        query=search_query,
        max_results=200
    )