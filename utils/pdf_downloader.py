import requests
import json
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 配置
METADATA_PATH = Path(__file__).parent.parent / "data" / "arxiv_metadata.jsonl"
DOWNLOAD_DIR = Path(__file__).parent.parent / "data" / "pdf_data"

def download_pdfs():
    """读取元数据文件，批量下载PDF。"""
    if not METADATA_PATH.exists():
        logger.error(f"元数据文件未找到: {METADATA_PATH}")
        return

    logger.info("开始批量下载PDF文件...")

    # 1. 加载元数据
    papers_to_download = []
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                papers_to_download.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.error(f"解析JSONL行时出错: {e}")

    # 2. 批量下载
    for paper in tqdm(papers_to_download, desc="下载PDF"):
        pdf_url = paper.get("pdf_url")
        paper_id = paper.get("id")

        if not pdf_url:
            logger.warning(f"论文 {paper_id} 缺少pdf_url，跳过。")
            continue

        file_name = f"{paper_id}.pdf"
        output_path = DOWNLOAD_DIR / file_name

        if output_path.exists():
            # logger.debug(f"文件 {file_name} 已存在，跳过。")
            continue

        try:
            response = requests.get(pdf_url, stream=True, timeout=15)
            response.raise_for_status()  # 检查HTTP错误

            with open(output_path, 'wb') as pdf_file:
                for chunk in response.iter_content(chunk_size=8192):
                    pdf_file.write(chunk)

            # logger.info(f"成功下载: {file_name}")

        except requests.exceptions.RequestException as e:
            logger.error(f"下载 {file_name} 时出错 ({e})")
        except Exception as e:
            logger.error(f"处理 {file_name} 时发生意外错误: {e}")

    logger.info("所有PDF下载尝试完成。")


if __name__ == "__main__":
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    download_pdfs()