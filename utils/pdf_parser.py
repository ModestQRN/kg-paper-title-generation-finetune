import fitz  # PyMuPDF
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PDFParser:
    """
    PDF解析模块，用于从PDF文件中提取文本内容
    """

    def __init__(self):
        self.sections = [
            "abstract", "introduction", "related work", "methodology",
            "method", "experiment", "results", "discussion",
            "conclusion", "references"
        ]
        # 分块重叠大小
        self.chunk_overlap = 200

    # ----------------------------------------------------
    # 核心方法: 提取全文
    # ----------------------------------------------------
    def extract_text(self, pdf_path: str) -> str:
        """
        从PDF提取所有文本内容
        """
        try:
            logger.debug(f"正在打开 PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                logger.debug(f"正在提取第 {page_num + 1}/{len(doc)} 页文本...")
                page = doc.load_page(page_num)
                # 使用 "text" 方法提取文本
                text += page.get_text("text") + "\n\n"

            logger.debug("所有页面文本提取完成。")
            doc.close()
            return text
        except Exception as e:
            logger.error(f"从PDF提取文本时出错: {e}")
            raise

    # ----------------------------------------------------
    # 核心方法: 提取元数据
    # ----------------------------------------------------
    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        从PDF文件提取元数据
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            # 简化元数据
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "creationDate": metadata.get("creationDate", "")
            }
        except Exception as e:
            logger.error(f"提取PDF元数据时出错: {e}")
            return {}

    # ----------------------------------------------------
    # 核心方法: 提取章节内容 (修复了索引越界问题)
    # ----------------------------------------------------
    def extract_sections(self, full_text: str) -> Dict[str, str]:
        """
        从全文中提取关键章节内容
        """
        sections_content = {}
        text = full_text.lower()
        logger.debug("开始提取章节...")

        # 使用正则表达式查找各个章节
        for i, section in enumerate(self.sections):

            # 设置下一个章节作为边界，防止索引越界
            boundary = ""
            if i < len(self.sections) - 1:
                # 匹配下一个章节标题 或 常见的末尾标题 (references/acknowledgements/appendix) 或 文件末尾
                boundary = f"^{self.sections[i + 1]}|references|acknowledgements|appendix|$"
            else:
                # 最后一个章节，只匹配到文件末尾
                boundary = "$"

            # 构造正则表达式
            pattern = re.compile(
                # ^section\s*\n+(.*?)\n+(?=boundary)
                f"^{section}\\s*\\n+(.*?)\\n+(?={boundary})",
                re.DOTALL | re.MULTILINE | re.IGNORECASE)

            match = pattern.search(text)

            if match:
                content = match.group(1).strip()
                # 排除只有一个标题的情况
                if len(content) > 50:
                    sections_content[section] = content

        return sections_content

    # ----------------------------------------------------
    # 核心方法: 文本分块 (修复了 rfind 性能和末尾无限循环)
    # ----------------------------------------------------
    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        将长文本分割成重叠的块。
        """
        if not text:
            logger.warning("输入文本为空，无法分块。")
            return []
        chunks = []
        start = 0
        text_len = len(text)
        overlap = self.chunk_overlap

        while start < text_len:
            # 计算当前块的结束位置
            end = min(start + chunk_size, text_len)

            # ------------------------------------------------------------------
            # 性能修复：禁用 rfind 逻辑 (避免在循环中对长字符串进行多次昂贵的搜索)
            # ------------------------------------------------------------------
            if end < text_len:
                last_period = -1  # 禁用句子边界对齐

                if last_period != -1:
                    end = last_period + 2

            # 检查 start 和 end 的有效性，防止空块
            if end > start:
                chunks.append(text[start:end])

                # --- 关键 FIX: 仅在未达到文件末尾时，才为下一个循环设置重叠起始点 ---
                if end < text_len:
                    # 如果不是最后一个块，计算下一个块的重叠起始点
                    start = end - self.chunk_overlap
                else:
                    # 如果 end == text_len (已到达文件末尾)，设置 start = text_len
                    # 确保 while 循环在下一轮终止，避免无限循环。
                    start = text_len
            else:
                # 如果 end <= start，说明无法继续有效分块，跳出循环
                break

        return chunks

    # ----------------------------------------------------
    # 核心方法: 统一处理流程
    # ----------------------------------------------------
    def process_pdf(self, pdf_path: str, chunk_size: int = 1000) -> Dict:
        """
        处理PDF文件，提取文本、元数据和分块内容
        """
        # 1. 提取元数据
        metadata = self.extract_metadata(pdf_path)

        # 2. 提取全文
        try:
            full_text = self.extract_text(pdf_path)
        except Exception:
            logger.error(f"跳过文件 {pdf_path}，无法提取文本。")
            return {"metadata": metadata, "sections": {}, "full_text": "", "chunks": [], "sectioned_chunks": {}}

        # 3. 提取章节
        sections = self.extract_sections(full_text)
        logger.debug(f"章节提取完成。提取到 {len(sections)} 个章节。")

        # 4. 将全文分块
        logger.debug("开始对全文进行分块...")
        chunks = self.chunk_text(full_text, chunk_size)
        logger.debug(f"全文分块完成。共 {len(chunks)} 块。")

        # 5. 将每个章节分块
        logger.debug("开始对各章节内容进行分块...")
        sectioned_chunks = {}
        for section, content in sections.items():
            sectioned_chunks[section] = self.chunk_text(content, chunk_size)
        logger.debug("章节内容分块完成。")

        return {
            "metadata": metadata,
            "sections": sections,
            "full_text": full_text,
            "chunks": chunks,
            "sectioned_chunks": sectioned_chunks
        }