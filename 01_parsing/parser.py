# from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form, status
# from fastapi.staticfiles import StaticFiles
# from typing import List

# from elasticsearch import Elasticsearch
# from langchain_elasticsearch import ElasticsearchStore
# from langchain_huggingface import HuggingFaceEmbeddings
# from uuid import uuid4
# import torch
# import numpy as np
# import shutil
# import random
# import logging

import os
# os.chdir("/home/share/00_API/routers")

TEMP_DIR = "static"
# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    FormatOption,
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    MarkdownFormatOption,
    ExcelFormatOption,
    PowerpointFormatOption,
    CsvFormatOption,
    HTMLFormatOption
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.asciidoc_backend import AsciiDocBackend
from docling.backend.csv_backend import CsvDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
# from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.json.docling_json_backend import DoclingJSONBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.backend.xml.jats_backend import JatsDocumentBackend
from docling.backend.xml.uspto_backend import PatentUsptoDocumentBackend
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
import re

### 아래가 파일 첨부와 파싱을 비동기 처리로 개선한 함수임
import asyncio
import re
from typing import Any

from langchain_core.documents import Document



# 모듈 최상단에 패턴 컴파일
NEWLINE_PATTERN = re.compile(r'\r\n\d+')

def normalize_newlines(text: str) -> str:
    """개행문자 정규화 (동기 함수)"""
    return NEWLINE_PATTERN.sub('\n', text)

# 공유 가능한 옵션 정의
DEFAULT_PIPELINE_OPTIONS = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    ocr_options=EasyOcrOptions(lang=["en", "ko"])
    )

async def all_in_one_hybrid(path: str) -> list[str]:
    """비동기 문서 변환 파이프라인 (페이지 단위 리스트 반환)"""
    # 1. 옵션 설정
    pipeline_options = DEFAULT_PIPELINE_OPTIONS

    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX,
            InputFormat.XLSX, InputFormat.HTML, InputFormat.PPTX,
            InputFormat.MD, InputFormat.CSV
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            ),
            InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
            InputFormat.PPTX: PowerpointFormatOption(
                pipeline_cls=SimplePipeline, backend=MsPowerpointDocumentBackend
            ),
            InputFormat.XLSX: ExcelFormatOption(
                pipeline_cls=SimplePipeline, backend=MsExcelDocumentBackend
            ),
            InputFormat.MD: MarkdownFormatOption(
                pipeline_cls=SimplePipeline, backend=MarkdownDocumentBackend
            ),
            InputFormat.HTML: HTMLFormatOption(
                pipeline_cls=SimplePipeline, backend=HTMLDocumentBackend
            ),
            InputFormat.CSV: CsvFormatOption(
                pipeline_cls=SimplePipeline, backend=CsvDocumentBackend
            ),
        }
    )

    # 2. 비동기 변환 실행
    loop = asyncio.get_running_loop()
    conversion_result = await loop.run_in_executor(
        None,
        doc_converter.convert,
        path
    )

    # 3. 결과 처리
    docs = conversion_result.document
    normalized_docs = normalize_newlines(docs.export_to_markdown())
    normalized_docs = [Document(page_content=normalized_docs)]

    # 4. 마크다운 스플릿 적용 여부
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
            ("####", "Header_4"),
        ],
        )
    normalized_docs = [split for doc in normalized_docs for split in splitter.split_text(doc.page_content)]

    return normalized_docs


if __name__ == "__main__":
    path = "D:/AI_Labs/00_data/Unit_Cooler.pdf"
    result = asyncio.run(all_in_one_hybrid(path=path))
    for doc in result:
        print(doc)
        print("")
    print(len(result))