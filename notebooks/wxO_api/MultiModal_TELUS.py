import os
# Suppress Milvus C++ server logs (glog)
os.environ["GLOG_minloglevel"] = "3"
# Disable HuggingFace tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import io
import re
import json
import fitz  # PyMuPDF
import logging
import base64
import tempfile
import requests
from PIL import Image
from pathlib import Path
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
from transformers import GPT2TokenizerFast
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_ibm import WatsonxLLM
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem
from langchain_core.documents import Document as LangDocument

from dotenv import load_dotenv

# ----------------------------------------
# Regex matchers for CLI
load_dotenv()
PROMPT_RX = re.compile(r'^[\w\-\./]+@[\w\-\./]+[>%#]\s+.+')
CONFIG_RX = re.compile(r'^\s*(?:set|delete|show|request|clear|commit|rollback)\s+.+', re.IGNORECASE)

def is_cli(text: str) -> bool:
    return bool(PROMPT_RX.match(text.strip()) or CONFIG_RX.match(text.strip()))

def extract_colored_cli_spans(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    cli_spans = []
    for page_idx, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    if not is_cli(text):
                        continue
                    c = span.get("color", 0)
                    r, g, b = (c >> 16 & 0xFF, c >> 8 & 0xFF, c & 0xFF)
                    cli_spans.append({
                        "page": page_idx,
                        "cli": text.strip(),
                        "color": (r, g, b),
                        "bbox": span["bbox"],
                    })
    doc.close()
    return cli_spans

def save_pdf_as_txt(pdf_path: str, txt_path: str):
    doc = fitz.open(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as out:
        for page in doc:
            out.write(page.get_text())
            out.write("\f")
    doc.close()

def extract_clis(txt_path: str) -> list[str]:
    clis = []
    with open(txt_path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip()
            if not (PROMPT_RX.match(line) or CONFIG_RX.match(line)):
                continue
            stripped = line.strip()
            if len(stripped) > 200 or ' ' not in stripped:
                continue
            clis.append(stripped)
    seen = set()
    unique = []
    for cmd in clis:
        if cmd not in seen:
            seen.add(cmd)
            unique.append(cmd)
    return unique

def get_iam_access_token(api_key: str) -> str:
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = requests.post(
        iam_url,
        data={"apikey": api_key, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    response.raise_for_status()
    return response.json()["access_token"]

def main():

    print("Loading and OCR-ing PDF...")

    # Load and OCR PDF
    with open(pdf_path, "rb") as f_in:
        reader = PdfReader(f_in)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        pdf_bytes_io = io.BytesIO()
        writer.write(pdf_bytes_io)
    pdf_bytes = pdf_bytes_io.getvalue()

    print("PDF loaded into memory.")
    global cli_spans

    with open(pdf_path, "wb") as f_out:
        f_out.write(pdf_bytes)

    # Extract CLI spans with colors
    cli_spans = extract_colored_cli_spans(pdf_bytes)
    print(f"Found {len(cli_spans)} colored CLI spans.")

    print("Saving PDF as text...")
    save_pdf_as_txt(pdf_path, txt_path)
    commands = extract_clis(txt_path)
    print("Extracting CLI Commands...")
    # for cmd in commands:
    #     print(cmd)

    # Convert to Docling document
    stream_for_converter = io.BytesIO(pdf_bytes)
    document_stream = DocumentStream(
        name=pdf_path,
        stream=stream_for_converter,
        input_format=InputFormat.PDF
    )
    pipeline_opts = PdfPipelineOptions(do_ocr=False, generate_picture_images=True)
    format_opts = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    converter = DocumentConverter(format_options=format_opts)
    global converted_document
    converted_document = converter.convert(source=document_stream).document
    print("Converted PDF to Docling document.")

    print("Chunking document...")
    # Chunking
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    max_tokens = tokenizer.model_max_length
    chunker = HybridChunker(tokenizer=tokenizer, chunk_size=max_tokens, chunk_overlap=20)

    doc_id = 0
    global texts
    texts = []
    for chunk in chunker.chunk(converted_document):
        items = chunk.meta.doc_items
        if len(items) == 1 and isinstance(items[0], TableItem):
            continue
        refs = " ".join(item.get_ref().cref for item in items)
        tokens = tokenizer.encode(chunk.text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        safe_text = tokenizer.decode(tokens, skip_special_tokens=True)
        texts.append(LangDocument(
            page_content=safe_text,
            metadata={"doc_id": (doc_id := doc_id + 1), "source": pdf_path, "ref": refs}
        ))
    print(f"Prepared {len(texts)} text chunks for embedding.")

    print("Initializing vector database...")
    # Initialize vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    global vector_db
    vector_db = Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": db_file},
        auto_id=True,
        index_params={"index_type": "AUTOINDEX"},
    )
    vector_db.add_documents(texts)
    print(f"✅ {len(texts)} documents added to Milvus vector DB")

if __name__ == "__main__":
    # Define paths before running main
    pdf_path = "CSR-ACX7024-configuration-guide-v1.1.pdf"
    txt_path = pdf_path.replace(".pdf", ".txt")
    main()

    print("Setting up LLM...")
    # LLM Setup
    API_KEY = os.getenv("WATSONX_API_KEY")
    PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    access_token = get_iam_access_token(API_KEY)
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-405b-instruct",
        url="https://us-south.ml.cloud.ibm.com",
        apikey=API_KEY,
        project_id=PROJECT_ID
    )

    # Custom question
    user_question = "Are there any Junos OS commands that are in red in this document? If so, print them please."
    print("Retrieving relevant documents from vector DB...")
    # Use the new retriever.invoke method instead of deprecated get_relevant_documents
    docs = vector_db.as_retriever().invoke(user_question, k=5)
    context = "\n\n".join(d.page_content for d in docs)
    retrieved_pages = {d.metadata.get("page") for d in docs if d.metadata.get("page")}
    spans_for_prompt = [s for s in cli_spans if s["page"] in retrieved_pages and s["color"] == (255, 0, 0)]
    json_color_data = json.dumps(spans_for_prompt, ensure_ascii=False)

    prompt = f"""
You are a Junos CLI expert. Use ONLY the context and color metadata to answer.
COLOR_DATA: {json_color_data}
Context: {context}
Question: {user_question}
"""
    print("Invoking LLM with prompt...")
    # Pass generation parameters directly as keyword arguments
    response = llm.invoke(
        prompt,
        decoding_method="sample",
        max_new_tokens=1000,
        temperature=0.5,
        top_p=1,
        top_k=40
    )
    print("\n🧠 LLM Response:")
    print(response.strip())
    print("LLM invocation complete.")
