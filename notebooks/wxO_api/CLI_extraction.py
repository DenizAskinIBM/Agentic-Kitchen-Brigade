import openai
from tqdm import tqdm
import time
from langchain_openai import ChatOpenAI
def save_output(md_file_path, cli_blocks):
    with open(md_file_path, 'w') as f:
        f.write('\n\n'.join(cli_blocks))
import subprocess
from bs4 import BeautifulSoup
def extract_cli_from_html(html_file):
    import re
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')

    cli_pattern = re.compile(r'^\s*(set|show|edit|request|clear|delete|run|ping|traceroute|monitor|op|restart|start|stop|commit|configure|load|save|rollback|exit|help|file|cli|junos)', re.IGNORECASE)
    cli_commands = []

    for block in soup.find_all(['pre', 'code', 'div']):
        if 'monospace' in str(block.get('style')) or block.name in ['pre', 'code']:
            lines = block.get_text().split('\n')
            for line in lines:
                if cli_pattern.match(line.strip()) and len(line.strip().split()) > 1:
                    cli_commands.append(f"[CLI]{line.strip()}[/CLI]")

    return cli_commands
import io
import pymupdf as fitz
from ibm_watsonx_ai.foundation_models import ModelInference

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions

import os
os.environ['WATSONX_API_KEY'] = "c5jsf7CBqlU_kPW8Hj80VnU13RX2ezGPTs6Fyg0-1s3A"
os.environ['WATSONX_PROJECT_ID'] = "d3d3c03e-2dd6-4892-b7d7-139c5dc8a1d0"

API_KEY="c5jsf7CBqlU_kPW8Hj80VnU13RX2ezGPTs6Fyg0-1s3A"
URL="https://us-south.ml.cloud.ibm.com/"
PROJECT_ID="d3d3c03e-2dd6-4892-b7d7-139c5dc8a1d0"

from langchain_ibm import WatsonxLLM

parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

# === Step 1: Define the LLM (LLaMA on Watsonx) ===
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="c5jsf7CBqlU_kPW8Hj80VnU13RX2ezGPTs6Fyg0-1s3A",           # 🔁 Use same key from get_iam_access_token()
    project_id="d3d3c03e-2dd6-4892-b7d7-139c5dc8a1d0",  # Replace with yours
    params=parameters
)

validator_llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=URL,
    apikey=API_KEY,
    project_id=PROJECT_ID,
    params=parameters
)

openai_apikey = os.getenv("OPENAI_API_KEY")
llm_chat_gpt = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_apikey
)

# PDF-to-text extraction logic using PyMuPDF
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    return all_text

if __name__ == "__main__":
    import re

    print("\n🚀 Running extraction via HTML parsing...")
    from pdfminer.high_level import extract_text_to_fp
    from io import BytesIO

    def convert_pdf_to_html(pdf_path, html_output_path):
        output_buffer = BytesIO()
        with open(pdf_path, 'rb') as pdf_file:
            extract_text_to_fp(pdf_file, output_buffer, output_type='html')
        html_content = output_buffer.getvalue().decode('utf-8')
        with open(html_output_path, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)

    convert_pdf_to_html("system-mgmt-monitoring.pdf", "output.html")
    html_blocks = extract_cli_from_html("output.html")
    html_cli_output = []
    for _ in tqdm(html_blocks, desc="HTML CLI Extraction (HTML Parsing)"):
        html_cli_output.append(_)  # skip print
    save_output("html_output.md", html_cli_output)

    print("\n🚀 Running extraction with regex+LLM...")
    regex_llm_output = []
    doc = fitz.open("system-mgmt-monitoring.pdf")
    for i, page in enumerate(tqdm(doc, desc="Regex+LLM Extraction"), start=1):
        page_text = page.get_text()
        candidate_lines = []
        for line in page_text.splitlines():
            if re.match(r"^\s*(set|show|edit|request|clear|delete|run|user@\S+[>#]|ping|traceroute|monitor|op|restart|start|stop|commit|configure|load|save|rollback|exit|help|file|cli|junos)", line):
                candidate_lines.append(line.strip())
        if not candidate_lines:
            continue
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Junos OS CLI expert. Given command-line fragments, identify and refine actual CLI commands.
Wrap each valid command between [CLI] and [/CLI].
<|eot_id|><|start_header_id|>user<|end_header_id|>
\"\"\"
{chr(10).join(candidate_lines)}
\"\"\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        response = llm.invoke(full_prompt)
        extracted = re.findall(r"\[CLI\](.*?)\[/CLI\]", response, re.DOTALL)
        extracted = [x.strip() for x in extracted if len(x.strip().split()) > 1]
        regex_llm_output.extend([f"[CLI]{cmd}[/CLI]" for cmd in extracted])

    save_output("regex_llm_output.md", regex_llm_output)

    print("\n🤖 Evaluating outputs using LLM-as-a-judge...")
    from langchain_core.messages import HumanMessage, SystemMessage

    judge_messages = [
        SystemMessage(content="You are a judge comparing two methods of extracting CLI commands from a Junos OS PDF document. The goal is to accurately identify CLI commands and represent them using [CLI] and [/CLI] tags."),
        HumanMessage(content=f"""You are provided:
- The source PDF: 'system-mgmt-monitoring.pdf'
- Method 1 output (regex + LLM):\n{open('regex_llm_output.md').read()}
- Method 2 output (HTML parsing):\n{open('html_output.md').read()}

Evaluate which file has more accurate, complete, and well-formatted CLI command extraction.
Output only one line:(regex + LLM) or (HTML parsing).""")
    ]

    judge_result = llm_chat_gpt.invoke(judge_messages)
    print("\n🏁 LLM-Judge Decision:", judge_result.content.strip())