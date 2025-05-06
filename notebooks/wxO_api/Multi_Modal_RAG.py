#!/usr/bin/env python3
"""
Granite Multimodal RAG (ES v2)  —  flat-file script

* Local or IBM COS PDF ingestion  (toggle with --cloud / use_cloud flag)
* Docling conversion → text chunks, tables, picture captions
* Output: list[langchain.Document] ready for RAG

© 2025  Deniz Askin  •  Apache 2.0
"""

from __future__ import annotations

import base64, io, json, logging, os, requests, sys
from typing import List

from PIL import Image
from transformers import GPT2TokenizerFast
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import TableItem

# ──────────────────────────  Configuration  ──────────────────────────
LOCAL_PDF_PATH   = "CSR-ACX7024-configuration-guide-v1.1.pdf"
WATSONX_MODEL_ID = "meta-llama/llama-3-2-90b-vision-instruct"
WATSONX_URL      = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ──────────────────────────  Helpers  ────────────────────────────────
def get_pdf_bytes(use_cloud: bool) -> tuple[bytes, str]:
    """Load PDF from local disk or IBM COS, depending on flag."""
    if not use_cloud:
        log.info("Reading local PDF: %s", LOCAL_PDF_PATH)
        with open(LOCAL_PDF_PATH, "rb") as f:
            return f.read(), os.path.basename(LOCAL_PDF_PATH)

    import ibm_boto3
    from botocore.client import Config

    cos_client = ibm_boto3.client(
        "s3",
        ibm_api_key_id=os.environ["COS_API_KEY"],
        ibm_auth_endpoint="https://iam.cloud.ibm.com/identity/token",
        config=Config(signature_version="oauth"),
        endpoint_url=os.getenv(
            "COS_ENDPOINT",
            "https://s3.us.cloud-object-storage.appdomain.cloud",
        ),
    )
    bucket, key = os.environ["COS_BUCKET"], os.environ["COS_OBJECT_KEY"]
    log.info("Fetching %s from bucket %s", key, bucket)
    return cos_client.get_object(Bucket=bucket, Key=key)["Body"].read(), key


def iam_token(api_key: str) -> str:
    """Exchange IBM Cloud API-key for IAM token."""
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"apikey": api_key, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def caption_picture(
    img_b64: str,
    above: str,
    below: str,
    access_token: str,
    project_id: str,
) -> str:
    """Call Watsonx Vision → JSON-only caption string."""
    system_schema = (
        "You always respond strictly in JSON format matching this schema:\n"
        '{ "images": [ { "image_id": "", "title": "", "type": "", '
        '"page_reference": "", "revision_or_version": "", "author_or_source": "", '
        '"creation_or_issue_date": "", "description": "", "diagram_elements": [], '
        '"connections_or_relationships": [], "purpose_or_scenario": "", '
        '"associated_vr_or_network_context": [], "relevant_configuration_details": [], '
        '"notes_and_annotations": "" } ] }\nOnly return valid JSON.'
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_schema},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the image below using the surrounding context.\n\n"
                            f"Text above:\n{above}\n\nText below:\n{below}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            },
        ],
        "project_id": project_id,
        "model_id": WATSONX_MODEL_ID,
        "max_tokens": 2500,
        "temperature": 0.7,
        "top_p": 1,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    try:
        r = requests.post(WATSONX_URL, headers=headers, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        log.warning("Vision API failed: %s", exc)
        return '{"images":[{"description":"[unavailable]"}]}'


# ──────────────────────────  Main pipeline  ──────────────────────────
def run_pipeline(use_cloud: bool) -> List[Document]:
    """Full PDF → RAG pipeline.  Returns list of LangChain Documents."""
    pdf_bytes, pdf_name = get_pdf_bytes(use_cloud)

    # 1  Docling conversion
    ds = DocumentStream(pdf_name, io.BytesIO(pdf_bytes), InputFormat.PDF)
    conv = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    do_ocr=False, generate_picture_images=True
                )
            )
        }
    )
    doc = conv.convert(ds).document
    log.info("Converted → %d pages / %d pictures", len(doc.pages), len(doc.pictures))

    # 2  Chunk text
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    chunker = HybridChunker(tokenizer=tokenizer)
    docs: List[Document] = []
    doc_id = 0
    for ch in chunker.chunk(doc):
        if len(ch.meta.doc_items) == 1 and isinstance(
            ch.meta.doc_items[0], TableItem
        ):
            continue
        refs = " ".join(it.get_ref().cref for it in ch.meta.doc_items)
        doc_id += 1
        docs.append(
            Document(
                page_content=ch.text,
                metadata={"doc_id": doc_id, "source": pdf_name, "ref": refs},
            )
        )

    # 3  Tables
    for table in doc.tables:
        if table.label != DocItemLabel.TABLE:
            continue
        doc_id += 1
        docs.append(
            Document(
                page_content=table.export_to_markdown(),
                metadata={
                    "doc_id": doc_id,
                    "source": pdf_name,
                    "ref": table.get_ref().cref,
                    "type": "table",
                },
            )
        )

    # 4  Pictures
    if os.getenv("WATSONX_API_KEY"):
        token = iam_token(os.environ["WATSONX_API_KEY"])
        project = os.getenv("WATSONX_PROJECT", "")
    else:
        token = project = ""
        log.warning("WATSONX_API_KEY not set – skipping caption generation")

    pic_count = 0
    for ref in doc.body.children:
        if not ref.cref.startswith("#/pictures/"):
            continue

        pic_idx = int(ref.cref.split("/")[-1])
        image = doc.pictures[pic_idx].get_image(doc)
        if image is None:
            continue

        image.thumbnail((1024, 1024))
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=70)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        above = below = ""
        idx = doc.body.children.index(ref)
        if idx > 0 and doc.body.children[idx - 1].cref.startswith("#/texts/"):
            above_idx = int(doc.body.children[idx - 1].cref.split("/")[-1])
            above = doc.texts[above_idx].text.strip()
        if idx + 1 < len(doc.body.children) and doc.body.children[idx + 1].cref.startswith("#/texts/"):
            below_idx = int(doc.body.children[idx + 1].cref.split("/")[-1])
            below = doc.texts[below_idx].text.strip()

        caption = caption_picture(img_b64, above, below, token, project) if token else \
                  '{"images":[{"description":"[skipped]"}]}'

        doc_id += 1
        docs.append(
            Document(
                page_content=caption,
                metadata={
                    "doc_id": doc_id,
                    "source": pdf_name,
                    "ref": ref.cref,
                    "type": "picture",
                    "image_base64": img_b64,
                },
            )
        )
        pic_count += 1

    log.info("Created %d text, %d table, %d picture docs – total %d.",
             len([d for d in docs if d.metadata.get("type") is None]),
             len([d for d in docs if d.metadata.get("type") == "table"]),
             pic_count, len(docs))
    return docs


# ──────────────────────────  CLI harness  ──────────────────────────
def main(use_cloud: bool = False) -> None:
    # run the end‑to‑end pipeline
    try:
        docs = run_pipeline(use_cloud=use_cloud)
    except Exception as exc:  # noqa: BLE001
        log.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)

    # quick sanity-print
    for d in docs[:3]:
        print("\n— Doc snippet —")
        print(json.dumps(d.metadata, indent=2))
        print(d.page_content[:400], "..." if len(d.page_content) > 400 else "")


if __name__ == "__main__":
    # Toggle here if you prefer a hard‑coded default
    use_cloud = False
    main(use_cloud)