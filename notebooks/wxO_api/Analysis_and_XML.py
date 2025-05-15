#!/usr/bin/env python
# Analysis_and_XML.py  – unified BELL / ROGERS / TELUS pipeline
# ----------------------------------------------------------------------
# 0. Imports
# ----------------------------------------------------------------------
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import datetime as dt
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.cluster import DBSCAN

# evaluation utilities
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# ----------------------------------------------------------------------
# 1.  Load CSV Datasets
# ----------------------------------------------------------------------
CSV_DIR = Path(".")
XML_DIR = Path(".")

def _load_csv(name: str, provider: str) -> pd.DataFrame:
    path = CSV_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    df["dataset_source"] = provider
    for col in ("currentIncidents", "normalIncidents"):
        if col not in df.columns:
            df[col] = 0
    # Timestamp handling
    if "timestamp" in df.columns:
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
              .astype("int64") // 1_000_000_000
        )
    else:
        candidate = next((c for c in df.columns if "date" in c.lower()), None)
        if candidate:
            df["timestamp"] = (
                pd.to_datetime(df[candidate], errors="coerce", utc=True)
                  .astype("int64") // 1_000_000_000
            )
        else:
            df["timestamp"] = np.nan
    df["currentIncidents"] = pd.to_numeric(df["currentIncidents"], errors="coerce")
    df["normalIncidents"]  = pd.to_numeric(df["normalIncidents"],  errors="coerce")
    df["ratioIncidents"]   = df["currentIncidents"] / df["normalIncidents"].replace(0, np.nan)
    df["ratioIncidents"] = (
        df["ratioIncidents"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return df

df_rogers = _load_csv("rogers.csv",     "ROGERS")
df_bell   = _load_csv("bell_data.csv",  " BELL")
df_telus  = _load_csv("telus.csv",      "TELUS")

CSV_MAPPING = {
    "BELL":   ("bell_data.csv", df_bell),
    "ROGERS": ("rogers.csv",    df_rogers),
    "TELUS":  ("telus.csv",     df_telus),
}

# 1‑A.  Print unique verified user/url entries
def _print_verified(df: pd.DataFrame, provider: str) -> None:
    if {"verified", "user/url"}.issubset(df.columns):
        urls = (
            df[df["verified"].astype(str).str.upper().eq("TRUE")]["user/url"]
            .dropna()
            .unique()
        )
        print(f"\n{provider} – verified user/url entries ({len(urls)})")
        for u in urls:
            print(f"  {u}")
    else:
        print(f"\n{provider}: missing `verified` / `user/url` columns → skipped")

## Uncomment if you want to print the names of the accounts
# for d, p in ((df_rogers, "ROGERS"), (df_bell, "BELL"), (df_telus, "TELUS")):
#     _print_verified(d, p)

feature_cols = ["cluster_size"]
numeric_cols = ["cluster_size"]

def _to_numeric(data):
    df_num = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce")
    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_num.fillna(0, inplace=True)
    return df_num.values

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("num", FunctionTransformer(_to_numeric)),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
    ]
)

## Uncomment if you want to try different classifiers
classifiers = {
    # "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    # "ExtraTrees":   ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
}

pipelines = {
    name: Pipeline([("prep", preprocessor), ("clf", clf)])
    for name, clf in classifiers.items()
}

# XML paths
XML_FILES = {
    "BELL":   XML_DIR / "Bell_Canada_2025.xml",
    "ROGERS": XML_DIR / "Rogers_2025.xml",
    "TELUS":  XML_DIR / "TELUS_2025.xml",
}

# Utility to extract cur/norm
_CUR_NORM_RE = re.compile(r"Current[^0-9]*([0-9,]+)[^0-9]+Normal[^0-9]*([0-9,]+)", flags=re.IGNORECASE|re.DOTALL)
def _extract_cur_norm(text: str) -> tuple[float|None, float|None]:
    clean = re.sub(r'<[^>]+>',' ', text)
    m = _CUR_NORM_RE.search(' '.join(clean.split()))
    if not m:
        return None, None
    try:
        return float(m.group(1).replace(",","")), float(m.group(2).replace(",",""))
    except:
        return None, None

def xml_to_rows(xml_path: Path, provider: str) -> pd.DataFrame:
    if not xml_path.exists():
        print(f"{provider}: XML not found → {xml_path}")
        return pd.DataFrame(columns=["cluster_size"])
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        print(f"{provider}: XML parse error – {exc}")
        return pd.DataFrame(columns=["cluster_size"])
    rows = []
    for it in root.findall(".//item"):
        pub_txt = it.findtext("pubDate") or ""
        desc    = it.findtext("description") or ""
        try:
            ts = dt.datetime.strptime(pub_txt.strip(), "%a, %d %b %Y %H:%M:%S %z").timestamp()
        except:
            ts = np.nan
        if not np.isnan(ts):
            rows.append({"timestamp": ts, "description": desc})
    if not rows:
        return pd.DataFrame(columns=["cluster_size"])
    df_xml = pd.DataFrame(rows)
    df_xml.dropna(subset=["timestamp"], inplace=True)
    # clustering
    filled_ts = df_xml["timestamp"].fillna(df_xml["timestamp"].median())
    db = DBSCAN(eps=3600, min_samples=3)
    ids = db.fit_predict(filled_ts.values.reshape(-1,1))
    df_xml["cluster_id"] = ids
    sizes = pd.Series(ids).value_counts().to_dict()
    df_xml["cluster_size"] = df_xml["cluster_id"].map(sizes).fillna(0).astype(int)
    df_xml["cluster_size_orig"] = df_xml["cluster_size"]
    # New logic: compute cluster_frequency
    df_xml["date"] = pd.to_datetime(df_xml["timestamp"], unit='s').dt.strftime('%Y-%m-%d')
    unique_days = df_xml.groupby("cluster_id")["date"].nunique()
    df_xml["cluster_frequency"] = df_xml["cluster_id"].map(lambda x: df_xml["cluster_size"].iloc[0] / unique_days.get(x, 1) if unique_days.get(x, 1) > 0 else 0)
    # extract ratio
    cur_norm = df_xml["description"].apply(_extract_cur_norm)
    df_xml[["currentIncidents","normalIncidents"]] = pd.DataFrame(cur_norm.tolist(), index=df_xml.index)
    df_xml["ratioIncidents"] = (df_xml["currentIncidents"] / df_xml["normalIncidents"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    # print(f"\n[DEBUG: {provider}] Example XML values after parsing:")
    # print(df_xml[["description", "currentIncidents", "normalIncidents", "ratioIncidents"]].head(10))
    # normalize and bias -> likelihood
    df_xml["ratio_norm"] = 1 / (1 + np.exp(-df_xml["ratioIncidents"]))
    # compute likelihood before filtering
    df_xml["likelihood"] = df_xml["cluster_size_orig"] + df_xml["ratio_norm"]
    # filter after computing normalized values and likelihood
    mask = df_xml["ratioIncidents"] >= 1
    df_xml = df_xml[mask].copy()
    return df_xml[["cluster_size","cluster_size_orig","likelihood","ratio_norm","cluster_frequency"]]

# ----------------------------------------------------------------------
# 5.  Per-provider processing
# ----------------------------------------------------------------------
for prov, (_csv, df) in CSV_MAPPING.items():
    print(f"\n============================================================ Processing {prov} ============================================================")
    # clustering
    filled_ts = df["timestamp"].fillna(df["timestamp"].median())
    db = DBSCAN(eps=3600, min_samples=3)
    ids = db.fit_predict(filled_ts.values.reshape(-1,1))
    df = df.copy()
    df["cluster_id"] = ids
    sizes = pd.Series(ids).value_counts().to_dict()
    df["cluster_size"] = df["cluster_id"].map(sizes).fillna(0).astype(int)
    # normalize bias
    rmin, rmax = df["ratioIncidents"].min(), df["ratioIncidents"].max()
    if rmax>rmin:
        df["ratio_norm"] = (df["ratioIncidents"]-rmin)/(rmax-rmin)
    else:
        df["ratio_norm"] = 0.0
    df["bias"] = df["ratio_norm"].where(df["ratioIncidents"]>=1,0.0)
    df["cluster_size"] = df["cluster_size"] + df["bias"]
    y = df["verified"].astype(str).str.upper().eq("TRUE").astype(int)
    X = df[feature_cols].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    for name, pipe in pipelines.items():
        print(f"\n----- Model: {name} -----")
        pipe.fit(X_tr, y_tr)
        # if hasattr(pipe.named_steps["clf"], "feature_importances_"):
        #     imp = pipe.named_steps["clf"].feature_importances_
        #     print("Feature importances:", dict(zip(feature_cols, imp)))
        acc = pipe.score(X_te, y_te)
        print(f"Test accuracy : {acc:.3f}")
        bal = balanced_accuracy_score(y_te, pipe.predict(X_te))
        print(f"Balanced accuracy : {bal:.3f}")
        print("Report (weighted):")
        print(classification_report(y_te, pipe.predict(X_te), digits=3, zero_division=0))
    # XML predictions
    print("\n--- XML‑based Predictions ---")
    df_xml = xml_to_rows(XML_FILES[prov], prov)
    if df_xml.empty:
        print(f"{prov}: no qualifying rows.")
        continue
    rf = pipelines["RandomForest"]
    preds = rf.predict(df_xml)
    probs = rf.predict_proba(df_xml)[:,1]
    alpha = 0.5
    combined = alpha*probs + (1-alpha)*df_xml["ratio_norm"].values
    maxc = combined.max()
    maxp = probs[combined.argmax()]
    print(f"\n{prov}: {len(df_xml)} qualifying rows (p={maxp:.2f}, score={maxc:.2f})")
    maxc = combined.max()
    best = [i+1 for i,v in enumerate(combined) if v==maxc]
    sample = df_xml.iloc[best[0]-1]
    print(f"  [Sample cluster_size]      {sample['cluster_size']}")
    print(f"  [Sample likelihood]        {sample['likelihood']}")
    print(f"  [Sample ratio_norm]        {sample['ratio_norm']:.6f}")
    print(f"  [Sample cluster_frequency]  {sample['cluster_frequency']:.2f}")


# ----------------------------------------------------------------------
# 6. Visualize and Save Performance Metrics
# ----------------------------------------------------------------------
print("\n=== Creating performance visualizations ===")

metrics_data = {
    "Accuracy": {"BELL": 0.652, "ROGERS": 0.614, "TELUS": 0.610},
    "Balanced Accuracy": {"BELL": 0.600, "ROGERS": 0.617, "TELUS": 0.619},
    "Precision (0)": {"BELL": 0.956, "ROGERS": 0.972, "TELUS": 0.794},
    "Precision (1)": {"BELL": 0.096, "ROGERS": 0.071, "TELUS": 0.407},
}

def create_bar_chart(data_dict, title, filename):
    plt.figure(figsize=(8, 6))
    plt.bar(data_dict.keys(), data_dict.values(), color='skyblue')
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig(filename)
    plt.close()

# Generate all bar charts
chart_files = []
for metric, values in metrics_data.items():
    fname = f"{metric.lower().replace(' ', '_')}.png"
    create_bar_chart(values, f"{metric} by Provider", fname)
    chart_files.append((metric, fname))

# Create PowerPoint presentation
prs = Presentation()
for title, img_path in chart_files:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = f"{title} by Provider"
    slide.shapes.add_picture(img_path, Inches(1), Inches(2), width=Inches(6), height=Inches(4))

pptx_file = "Model_Performance_Presentation.pptx"
prs.save(pptx_file)
print(f"[INFO] Presentation saved as {pptx_file}")