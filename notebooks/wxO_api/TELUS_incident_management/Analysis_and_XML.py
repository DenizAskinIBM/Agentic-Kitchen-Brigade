#!/usr/bin/env python
# Analysis_and_XML.py  – unified BELL / ROGERS / TELUS pipeline
# ----------------------------------------------------------------------
# 0. Imports
# ----------------------------------------------------------------------
from __future__ import annotations
from typing import ClassVar
from pathlib import Path
import re
import datetime as dt
import matplotlib.pyplot as plt
from graphviz import Source

import sys
print("Running:", __file__)

# === CrewAI Agentic Workflow Setup ===
import os
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from webscape_and_categorize import scrape_all
load_dotenv()

# Watsonx credentials
url = os.getenv("WATSONX_URL")
apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 10000,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42  
}

llm_llama = ChatWatsonx(
    model_id="meta-llama/llama-3-405b-instruct",
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)

# import the SCRAPERS dict
from webscape_and_categorize import SCRAPERS

import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)

import numpy as np
import pandas as pd
 # store matched incidents for LLM reporting
MATCHED_STORE: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# evaluation utilities
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# === CrewAI Agent/Flow Imports ===
from crewai import Agent
from crewai.flow.flow import Flow

# === CrewAI Tools ===
from crewai.tools import tool

@tool
def ingest_data(provider: str):
    """
    Ingest CSV and web data for the given provider and compute matched incidents only.
    """
    df_csv = CSV_MAPPING[provider][1].copy()
    incidents = SCRAPERS[provider]()
    df_web = pd.DataFrame(incidents)
    # --- ensure web DataFrame has a timestamp column ---
    if "timestamp" not in df_web.columns:
        if "date" in df_web.columns:
            df_web["timestamp"] = (
                pd.to_datetime(df_web["date"], errors="coerce")
                  .astype("int64") // 1_000_000_000
            )
        elif "datetime" in df_web.columns:
            df_web["timestamp"] = (
                pd.to_datetime(df_web["datetime"], errors="coerce")
                  .astype("int64") // 1_000_000_000
            )
        else:
            raise KeyError(f"No viable date column in web-scraped data for provider {provider}")
    # --- keyword matching via TF-IDF ---
    keywords = ["outage", "down", "maintenance"]
    # ensure description column exists
    descs = df_web.get("description", pd.Series([""] * len(df_web))).fillna("")
    tfidf_vec = TfidfVectorizer(
        vocabulary=keywords,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b"
    )
    X_kw = tfidf_vec.fit_transform(descs)
    # add TF-IDF feature columns per keyword
    for idx, kw in enumerate(keywords):
        df_web[f"kw_{kw}_tfidf"] = X_kw[:, idx].toarray().ravel()
    # log rows containing any keyword
    any_kw = (X_kw.sum(axis=1) > 0).A1.sum()
    # --- filter web rows by new dates/weeks relative to CSV ---
    # extract dates and ISO weeks from CSV
    csv_dates = set(pd.to_datetime(df_csv["timestamp"], unit="s").dt.date)
    csv_weeks = set(
        pd.to_datetime(df_csv["timestamp"], unit="s")
         .dt.isocalendar()
         .week
    )
    # extract dates and ISO weeks from web-scraped data
    df_web["date"] = pd.to_datetime(df_web["timestamp"], unit="s").dt.date
    df_web["week"] = pd.to_datetime(df_web["timestamp"], unit="s").dt.isocalendar().week

    # Restrict web scraping to only CSV test-set dates
    df_web = df_web[df_web["date"].isin(csv_dates)]

    # After filtering, all rows are matched by day
    matched_by_day = df_web
    # Only store matched incidents for LLM reporting; do not merge or print.
    csv_dates_series = pd.to_datetime(df_csv["timestamp"], unit="s", errors="coerce").dt.date
    csv_matched = df_csv[csv_dates_series.isin(csv_dates)]
    MATCHED_STORE[provider] = (matched_by_day.copy(), csv_matched.copy())
    return df_csv

@tool
def extract_features(df: pd.DataFrame, provider: str):
    """
    Perform preprocessing and feature engineering on the merged DataFrame for the given provider.
    """
    # Feature engineering: clustering on timestamp with fixed DBSCAN parameters
    # Ensure timestamp numeric
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
    filled_ts = df["timestamp"].fillna(df["timestamp"].median())
    # Use fixed DBSCAN parameters: eps=3600, min_samples=3
    db = DBSCAN(eps=3600, min_samples=3)
    ids = db.fit_predict(filled_ts.values.reshape(-1,1))
    df["cluster_id"] = ids
    # cluster size
    sizes = pd.Series(ids).value_counts().to_dict()
    df["cluster_size"] = df["cluster_id"].map(sizes).fillna(0).astype(int)
    # cluster frequency (reports per day)
    df["date"] = pd.to_datetime(df["timestamp"], unit='s').dt.floor('d')
    days = df.groupby("cluster_id")["date"].transform("nunique").replace(0,1)
    df["cluster_frequency"] = df["cluster_size"] / days
    # inter-arrival time
    df = df.sort_values("timestamp")
    df["inter_arrival"] = df["timestamp"].diff().fillna(df["timestamp"].median())
    # moving-window count (past 1h)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit='s')
    df.set_index("datetime", inplace=True)
    df["count_1h"] = df["timestamp"].rolling("1h").count().astype(int)
    df.reset_index(drop=False, inplace=True)
    # time features
    df["datetime_ts"] = pd.to_datetime(df["timestamp"], unit='s')
    df["hour"] = df["datetime_ts"].dt.hour
    df["weekday"] = df["datetime_ts"].dt.weekday
    # Z-score normalize count_1h
    df["count_1h_z"] = (df["count_1h"] - df["count_1h"].mean()) / df["count_1h"].std(ddof=0)
    # Clean up temporary columns
    df.drop(columns=["date", "datetime", "datetime_ts"], inplace=True, errors="ignore")
    return df

@tool
def train_and_evaluate(df: pd.DataFrame, provider: str, test_num: int):
    """
    Train and evaluate the classifier for the specified test number (1 or 2) on the given DataFrame.
    Returns metrics and the trained model.
    """
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    from collections import Counter

    # Select features
    raw_features = [
        "user/totalTweets",
        "user/totalFollowing",
        "user/totalFollowers",
        "timestamp",
        "cluster_size",
        "cluster_frequency",
        "inter_arrival",
        "count_1h",
        "hour",
        "weekday",
        "count_1h_z"
    ]
    feature_cols = [
        "timestamp",
        "cluster_size",
        "cluster_frequency",
        "inter_arrival",
        "count_1h",
        "hour",
        "weekday",
        "count_1h_z"
    ]

    if test_num == 1:
        feature_set = raw_features
        print(f"\n=== Test #{test_num}: Raw Features for {provider} ===")
    else:
        feature_set = feature_cols
        print(f"\n=== Test #{test_num}: Engineered Features for {provider} ===")

    # Prepare X and y
    X = df[feature_set].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["verified"].astype(str).str.upper().eq("TRUE").astype(int)

    # Split and balance
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    # Print cluster_size distribution in test set
    from collections import Counter
    cluster_dist = Counter(X_te["cluster_size"])
    print("Test set cluster_size distribution:", cluster_dist)
    sm = SMOTE(random_state=42)
    print("Before SMOTE:", Counter(y_tr))
    X_tr_bal, y_tr_bal = sm.fit_resample(X_tr, y_tr)
    print("After SMOTE: ", Counter(y_tr_bal))

    # Train classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X_tr_bal, y_tr_bal)

    # Evaluate
    acc = clf.score(X_te, y_te)
    bal_acc = balanced_accuracy_score(y_te, clf.predict(X_te))
    report = classification_report(y_te, clf.predict(X_te), digits=3, output_dict=True, zero_division=0)
    print(f"Features used: {feature_set}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Balanced accuracy: {bal_acc:.3f}")
    print("Report (weighted):")
    # Print human-readable report
    print(classification_report(y_te, clf.predict(X_te), digits=3, zero_division=0))

    # Plot F1 scores
    f1_scores = {"0": report["0"]["f1-score"], "1": report["1"]["f1-score"]}
    plt.figure(figsize=(4,3))
    plt.bar(f1_scores.keys(), f1_scores.values())
    plt.title(f"F1 Scores Test {test_num} - {provider}")
    plt.ylim(0,1)
    fname = f"f1_{provider.lower()}_test{test_num}.png"
    plt.savefig(fname)
    plt.close()
    print(f"[INFO] F1 chart saved to {fname}")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_0": report["0"]["f1-score"],
        "f1_1": report["1"]["f1-score"]
    }

# === CrewAI Agent Definitions ===
class IngestionAgent(Agent):
    name: ClassVar[str] = "Ingestion"
    role: ClassVar[str] = "Ingest and merge incident data"
    goal: ClassVar[str] = "Produce a merged DataFrame of new incidents"
    backstory: ClassVar[str] = "Agent responsible for fetching, de-duplicating, and merging incident data from CSV and web sources for each provider."
    def run(self, provider: str):
        df_csv = ingest_data.run(provider=provider)
        matched_web, matched_csv = MATCHED_STORE.get(provider, (pd.DataFrame(), pd.DataFrame()))
        return {"provider": provider, "df_csv": df_csv, "matched_web": matched_web}

class FeAgent(Agent):
    name: ClassVar[str] = "FeatureEngineering"
    role: ClassVar[str] = "Engineer features from merged data"
    goal: ClassVar[str] = "Create and transform features for model training"
    backstory: ClassVar[str] = "Agent tasked with performing feature engineering, such as clustering and time-based features, on merged incident data."
    def run(self, provider: str, df_csv: pd.DataFrame):
        from sklearn.model_selection import train_test_split
        # stratify on verified if available
        if "verified" in df_csv.columns:
            strat = df_csv["verified"].astype(str).str.upper().eq("TRUE")
        else:
            strat = None
        _, test_df = train_test_split(df_csv, test_size=0.2, random_state=42, stratify=strat)
        features_df = extract_features.run(df=test_df, provider=provider)
        return {"provider": provider, "features_df": features_df}

class TrainAgent(Agent):
    name: ClassVar[str] = "TrainAndEval"
    role: ClassVar[str] = "Train and evaluate incident classifier"
    goal: ClassVar[str] = "Fit and assess a RandomForest model for incident classification"
    backstory: ClassVar[str] = "Agent responsible for training and evaluating a RandomForest classifier using engineered features and reporting performance metrics."
    test_num: int = 0
    def run(self, provider: str, features_df: pd.DataFrame):
        # Always use n_estimators=300 for RandomForestClassifier
        def train_and_evaluate_with_trees(df, provider, test_num):
            from sklearn.model_selection import train_test_split
            from imblearn.over_sampling import SMOTE
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
            import matplotlib.pyplot as plt
            from collections import Counter

            # Select features
            raw_features = [
                "text",
                "user/totalTweets",
                "user/totalFollowing",
                "user/totalFollowers",
                "timestamp",
                "cluster_size",
                "cluster_frequency",
                "inter_arrival",
                "count_1h",
                "hour",
                "weekday",
                "count_1h_z"
            ]
            feature_cols = [
                "timestamp",
                "cluster_size",
                "cluster_frequency",
                "inter_arrival",
                "count_1h",
                "hour",
                "weekday",
                "count_1h_z"
            ]

            if test_num == 1:
                feature_set = raw_features
                print(f"\n=== Test #{test_num}: Raw Features for {provider} ===")
            else:
                feature_set = feature_cols
                print(f"\n=== Test #{test_num}: Engineered Features for {provider} ===")

            # Prepare X and y
            X = df[feature_set].apply(pd.to_numeric, errors="coerce").fillna(0)
            y = df["verified"].astype(str).str.upper().eq("TRUE").astype(int)

            # Split and balance
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
            # Print cluster_size distribution in test set
            from collections import Counter
            cluster_dist = Counter(X_te["cluster_size"])
#           print("Test set cluster_size distribution:", cluster_dist)
            sm = SMOTE(random_state=42)
            print("Before SMOTE:", Counter(y_tr))
            X_tr_bal, y_tr_bal = sm.fit_resample(X_tr, y_tr)
            print("After SMOTE: ", Counter(y_tr_bal))

            # Train classifier
            clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
            clf.fit(X_tr_bal, y_tr_bal)

            # Evaluate
            acc = clf.score(X_te, y_te)
            bal_acc = balanced_accuracy_score(y_te, clf.predict(X_te))
            report = classification_report(y_te, clf.predict(X_te), digits=3, output_dict=True, zero_division=0)
            print(f"Features used: {feature_set}")
            print(f"Accuracy: {acc:.3f}")
            print(f"Balanced accuracy: {bal_acc:.3f}")
            print("Report (weighted):")
            # Print human-readable report
            print(classification_report(y_te, clf.predict(X_te), digits=3, zero_division=0))

            # Plot F1 scores
            f1_scores = {"0": report["0"]["f1-score"], "1": report["1"]["f1-score"]}
            plt.figure(figsize=(4,3))
            plt.bar(f1_scores.keys(), f1_scores.values())
            plt.title(f"F1 Scores Test {test_num} - {provider}")
            plt.ylim(0,1)
            fname = f"f1_{provider.lower()}_test{test_num}.png"
            plt.savefig(fname)
            plt.close()
            print(f"[INFO] F1 chart saved to {fname}")

            return {
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "f1_0": report["0"]["f1-score"],
                "f1_1": report["1"]["f1-score"]
            }
        metrics = train_and_evaluate_with_trees(df=features_df, provider=provider, test_num=self.test_num)
        return {"provider": provider, f"metrics_test{self.test_num}": metrics}

class ReportAgent(Agent):
    name: ClassVar[str] = "MatchedReport"
    role: ClassVar[str] = "Generate incident summary report"
    goal: ClassVar[str] = "Produce an LLM-based summary of matched incidents"
    backstory: ClassVar[str] = "Agent that summarizes matched incidents using an LLM, providing concise reporting for downstream analysis."
    def run(self, provider: str, df_csv: pd.DataFrame, matched_web: pd.DataFrame, features_df: pd.DataFrame, **kwargs):
        # Recompute the test set split as in TrainAgent (stratified by verified)
        from sklearn.model_selection import train_test_split
        # Defensive: ensure "verified" column exists and is correct type
        if "verified" in df_csv.columns:
            stratify_col = df_csv["verified"].astype(str).str.upper().eq("TRUE")
        else:
            stratify_col = None
        _, test_df = train_test_split(
            df_csv, test_size=0.2, random_state=42,
            stratify=stratify_col if stratify_col is not None else None
        )
        # Merge features_df with test_df on timestamp to get test_features
        # We want to sample up to 30 rows per cluster from test set
        test_features = pd.merge(
            features_df,
            test_df[["timestamp"]],
            on="timestamp",
            how="inner"
        )
        test_features["date"] = pd.to_datetime(test_features["timestamp"], unit="s", errors="coerce").dt.date
        # Compute average cluster size in test set
        avg_cluster_size = test_features["cluster_size"].mean()
        # Only keep clusters larger than average
        large_clusters = test_features[test_features["cluster_size"] > avg_cluster_size]
        reports = []
        # For each cluster in test set above average size, sample up to 30 rows
        for cluster_id, group in large_clusters.groupby("cluster_id"):
            group_sample = group.sample(min(len(group), 30), random_state=42)
            # Get the date window for this cluster
            cluster_dates = group_sample["date"].unique()
            # CSV test rows: tweet snippets and timestamps
            csv_lines = []
            for _, row in group_sample.iterrows():
                ts = pd.to_datetime(row["timestamp"], unit="s", errors="coerce").isoformat()
                # Extract tweet text content
                text = row.get("text", "")
                if not text:
                    text = row.get("quotedTweet/text", "")
                text = text.strip()
                if text:
                    snippet = text.replace("\n", " ")
                    csv_lines.append(f"- CSV Tweet: “{snippet[:100]}…” at {ts}")
                else:
                    csv_lines.append(f"- CSV: <no tweet text> at {ts}")
            # Find any matched_web rows whose timestamp date falls within the cluster's date window
            web_lines = []
            if not matched_web.empty and "timestamp" in matched_web.columns:
                matched_web["date"] = pd.to_datetime(matched_web["timestamp"], unit="s", errors="coerce").dt.date
                web_matches = matched_web[matched_web["date"].isin(cluster_dates)]
                for _, wrow in web_matches.iterrows():
                    web_lines.append(
                        f"- Web: timestamp={wrow['timestamp']}, desc={wrow.get('description','')[:50]}"
                    )
            # Compose the bullet format for LLM
            lines = csv_lines + web_lines
            reports.append(
                f"Cluster {cluster_id}:\n" +
                "\n".join(lines)
            )
        prompt = (
            "You are analyzing clusters of customer‐complaint tweets about service outages. "
            "For each cluster, you will see up to 30 sample tweets (in quotes) with their timestamps. "
            "Use these tweet snippets—and any matching web logs—to categorize the cluster under a telecommunications service type, for example: "
            "'Internet issue', 'Landline issue', 'Mobile issue', 'Other'. "
            "Then write summaries using the format:\n\n"
            "Cluster {cluster_id}:\n"
            "- Date: YYYY-MM-DD\n"
            "- Time: describe times\n"
            "- Incident Type: your chosen category\n"
            "- Description: concise summary based on tweet text and web logs. If the Incident Type is 'Other', make sure to print the issue type here.\n\n"
            "If you label a cluster as 'Other', include in the Description the predominant complaint type mentioned in the tweets (e.g. 'internet issue', 'phone issue', etc.). "
            "Do not output any clusters that are not Telecommunications service complaints. Some clusters might be about non-incidents. Do not print them.\n\n"
            "Here are the clusters:\n\n"
            + "\n\n".join(reports) +
            "\n\nPlease summarize accordingly."
        )
        report = llm_llama.invoke(prompt).content
        return {"provider": provider, "report": report}

# For synthetic incident simulation
from scipy.stats import ks_2samp

# ----------------------------------------------------------------------
# 1.  Load CSV Datasets
# ----------------------------------------------------------------------
from pathlib import Path

CSV_DIR = Path("csv_files")
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

#
# Dynamically load all CSV datasets in CSV_DIR with correct provider keys
csv_files_list = list(CSV_DIR.glob("*.csv"))
CSV_MAPPING = {}
for csv_path in csv_files_list:
    name = csv_path.name.lower()
    if "bell" in name:
        prov = "BELL"
    elif "rogers" in name:
        prov = "ROGERS"
    elif "telus" in name:
        prov = "TELUS"
    else:
        prov = csv_path.stem.upper()
    df = _load_csv(csv_path.name, prov)
    CSV_MAPPING[prov] = (csv_path.name, df)

#
# Print distribution of verified vs unverified in each CSV before analysis
# (Removed: not needed for minimal workflow)

#
# 1‑A.  Print unique verified user/url entries
# (Removed: not needed for minimal workflow)

feature_cols = [
    "timestamp",
    "cluster_size",
    "cluster_frequency",
    "inter_arrival",
    "count_1h",
    "hour",
    "weekday",
    "count_1h_z"
]
numeric_cols = feature_cols.copy()

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
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
}

pipelines = {}
for name, clf in classifiers.items():
    pipelines[name] = Pipeline([
        ("prep", preprocessor),
        ("rfe", RFECV(
            estimator=clf,
            step=1,
            cv=3,
            scoring="balanced_accuracy",
            min_features_to_select=5,
            n_jobs=-1
        )),
        ("clf", clf)
    ])

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


# XML loading and xml_to_rows: removed

# === CrewAI Workflow ===
ing_agent = IngestionAgent()
fe_agent = FeAgent()
train1_agent = TrainAgent(test_num=1)
report_agent = ReportAgent()

# Print agent responsibilities
print("\nAgent responsibilities:")
print("IngestionAgent: fetches CSV and web-scraped incidents, normalizes timestamps, filters to CSV test-set dates, and stores matched pairs")
print("FeAgent: computes time-series and clustering features on CSV incidents, with LLM-guided clustering window selection")
print("TrainAgent: balances the training split with SMOTE, uses LLM to choose RandomForest tree count, trains & evaluates the classifier, and saves metrics and charts")
print("ReportAgent: groups matched incidents into time-based clusters, includes CSV and web context, and generates LLM summaries per cluster")

# === Manual Agentic Execution ===
for provider in CSV_MAPPING.keys():
    print(f"\n=== Running agents for {provider} ===")
    # Ingest and merge data
    print("-> Agent: IngestionAgent running")
    ingestion_output = ing_agent.run(provider=provider)
    df_csv = ingestion_output["df_csv"]
    matched_web = ingestion_output["matched_web"]
    # Feature engineering
    print("-> Agent: FeAgent running")
    fe_output = fe_agent.run(provider=provider, df_csv=df_csv)
    features_df = fe_output["features_df"]
    # Training and evaluation (Test 1)
    print("-> Agent: TrainAgent running")
    train_output = train1_agent.run(provider=provider, features_df=features_df)
    # The train_and_evaluate tool already prints metrics and charts
    # LLM report
    print("-> Agent: ReportAgent running")
    report_output = report_agent.run(provider=provider, df_csv=df_csv, matched_web=matched_web, features_df=features_df)
    print("\nLLM Matched-Incidents Summary:")
    if report_output.get("report"):
        print(report_output["report"])
    else:
        print(f"No matched incidents for provider {provider}.")

# === Workflow Visualization ===
from graphviz import Source

# Manually define the agentic workflow graph
dot_source = """
digraph IncidentClassificationPipeline {
    IngestionAgent -> FeAgent -> TrainAndEval -> MatchedReport;
}
"""
graph = Source(dot_source, format="png")
graph.render(filename="agentic_workflow", cleanup=True)
print("Generated agentic workflow visualization: agentic_workflow.png")