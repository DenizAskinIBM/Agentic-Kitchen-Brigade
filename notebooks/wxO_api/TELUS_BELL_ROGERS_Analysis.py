import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------------------
# We train with both the Rogers dataset and the newly‑provided Bell dataset.

df_rogers = pd.read_csv("rogers.csv", dtype=str).fillna("")
df_rogers["dataset_source"] = "ROGERS"

df_bell = pd.read_csv("bell_data.csv", dtype=str).fillna("")
df_bell["dataset_source"] = "BELL"

df_telus = pd.read_csv("telus.csv", dtype=str).fillna("")
df_telus["dataset_source"] = "TELUS"

# -----------------------------------------------------------------------------
# Print verified user/url entries (unique) per provider
# -----------------------------------------------------------------------------
def _print_verified_urls(df, provider_name):
    """
    Print the unique values from the 'user/url' column for rows whose
    'verified' flag is TRUE.  If necessary columns are missing, we skip.
    """
    if "verified" not in df.columns or "user/url" not in df.columns:
        print(f"{provider_name}: columns 'verified' or 'user/url' missing.")
        return

    urls = (
        df[df["verified"].astype(str).str.strip().str.upper() == "TRUE"]["user/url"]
        .dropna()
        .unique()
        .tolist()
    )

    print(f"\n{provider_name} – verified user/url entries ({len(urls)} unique):")
    for url in urls:
        print(f"  {url}")

_print_verified_urls(df_rogers, "ROGERS")
_print_verified_urls(df_bell,   "BELL")
_print_verified_urls(df_telus,  "TELUS")

# Combine them into a single dataframe
df = pd.concat([df_rogers, df_bell, df_telus], axis=0, ignore_index=True)

# -----------------------------------------------------------------------------
# 2. Define Target (y) and Features (X)
# -----------------------------------------------------------------------------
# The target is the "verified" column, which we'll treat as binary (True/False).
# Convert it to 1/0 for easier modeling.
y = df["verified"].apply(lambda x: 1 if x.strip().upper() == "TRUE" else 0)

# For demonstration, let's pick a subset of columns that might carry signal.
# In practice, you would experiment with more columns or text fields.
feature_cols = [
    "likes",            # numeric
    "isReply",          # boolean-like
    "isRetweet",        # boolean-like
    "text",             # text
    "searchQuery",      # categorical-like
    "user/location",    # categorical-like
    "user/totalFollowers", 
    "user/totalFollowing"
]

# Create X from these features. 
# (Any columns not in the CSV should be removed from this list or created if needed.)
X = df[feature_cols].copy()

# Keep track of which provider each row came from so we can
# report accuracy separately later.
source = df["dataset_source"]

# -----------------------------------------------------------------------------
# 3. Preprocessing
# -----------------------------------------------------------------------------
# We'll split the columns into:
#   - numeric_cols: will be converted to float and scaled
#   - bool_cols: will be converted to 0/1
#   - text_cols: we'll do a TfidfVectorizer
#   - cat_cols: we'll do one-hot encoding
#
# Because everything in df is read as string (dtype=str), we need to convert them.

numeric_cols = ["likes", "user/totalFollowers", "user/totalFollowing"]
bool_cols = ["isReply", "isRetweet"]
text_cols = ["text"]
cat_cols = ["searchQuery", "user/location"]

def to_numeric(x):
    """
    Convert any dataframe/array‐like selection to a numeric numpy array.

    * If `x` is a pandas DataFrame/Series we apply `pd.to_numeric`
      column‑wise, coercing errors to NaN, then fill NaNs with 0.
    * If `x` is a NumPy ndarray we first turn it into a DataFrame for
      convenience, apply the same conversion, and return the underlying
      values.

    The shape of the returned array always matches the original 2‑D
    selection so that ColumnTransformer keeps the correct column count.
    """
    if isinstance(x, pd.Series):
        x = x.to_frame()

    if isinstance(x, pd.DataFrame):
        return (
            x.apply(pd.to_numeric, errors="coerce")
             .fillna(0)
             .values
        )

    # fall‑back for ndarray or other array‑like
    x_df = pd.DataFrame(x)
    return (
        x_df.apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values
    )

def to_bool(x):
    """
    Convert boolean‑like text to 0/1 numeric array.
    Accepts DataFrame, Series or ndarray.
    """
    truthy = {"TRUE", "YES", "1", "T", "Y"}
    falsy  = {"FALSE", "NO", "0", "F", "N"}

    if isinstance(x, pd.Series):
        x = x.to_frame()

    if isinstance(x, pd.DataFrame):
        return (
            x.apply(
                lambda col: col.astype(str)
                            .str.strip()
                            .str.upper()
                            .map(lambda v: 1 if v in truthy else 0),
                axis=0
            ).values
        )

    # ndarray or other
    flat = pd.Series(x.ravel().astype(str))
    out = flat.str.strip().str.upper().map(lambda v: 1 if v in truthy else 0)
    return out.values.reshape(x.shape[0], -1)

numeric_transformer = Pipeline([
    ("to_numeric", FunctionTransformer(to_numeric)),
    ("scaler", StandardScaler())
])

bool_transformer = Pipeline([
    ("to_bool", FunctionTransformer(to_bool)) 
    # We could add a scaler if desired, but 0/1 typically doesn't need scaling
])

 # Text needs to be a 1‑D iterable (list/Series) for TfidfVectorizer.
 # ColumnTransformer passes a 2‑D DataFrame; we first squeeze it.
text_transformer = Pipeline([
    (
        "extract",
        FunctionTransformer(
            lambda x: x.squeeze(axis=1) if isinstance(x, pd.DataFrame) else x,
            validate=False
        )
    ),
    (
        "tfidf",
        TfidfVectorizer(
            max_features=500,        # you can tweak this
            stop_words="english"
        )
    )
])

cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_cols),
    ("bool", bool_transformer, bool_cols),
    ("text", text_transformer, text_cols),
    ("cat", cat_transformer, cat_cols)
])

# -----------------------------------------------------------------------------
# 4. Modeling Pipeline
# -----------------------------------------------------------------------------
# We'll use a RandomForestClassifier to see which features matter most.
# Note that the text transformer can produce many features, so the
# feature_importances_ array will be large. We'll still demonstrate
# how to extract and print them.

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# -----------------------------------------------------------------------------
# 5. Train/Test Split and Fit
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test, source_train, source_test = train_test_split(
    X,
    y,
    source,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 6. Evaluate and Print Feature Importances
# -----------------------------------------------------------------------------
# We'll look at the Random Forest's feature_importances_ and map them back
# to the transformed feature names. Note that TfidfVectorizer and OneHotEncoder
# each produce multiple columns.
#
# We can extract the list of all transformed feature names using the 
# ColumnTransformer and its transformers_ in combination with 
# 'get_feature_names_out' (for TfidfVectorizer and OneHotEncoder pipelines).

# Get the final RandomForestClassifier from the pipeline
rf = model.named_steps["classifier"]

# Get the preprocessor
ct = model.named_steps["preprocessor"]

# We'll collect feature names from each sub-transformer in the same order 
# that the column transformer produces them.
feature_names = []

# 1) Numeric columns -> single output each
for col in numeric_cols:
    feature_names.append(col)

# 2) Bool columns -> single output each
for col in bool_cols:
    feature_names.append(col)

# 3) Text columns (TfidfVectorizer) -> get_feature_names_out
#    we just do it for each text col set, but typically it's just one text column.
text_pipeline = ct.named_transformers_["text"].named_steps["tfidf"]
if hasattr(text_pipeline, "get_feature_names_out"):
    tfidf_features = text_pipeline.get_feature_names_out()
    # They will appear as 'word' or similar. 
    feature_names.extend(tfidf_features)
else:
    # If there's no get_feature_names_out, just skip
    feature_names.extend([f"text_{i}" for i in range(text_pipeline.shape[1])])

# 4) Cat columns (OneHotEncoder) -> get_feature_names_out
cat_pipeline = ct.named_transformers_["cat"]
if hasattr(cat_pipeline, "get_feature_names_out"):
    ohe_features = cat_pipeline.get_feature_names_out(cat_cols)
    feature_names.extend(ohe_features)
else:
    # If there's no get_feature_names_out, just name them generically
    cat_count = cat_pipeline.shape[1]
    feature_names.extend([f"cat_{i}" for i in range(cat_count)])

# Now, feature_importances_ should match the length of 'feature_names'
importances = rf.feature_importances_

# Pair them and sort by importance
feature_importance_pairs = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)

# Print the 20 most important features BEFORE reporting accuracies
print("\nMost Important 20 Features:")
for feature, score in feature_importance_pairs[:20]:
    print(f"{feature}: {score:.4f}")

# -----------------------------------------------------------------------------
# 7. Quick Accuracy Check
# -----------------------------------------------------------------------------
accuracy_overall = model.score(X_test, y_test)

# Evaluate on each provider separately
mask_bell  = source_test == "BELL"
mask_rog   = source_test == "ROGERS"
mask_telus = source_test == "TELUS"

accuracy_bell  = model.score(X_test[mask_bell],  y_test[mask_bell])  if mask_bell.any()  else float("nan")
accuracy_rog   = model.score(X_test[mask_rog],   y_test[mask_rog])   if mask_rog.any()   else float("nan")
accuracy_telus = model.score(X_test[mask_telus], y_test[mask_telus]) if mask_telus.any() else float("nan")

print(f"\nOverall Model Accuracy on Test Set: {accuracy_overall:.3f}")
print(f"Model Accuracy on BELL  Test Subset: {accuracy_bell:.3f}")
print(f"Model Accuracy on ROGERS Test Subset: {accuracy_rog:.3f}")
print(f"Model Accuracy on TELUS  Test Subset: {accuracy_telus:.3f}")