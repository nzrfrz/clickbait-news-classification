import pandas as pd
import ast
import re

# Paths (uploaded by user)
path_sim = "./text_with_similarity_labels.csv"
path_bert = "./text_with_bert_labels.csv"
path_lex = "./sensational_lex.csv"

# Output paths
out_csv = "./new_preprocessed_final.csv"
out_json = "./new_preprocessed_final_sample.json"

# Load datasets
sim = pd.read_csv(path_sim)
bert = pd.read_csv(path_bert)
lex = pd.read_csv(path_lex)

# Inspect columns to choose merge keys
sim_cols = sim.columns.tolist()
bert_cols = bert.columns.tolist()

# Try to find a stable key: prefer URL, else title
key_candidates = []
for c in ["url", "link", "id", "article_id"]:
  if c in sim.columns and c in bert.columns:
    key_candidates.append(c)
if not key_candidates:
  for c in sim.columns:
    if c.lower() in ["title","judul"] and c in bert.columns:
      key_candidates.append(c)

# Fallback: add a row index to align if no common key found
if not key_candidates:
  sim["__row_id__"] = range(len(sim))
  bert["__row_id__"] = range(len(bert))
  key = "__row_id__"
else:
  key = key_candidates[0]

# Merge
merged = pd.merge(sim, bert, on=key, how="inner", suffixes=("_sim","_bert"))

# Build sensational lexicon set
if "term" not in lex.columns:
  raise ValueError("Kolom 'term' tidak ditemukan pada sensational_lex.csv")
lex['term'] = lex['term'].astype(str).str.strip().str.lower()
sens_terms = set(t for t in lex['term'].tolist() if t and not t.isdigit() and len(t) > 1)

# Helper: tokenize words
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
def to_tokens(text):
  if not isinstance(text, str):
    return []
  return [t.lower() for t in WORD_RE.findall(text)]

# Determine title columns: prefer raw title if exists, else title_clean
title_raw_col = None
for c in merged.columns:
  if c.lower() in ["title","judul","headline"]:
    title_raw_col = c
    break
title_clean_col = None
for c in merged.columns:
  if "title_clean" in c:
    title_clean_col = c
    break

# Choose source texts
title_for_tokens = title_clean_col if title_clean_col else (title_raw_col or key)
title_for_punct = title_raw_col if title_raw_col else (title_clean_col or title_for_tokens)

# --- Features ---
# Cosine similarity (TF-IDF) from sim file: look for known names
cosine_tfidf_col = None
for c in merged.columns:
  if "cosine" in c.lower() and "bert" not in c.lower():
    cosine_tfidf_col = c
    break
token_overlap_col = None
for c in merged.columns:
  if "overlap" in c.lower():
    token_overlap_col = c
    break

# Cosine BERT
cosine_bert_col = None
for c in merged.columns:
  if "bert_similarity" in c.lower() or ("cosine" in c.lower() and "bert" in c.lower()):
    cosine_bert_col = c
    break

# Compute additional features
merged["__title_tokens__"] = merged[title_for_tokens].apply(to_tokens)
merged["title_length"] = merged["__title_tokens__"].apply(len)
merged["sensational_word_count"] = merged["__title_tokens__"].apply(lambda toks: sum(1 for t in toks if t in sens_terms))

# Punctuation features from RAW title if available
PUNCT_RE = re.compile(r'[?!…:;\"\'“”‘’]')
def punct_count(text):
  if not isinstance(text, str):
    return 0
  return len(PUNCT_RE.findall(text))

merged["punctuation_count"] = merged[title_for_punct].apply(punct_count)
merged["punctuation_ratio"] = merged.apply(
  lambda r: (r["punctuation_count"] / max(1, len(str(r[title_for_punct])))), axis=1
)

# Build final columns
final_cols = [key]
if title_raw_col: final_cols.append(title_raw_col)
if title_clean_col and title_clean_col not in final_cols: final_cols.append(title_clean_col)

# similarity
if cosine_tfidf_col: final_cols.append(cosine_tfidf_col)
if token_overlap_col: final_cols.append(token_overlap_col)
if cosine_bert_col: final_cols.append(cosine_bert_col)

# extras
final_cols += ["title_length","sensational_word_count","punctuation_count","punctuation_ratio"]

# labels if exist
for c in ["is_clickbait_auto","is_clickbait_bert","is_clickbait","label","target"]:
  if c in merged.columns and c not in final_cols:
    final_cols.append(c)

final = merged[final_cols].copy()

# Save outputs
final.to_csv(out_csv, index=False)
final.sample(10, random_state=42).to_json(out_json, orient="records", lines=True, force_ascii=False)

print("✅ Selesai! Hasil disimpan sebagai new_preprocessed_final_sample.csv & .json")