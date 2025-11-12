import re
import numpy as np
import pandas as pd
from collections import Counter

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

DATA_TO_EXTRACT = "./detik_news_clickbait_title_based.csv"

# 1) Load
df = pd.read_csv(DATA_TO_EXTRACT)

# 2) Infer columns
title_candidates = [c for c in df.columns if any(k in c.lower() for k in ["title","headline","judul"])]
label_candidates = [c for c in df.columns if any(k in c.lower() for k in ["label","clickbait","is_clickbait","class","target","kategori"])]

if not title_candidates:
  raise ValueError("Tidak menemukan kolom judul/headline. Pastikan ada kolom seperti 'title' atau 'headline'.")
if not label_candidates:
  raise ValueError("Tidak menemukan kolom label. Pastikan ada kolom seperti 'label' atau 'clickbait'.")

title_col = title_candidates[0]
label_col = label_candidates[0]

# 3) Normalize label
def normalize_label(x):
  if pd.isna(x):
    return np.nan
  s = str(x).strip().lower()
  if s in {"1","true","clickbait","ya","yes","y"}:
    return 1
  if s in {"0","false","non-clickbait","bukan","tidak","no","n","non clickbait","non"}:
    return 0
  try:
    v = float(s)
    return 1 if v >= 0.5 else 0
  except:
    return np.nan
  
df["label_bin"] = df[label_col].apply(normalize_label)
df = df.dropna(subset=["label_bin"]).copy()
df["label_bin"] = df["label_bin"].astype(int)

# Sastrawi stopword + stemmer
stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def preprocess_title(text):
  if not isinstance(text, str):
    return []
  text = text.lower().strip()
  # 1) stem kalimat penuh (Sastrawi efektif di level kalimat)
  text = stemmer.stem(text)
  # 2) tokenisasi kata
  toks = TOKEN_RE.findall(text)
  # 3) buang stopword & token numerik/1-char
  toks = [t for t in toks if t not in stopwords and not t.isdigit() and len(t) > 1]
  return toks

df["tokens"] = df[title_col].apply(preprocess_title)

# 5) Term frequency per class
cb_tokens = [t for toks in df.loc[df["label_bin"]==1, "tokens"] for t in toks]
non_tokens = [t for toks in df.loc[df["label_bin"]==0, "tokens"] for t in toks]

cb_counts = Counter(cb_tokens)
non_counts = Counter(non_tokens)

vocab = set(cb_counts.keys()) | set(non_counts.keys())
N_cb = sum(cb_counts.values())
N_non = sum(non_counts.values())
V = len(vocab)
alpha = 1.0

rows = []
for term in vocab:
  f_cb = cb_counts.get(term, 0)
  f_non = non_counts.get(term, 0)
  p_cb = (f_cb + alpha) / (N_cb + alpha * V)
  p_non = (f_non + alpha) / (N_non + alpha * V)
  log_odds = np.log(p_cb / p_non)
  dominance = p_cb - p_non
  ratio = (f_cb + 1.0) / (f_non + 1.0)
  rows.append({
    "term": term,
    "freq_clickbait": f_cb,
    "freq_nonclickbait": f_non,
    "p_clickbait": p_cb,
    "p_nonclickbait": p_non,
    "log_odds": log_odds,
    "dominance": dominance,
    "ratio_cb_non": ratio
  })

lex_df = pd.DataFrame(rows)
lex_df = lex_df[lex_df["freq_clickbait"] >= 3].copy()
lex_df.sort_values(["log_odds","freq_clickbait"], ascending=[False, False], inplace=True)
lex_df.reset_index(drop=True, inplace=True)

lex_df.to_csv('./sensational_lex.csv', index=False)
lex_df.head(50).to_json('./sensational_lex.json', orient="records", lines=True, force_ascii=False)

print("✅ Selesai! Hasil disimpan sebagai sensational_lex.csv & .json")