"""
distribution_similarity.py

Fungsi:
- Load merged CSV (harus berisi title_clean_sim / title_clean dan content_clean)
- Jika kolom similarity ada (cosine_similarity / bert_similarity / title_clean_sim), gunakan.
  Jika tidak ada, hitung similarity memakai sentence-transformers.
- Plot histogram + KDE + mean lines + boxplot untuk Clickbait vs Non-Clickbait.
- Save plot ke file PNG dan cetak ringkasan statistik.

Cara pakai:
python distribution_similarity.py --input merged_for_similarity_full.csv --out fig_similarity.png

Dependencies:
pip install pandas matplotlib seaborn numpy sentence-transformers tqdm
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detect_label_column(cols):
  candidates = ['label', 'is_clickbait', 'is_clickbait_auto', 'is_clickbait_bert', 'is_clickbait_manual']
  for c in candidates:
    if c in cols:
      return c
  # fallback: try 'y' or 'target'
  for c in ['y','target']:
    if c in cols:
      return c
  return None

def detect_similarity_column(cols):
  candidates = ['cosine_similarity','bert_similarity','title_content_sim','similarity','title_clean_sim']
  for c in candidates:
    if c in cols:
      return c
  return None

def compute_similarity_sentence_transformer(df, title_col, content_col, model_name="all-MiniLM-L6-v2", batch_size=64):
  """
  Hitung cosine similarity title vs content menggunakan sentence-transformers.
  Mengembalikan numpy array similarity (float64).
  """
  try:
    from sentence_transformers import SentenceTransformer, util
  except Exception as e:
    raise RuntimeError("Install `sentence-transformers` (pip install sentence-transformers) untuk menghitung similarity.") from e

  model = SentenceTransformer(model_name)
  titles = df[title_col].astype(str).tolist()
  contents = df[content_col].astype(str).tolist()

  sims = []
  # batch encoding (memory-friendly)
  for i in range(0, len(titles), batch_size):
    t_batch = titles[i:i+batch_size]
    c_batch = contents[i:i+batch_size]
    emb_t = model.encode(t_batch, convert_to_tensor=True)
    emb_c = model.encode(c_batch, convert_to_tensor=True)
    # cosine per pair
    cos = util.cos_sim(emb_t, emb_c).diagonal().cpu().numpy()
    sims.append(cos)
  sims = np.concatenate(sims, axis=0)
  return sims

def main(input_csv, output_png, title_col_hint=None, content_col_hint=None, similarity_col_hint=None, label_col_hint=None):
  df = pd.read_csv(input_csv, encoding='utf-8')

  # detect columns
  cols = df.columns.tolist()
  title_candidates = [title_col_hint] if title_col_hint else ['title_clean_sim','title_clean','title','title_clean_sim']
  content_candidates = [content_col_hint] if content_col_hint else ['content_clean','content','content_clean_sim','isi','body']
  title_col = next((c for c in title_candidates if c in cols), None)
  content_col = next((c for c in content_candidates if c in cols), None)
  similarity_col = similarity_col_hint if similarity_col_hint in cols else detect_similarity_column(cols) if not similarity_col_hint else (similarity_col_hint if similarity_col_hint in cols else None)
  label_col = label_col_hint if (label_col_hint and label_col_hint in cols) else detect_label_column(cols)

  if title_col is None or content_col is None:
    raise ValueError(f"Could not find title/content columns. Candidates checked. Available columns: {cols[:20]}")

  print("Using columns -> title:", title_col, ", content:", content_col)
  if label_col:
    print("Detected label column:", label_col)
  else:
    print("Label column not found automatically. Script will try to infer by 'is_clickbait' numeric or request user to provide label column.")
  
  # compute or pick similarity
  if similarity_col and similarity_col in cols:
    print("Found similarity column:", similarity_col, "— using it.")
    df['similarity'] = df[similarity_col].astype(float)
  else:
    print("No precomputed similarity column found — computing with SentenceTransformer (this may take time).")
    sims = compute_similarity_sentence_transformer(df, title_col, content_col, model_name="all-MiniLM-L6-v2", batch_size=64)
    df['similarity'] = sims

  # ensure label column: try to coerce booleans/numeric strings
  if not label_col:
    # try heuristics: find a column that looks binary
    for c in cols:
      if df[c].dropna().isin([0,1]).all():
        label_col = c
        print("Heuristic label column chosen:", label_col)
        break

  if not label_col:
    raise ValueError("No label column found. Provide label column name via --label argument.")

  # normalize label to 0/1 ints
  df[label_col] = df[label_col].apply(lambda x: 1 if str(x).strip() in ['1','True','true','clickbait','cb'] else 0)

  # filter out empty texts maybe
  df = df[(df[title_col].astype(str).str.strip() != "") & (df[content_col].astype(str).str.strip() != "")]
  print("Records used for plotting:", len(df))

  # Plotting: histogram + KDE + boxplot
  sns.set(style="whitegrid")
  plt.figure(figsize=(10,6))
  # histograms (density)
  sns.histplot(df[df[label_col]==1]['similarity'], bins=40, stat='density', color='tomato', label='Clickbait', kde=True, alpha=0.5)
  sns.histplot(df[df[label_col]==0]['similarity'], bins=40, stat='density', color='steelblue', label='Non-Clickbait', kde=True, alpha=0.5)

  # mean lines
  mean_cb = df[df[label_col]==1]['similarity'].mean()
  mean_ncb = df[df[label_col]==0]['similarity'].mean()
  plt.axvline(mean_cb, color='darkred', linestyle='--', label=f'Clickbait mean = {mean_cb:.3f}')
  plt.axvline(mean_ncb, color='navy', linestyle='--', label=f'Non-clickbait mean = {mean_ncb:.3f}')

  plt.title("Distribusi Similarity Judul–Isi: Clickbait vs Non-Clickbait")
  plt.xlabel("Similarity (cosine, embedding)")
  plt.ylabel("Density")
  plt.legend()
  plt.tight_layout()
  plt.savefig(output_png.replace('.png','_hist.png'), dpi=300)

  # boxplot
  plt.figure(figsize=(8,4))
  sns.boxplot(x=label_col, y='similarity', data=df.replace({label_col:{1:'Clickbait',0:'Non-Clickbait'}}))
  plt.title("Boxplot Similarity per Kategori")
  plt.xlabel("")
  plt.ylabel("Similarity")
  plt.tight_layout()
  plt.savefig(output_png.replace('.png','_box.png'), dpi=300)

  # save summary stats CSV
  stats = df.groupby(label_col)['similarity'].describe().T
  stats.to_csv(output_png.replace('.png','_summary_stats.csv'))

  print("Plots saved:", output_png.replace('.png','_hist.png'), output_png.replace('.png','_box.png'))
  print("Summary stats saved:", output_png.replace('.png','_summary_stats.csv'))
  return df, mean_cb, mean_ncb

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", "-i", required=True, help="Path to merged CSV (with title & content)")
  parser.add_argument("--output", "-o", default="similarity_distribution.png", help="Base name for output PNG files")
  parser.add_argument("--title-col", default=None, help="Optional: title column name")
  parser.add_argument("--content-col", default=None, help="Optional: content column name")
  parser.add_argument("--similarity-col", default=None, help="Optional: precomputed similarity column name")
  parser.add_argument("--label", default=None, help="Optional: label column name (binary 0/1)")
  args = parser.parse_args()
  main(args.input, args.output, title_col_hint=args.title_col, content_col_hint=args.content_col,
    similarity_col_hint=args.similarity_col, label_col_hint=args.label)
