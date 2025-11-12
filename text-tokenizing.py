import re
import pandas as pd

TEXT_FILE = "./celaned_text.csv"

df = pd.read_csv(TEXT_FILE)

assert 'title_clean' in df.columns and 'content_clean' in df.columns, "Kolom title_clean dan content_clean tidak ditemukan."

# Simple, robust tokenizer for Indonesian/English lowercase text:
# - keep a-z and 0-9
# - split on non-alphanumerics
# - drop empty tokens and 1-char punctuation leftovers
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

def tokenize(text: str):
  if not isinstance(text, str):
    return []
  return TOKEN_RE.findall(text.lower())

# Tokenize
df['title_tokenized'] = df['title_clean'].apply(tokenize)
df['content_tokenized'] = df['content_clean'].apply(tokenize)

# Save
df.to_csv('./text_tokenized.csv', index=False)
df.to_json('./text_tokenized.json', orient="records", force_ascii=False)
print("text tokenized saved...")