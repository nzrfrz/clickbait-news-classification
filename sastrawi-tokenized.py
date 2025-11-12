import ast
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

df = pd.read_csv("./text_tokenized.csv")

def parse_tokens(x):
  if isinstance(x, list):
    return x
  if not isinstance(x, str):
    return []
  try:
    val = ast.literal_eval(x)
    if isinstance(val, list):
      return [str(v) for v in val]
  except:
    pass
  x = x.strip("[]")
  return [t.strip(" '\"") for t in x.split(",") if t.strip()]

df["title_tokens"] = df["title_tokenized"].apply(parse_tokens)
df["content_tokens"] = df["content_tokenized"].apply(parse_tokens)

# Sastrawi stopword + stemmer
stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())

def clean_with_sastrawi(tokens):
  tokens = [t.lower() for t in tokens if isinstance(t,str)]
  # remove stopwords
  tokens = [t for t in tokens if t not in stopwords]
  # stemming
  tokens = [stemmer.stem(t) for t in tokens]
  # remove empty and single non-numeric char
  tokens = [t for t in tokens if t and not(len(t)==1 and not t.isdigit())]
  return tokens

df["title_tokens_sastrawi"] = df["title_tokens"].apply(clean_with_sastrawi)
df["content_tokens_sastrawi"] = df["content_tokens"].apply(clean_with_sastrawi)

df_out = df[["title_clean","content_clean","title_tokens_sastrawi","content_tokens_sastrawi"]]

df_out.to_csv("text_tokenized_sastrawi.csv", index=False)

df_sample = df_out.sample(5, random_state=42)  # untuk reproducibility
df_sample.to_json("text_tokenized_sastrawi_sample.json", orient="records", lines=True, force_ascii=False)

print("✅ Selesai! Hasil disimpan sebagai text_tokenized_sastrawi.csv & .json")