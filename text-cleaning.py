import pandas as pd

df = pd.read_csv('./detik_news_dropped.csv')

def clean_text_series(s: pd.Series) -> pd.Series:
  return (
    s.fillna("").astype(str)
    .str.lower()
    .str.replace(r'\\"', '', regex=True)   # unescape quotes
    .str.replace(r'\\/', '/', regex=True)   # unescape slash
    .str.replace(r'\\(?=[A-Za-z0-9\(\)\/\-\_])', '', regex=True) # NEW: remove remaining escape backslashes
    .str.replace(r'https?://\S+|www\.\S+', '', regex=True)  # URLs
    .str.replace(r'<[^>]+>', '', regex=True)  # HTML
    .str.replace(r'(?i)\[(?:gambas|foto|video|dok|infografis)(?::[^\]]*)?\]', '', regex=True)  # [Gambas:..]
    .str.replace(r'(?i)\bscroll\s+to\s+continue\s+with\s+content\b', '', regex=True)  # scroll CTA
    .str.replace(r'(?im)^\s*(baca\s+juga|simak\s+juga|simak\s+video|lihat\s+juga)\s*[:：-]?\s*.*$', '', regex=True)  # readmore CTA
    .str.replace(r'(?im)^\s*[a-z][a-z .()/-]{1,30}\s[-–]\s', '', regex=True)  # city dash start
    .str.replace(r'(?im)^\s*[a-z .()/-]{2,30},\s*(senin|selasa|rabu|kamis|jumat|jum\'at|sabtu|minggu)\s*\(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}\),?\s*', '', regex=True)  # city, weekday (date)
    .str.replace(r'\xa0', ' ', regex=True)  # NBSP
    .str.replace(r'\b\d{1,3}(?:[.,]\d{3})+\b', '', regex=True)  # thousand grouping
    .str.replace(r'[ \t\f\v]+', ' ', regex=True)  # collapse horizontal whitespace
    .str.replace(r'[ \t]*\n[ \t]*', '\n', regex=True)  # trim per-line
    .str.replace(r'\s+', ' ', regex=True)  # collapse to single spaces
    .str.strip()
  )

df['title_clean'] = clean_text_series(df['title'])
df['content_clean'] = clean_text_series(df['content'])

df.to_csv('./celaned_text.csv', index=False)
df.to_json('./celaned_text.json', orient="records", lines=False)
print("file exported....")