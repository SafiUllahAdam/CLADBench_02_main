import os, json, zipfile
import numpy as np
import pandas as pd
import camelot

PDF_URL = "https://arxiv.org/pdf/2206.09426"
PDF_PATH = "2206.09426.pdf"
OUTDIR = "tables_D4_D16"

# --- 1) Download (works in normal Python environments)
import urllib.request
urllib.request.urlretrieve(PDF_URL, PDF_PATH)

# --- 2) PyPDF2/Camelot compatibility patch (needed in some environments)
# If you DON'T get errors, you can remove this whole block.
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
import camelot.handlers as handlers

PdfWriter.addPage = PdfWriter.add_page
PdfReader.isEncrypted = property(lambda self: self.is_encrypted)
PdfReader.getNumPages = lambda self: len(self.pages)
PdfReader.numPages = property(lambda self: len(self.pages))
PdfReader.getPage = lambda self, i: self.pages[i]
handlers.PdfFileReader = PdfReader
handlers.PdfFileWriter = PdfWriter

def make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        c = "col" if c is None or str(c).strip() == "" else str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out

def clean_camelot_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # header row = the one containing "Datasets"
    header_idx = None
    for i, row in df.iterrows():
        if any(isinstance(v, str) and v.strip().lower() == "datasets" for v in row.values):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find header row containing 'Datasets'.")

    header = df.iloc[header_idx].tolist()
    out = df.iloc[header_idx + 1 :].copy()
    out.columns = make_unique(header)

    out = out.replace({"": np.nan, "nan": np.nan}).dropna(how="all").dropna(axis=1, how="all")
    return out

# Table mapping for this specific PDF:
# - D4..D5 single tables on pages 37..38
# - D6/D7 on page 39 (two tables)
# - D8/D9 on page 40 (needs area split)
# - D10/D11 on page 41 (needs area split)
# - D12/D13 on page 42 (needs area split)
# - D14/D15 on page 43 (two tables)
# - D16 on page 44 (first table; D17 is AUCPR and not requested)
#
# NOTE: page numbers here are PDF pages (1-indexed).
SPECS = {
    "D4":  dict(page="37", areas=None, idx=0),
    "D5":  dict(page="38", areas=None, idx=0),
    "D6":  dict(page="39", areas=None, idx=0),
    "D7":  dict(page="39", areas=None, idx=1),
    "D8":  dict(page="40", areas=None, idx=0),
    "D9":  dict(page="40", areas=["0,350,612,0"], idx=0),   # bottom half
    "D10": dict(page="41", areas=None, idx=0),
    "D11": dict(page="41", areas=["0,350,612,0"], idx=0),   # bottom half
    "D12": dict(page="42", areas=None, idx=0),
    "D13": dict(page="42", areas=["0,350,612,0"], idx=0),   # bottom half
    "D14": dict(page="43", areas=None, idx=0),
    "D15": dict(page="43", areas=None, idx=1),
    "D16": dict(page="44", areas=None, idx=0),
}

os.makedirs(OUTDIR, exist_ok=True)
all_tables = {}

for tid, spec in SPECS.items():
    kwargs = {}
    if spec["areas"] is not None:
        kwargs["table_areas"] = spec["areas"]

    t = camelot.read_pdf(PDF_PATH, pages=spec["page"], flavor="stream", **kwargs)
    df = clean_camelot_df(t[spec["idx"]].df)
    all_tables[tid] = df.to_dict(orient="records")

    df.to_csv(os.path.join(OUTDIR, f"{tid}.csv"), index=False)
    with open(os.path.join(OUTDIR, f"{tid}.json"), "w") as f:
        json.dump(all_tables[tid], f)

with open(os.path.join(OUTDIR, "all_tables.json"), "w") as f:
    json.dump(all_tables, f)

# Optional: zip outputs
zip_name = "tables_D4_D16.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(OUTDIR):
        z.write(os.path.join(OUTDIR, fn), arcname=os.path.join(OUTDIR, fn))

print("Done:", zip_name)