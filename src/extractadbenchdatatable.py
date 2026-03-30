import os, json, re
import numpy as np
import pandas as pd

PDF_URL = "https://arxiv.org/pdf/2206.09426"
PDF_PATH = "2206.09426.pdf"
OUTDIR = "tables_D4_D16"

SCORE_RE = re.compile(r'^([\d.]+)\((\d+)\)$')

# Column renames: broken camelot names -> correct model names
COLUMN_FIXES = {
    "col": None,   # resolved per-table below
    "col_1": None,
    "former": "Transformer",
}

# D4/D5 have SVDD and GMM as sub-headers; D6-D16 have Transformer
D4_D5_RENAMES = {"col": "SVDD", "col_1": "GMM"}
D6_PLUS_RENAMES = {"col": "Transformer"}

# Page/table mapping for ADBench paper PDF
SPECS = {
    "D4":  dict(page="37", areas=None, idx=0),
    "D5":  dict(page="38", areas=None, idx=0),
    "D6":  dict(page="39", areas=None, idx=0),
    "D7":  dict(page="39", areas=None, idx=1),
    "D8":  dict(page="40", areas=None, idx=0),
    "D9":  dict(page="40", areas=["0,350,612,0"], idx=0),
    "D10": dict(page="41", areas=None, idx=0),
    "D11": dict(page="41", areas=["0,350,612,0"], idx=0),
    "D12": dict(page="42", areas=None, idx=0),
    "D13": dict(page="42", areas=["0,350,612,0"], idx=0),
    "D14": dict(page="43", areas=None, idx=0),
    "D15": dict(page="43", areas=None, idx=1),
    "D16": dict(page="44", areas=None, idx=0),
}


def _parse_score(val):
    """Extract numeric score from 'score(rank)' string."""
    if not isinstance(val, str):
        return np.nan, np.nan
    val = val.strip()
    if not val or val.lower() in ("n/a(n/a)", "nan", "n/a"):
        return np.nan, np.nan
    m = SCORE_RE.match(val)
    if m:
        return float(m.group(1)), int(m.group(2))
    return np.nan, np.nan


def _fix_columns(df, tid):
    """Fix broken column names from two-row PDF headers."""
    renames = D4_D5_RENAMES if tid in ("D4", "D5") else D6_PLUS_RENAMES
    df = df.rename(columns=renames)
    return df


def _clean_and_split(df):
    """Split score(rank) cells into separate scores and ranks DataFrames."""
    # Drop sub-header row (second row with SVDD/GMM/former fragments)
    if len(df) > 0:
        first_row = df.iloc[0]
        non_empty = sum(1 for v in first_row.values if isinstance(v, str) and v.strip() and not SCORE_RE.match(v.strip()))
        if non_empty > 0 and not SCORE_RE.match(str(first_row.iloc[1]).strip()):
            df = df.iloc[1:]

    # Drop artifact rows
    df = df[df["Datasets"].notna()].copy()
    df = df[df["Datasets"].apply(lambda x: isinstance(x, str) and x.strip() != "")]

    model_cols = [c for c in df.columns if c != "Datasets"]
    scores = df[["Datasets"]].copy()
    ranks = df[["Datasets"]].copy()

    for col in model_cols:
        parsed = df[col].apply(_parse_score)
        scores[col] = parsed.apply(lambda x: x[0])
        ranks[col] = parsed.apply(lambda x: x[1])

    return scores.reset_index(drop=True), ranks.reset_index(drop=True)


def fix_existing_csvs():
    """Post-process already-extracted CSVs: fix headers, parse scores."""
    all_scores = {}

    for tid in SPECS:
        path = os.path.join(OUTDIR, f"{tid}.csv")
        if not os.path.exists(path):
            print(f"  SKIP {tid}: {path} not found")
            continue

        df = pd.read_csv(path, dtype=str)
        df = _fix_columns(df, tid)
        scores_df, ranks_df = _clean_and_split(df)
        all_scores[tid] = scores_df.to_dict(orient="records")

        scores_df.to_csv(os.path.join(OUTDIR, f"{tid}.csv"), index=False)
        ranks_df.to_csv(os.path.join(OUTDIR, f"{tid}.ranks.csv"), index=False)
        print(f"  {tid}: {len(scores_df)} datasets, columns: {list(scores_df.columns)}")

    with open(os.path.join(OUTDIR, "all_tables.json"), "w") as f:
        json.dump(all_scores, f)

    print(f"Done. Output in {OUTDIR}/")


def extract_from_pdf():
    """Full extraction from PDF using camelot."""
    import camelot
    import urllib.request

    # Download PDF if needed
    if not os.path.exists(PDF_PATH):
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)

    # PyPDF2/Camelot compatibility patch
    try:
        from PyPDF2 import PdfReader, PdfWriter
        import camelot.handlers as handlers
        PdfWriter.addPage = PdfWriter.add_page
        PdfReader.isEncrypted = property(lambda self: self.is_encrypted)
        PdfReader.getNumPages = lambda self: len(self.pages)
        PdfReader.numPages = property(lambda self: len(self.pages))
        PdfReader.getPage = lambda self, i: self.pages[i]
        handlers.PdfFileReader = PdfReader
        handlers.PdfFileWriter = PdfWriter
    except ImportError:
        pass

    def _is_subheader_row(row):
        vals = [str(v).strip() for v in row.values]
        non_empty = [v for v in vals if v and v != "nan"]
        return 0 < len(non_empty) <= len(vals) // 2

    def _merge_headers(main, sub):
        merged = []
        for m, s in zip(main, sub):
            m_str = str(m).strip() if m is not None else ""
            s_str = str(s).strip() if s is not None else ""
            if (not m_str or m_str == "nan") and s_str and s_str != "nan":
                merged.append(s_str)
            else:
                merged.append(m_str if m_str and m_str != "nan" else "")
        return merged

    def _make_unique(cols):
        seen = {}
        out = []
        for c in cols:
            c = c if c else "unnamed"
            if c not in seen:
                seen[c] = 0
                out.append(c)
            else:
                seen[c] += 1
                out.append(f"{c}_{seen[c]}")
        return out

    def clean_camelot_df(df):
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        header_idx = None
        for i, row in df.iterrows():
            if any(isinstance(v, str) and v.strip().lower() == "datasets" for v in row.values):
                header_idx = i
                break
        if header_idx is None:
            raise RuntimeError("Could not find header row containing 'Datasets'.")

        header = df.iloc[header_idx].tolist()
        data_start = header_idx + 1

        if data_start < len(df) and _is_subheader_row(df.iloc[data_start]):
            sub = df.iloc[data_start].tolist()
            header = _merge_headers(header, sub)
            data_start += 1

        header = ["Transformer" if h == "former" else h for h in header]
        out = df.iloc[data_start:].copy()
        out.columns = _make_unique(header)
        out = out.replace({"": np.nan, "nan": np.nan}).dropna(how="all").dropna(axis=1, how="all")
        out = out[out["Datasets"].notna() & (out["Datasets"].str.strip() != "")]

        model_cols = [c for c in out.columns if c != "Datasets"]
        scores = out[["Datasets"]].copy()
        ranks = out[["Datasets"]].copy()
        for col in model_cols:
            parsed = out[col].apply(_parse_score)
            scores[col] = parsed.apply(lambda x: x[0])
            ranks[col] = parsed.apply(lambda x: x[1])
        return scores.reset_index(drop=True), ranks.reset_index(drop=True)

    os.makedirs(OUTDIR, exist_ok=True)
    all_scores = {}

    for tid, spec in SPECS.items():
        kwargs = {}
        if spec["areas"] is not None:
            kwargs["table_areas"] = spec["areas"]
        t = camelot.read_pdf(PDF_PATH, pages=spec["page"], flavor="stream", **kwargs)
        scores_df, ranks_df = clean_camelot_df(t[spec["idx"]].df)
        all_scores[tid] = scores_df.to_dict(orient="records")
        scores_df.to_csv(os.path.join(OUTDIR, f"{tid}.csv"), index=False)
        ranks_df.to_csv(os.path.join(OUTDIR, f"{tid}.ranks.csv"), index=False)
        print(f"  {tid}: {len(scores_df)} datasets, columns: {list(scores_df.columns)}")

    with open(os.path.join(OUTDIR, "all_tables.json"), "w") as f:
        json.dump(all_scores, f)
    print(f"Done. Output in {OUTDIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract ADBench benchmark tables D4-D16")
    parser.add_argument("--from-pdf", action="store_true", help="Re-extract from PDF (requires camelot)")
    args = parser.parse_args()

    if args.from_pdf:
        print("Extracting from PDF...")
        extract_from_pdf()
    else:
        print("Fixing existing CSVs...")
        fix_existing_csvs()
