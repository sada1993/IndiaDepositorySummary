import re
import pdfplumber
import pandas as pd
import pathlib

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
pdf_path    = "NSDLe-CAS_104568374_APR_2025 (1)-unlocked.pdf"   # input CAS PDF
output_path = "equities_only.xlsx"                              # Excel to write

# ----------------------------------------------------------------------
# REGEX HELPERS
# ----------------------------------------------------------------------
numeric_re = re.compile(r"[0-9][0-9,]*\.?[0-9]*")
isin_re    = re.compile(r"^IN[A-Z0-9]{10}$")

# ----------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------
def extract_company(tokens, start_idx):
    """
    Build company name from tokens[start_idx:] until we hit a token
    containing any digit.  Removes trailing '#' if present.
    """
    comp_tokens = []
    for tok in tokens[start_idx:]:
        if re.search(r"\d", tok):
            break
        comp_tokens.append(tok)
    if comp_tokens and comp_tokens[-1] == "#":
        comp_tokens = comp_tokens[:-1]
    return " ".join(comp_tokens).strip()

def parse_accounts(lines):
    """
    Slice the PDF lines into account blocks.
    Returns a dict keyed by account name with depository + segment lines.
    """
    accounts = {}
    i, n = 0, len(lines)

    while i < n:
        # ----- two-line header -----
        if lines[i].strip() == "ACCOUNT HOLDER":
            depository = (
                "NSDL"
                if any("NSDL Demat Account" in lines[i - j] for j in range(1, 10) if i - j >= 0)
                else "CDSL"
            )
            account_name = lines[i + 1].strip()
            i += 2
        # ----- single-line header (e.g., corporate-bond account) -----
        elif "ACCOUNT HOLDER" in lines[i]:
            account_name = lines[i].split("ACCOUNT HOLDER")[0].strip()
            depository = (
                "NSDL"
                if any("NSDL Demat Account" in lines[i - j] for j in range(1, 10) if i - j >= 0)
                else "CDSL"
            )
            i += 1
        else:
            i += 1
            continue

        seg = []
        while i < n and "ACCOUNT HOLDER" not in lines[i]:
            seg.append(lines[i])
            i += 1

        accounts[account_name] = {"depository": depository, "segment": seg}

    return accounts

def parse_numeric(tokens):
    """Return (nums_f, market_price, value) or (None, None, None)."""
    nums = [t for t in tokens if numeric_re.fullmatch(t.replace(",", ""))]
    nums_f = [float(t.replace(",", "")) for t in nums]
    if len(nums_f) < 2:
        return None, None, None
    return nums_f, nums_f[-2], nums_f[-1]

def build_equity_dfs(accounts):
    """
    For each account, keep only Equities and return per-account DataFrames.
    """
    dfs = {}

    for acc_name, data in accounts.items():
        dep      = data["depository"]
        rows     = []
        curr_ast = None
        prev     = None

        for line in data["segment"]:
            # asset headers
            if "Equities" in line:
                curr_ast = "Equities"
                continue
            if "Corporate Bonds" in line or "Mutual Fund" in line:
                curr_ast = None          # ignore non-equity sections
                continue
            if curr_ast != "Equities":
                prev = line
                continue

            tokens   = line.split()
            isin_idx = next((i for i, t in enumerate(tokens) if isin_re.fullmatch(t)), None)
            if isin_idx is None:        # not an equity detail row
                prev = line
                continue

            nums_f, mp, val = parse_numeric(tokens)
            if nums_f is None:          # no numeric data
                prev = line
                continue

            # Current Balance
            if dep == "CDSL":
                curr_tok = next(
                    (t for t in tokens[isin_idx + 1 :] if numeric_re.fullmatch(t.replace(",", ""))),
