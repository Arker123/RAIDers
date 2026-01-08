import requests
from bs4 import BeautifulSoup
from pathlib import Path
from textwrap import wrap
from openpyxl import Workbook

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
TEMPERATURE = 0.0

CHUNK_SIZE = 1200

MIM_IDS = [
    617892, 614808, 615426, 615515, 616208, 616437, 617921,
    105400, 205250, 300857, 606070, 606640, 619133, 608030,
    608031, 608627, 611895, 612069, 612577, 613435, 613954,
    600795, 617839, 619141
]

COLUMNS = [
    "Gene / Locus",
    "Species",
    "Ancestry / Population",
    "Chromosome",
    "Associated Variant",
    "Functional Mechanism / Biological Effect",
    "Significance / Role",
    "MIM Number"
]

def ollama_generate(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE}
        },
        timeout=300
    )
    r.raise_for_status()
    return r.json()["response"].strip()

def fetch_omim_text(mim_id: int) -> str:
    url = f"https://omim.org/entry/{mim_id}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://omim.org/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    main = soup.find("div", {"id": "content"}) or soup.body
    text = main.get_text(separator="\n")

    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
    return "\n".join(lines)

def chunk_text(text: str):
    return wrap(text, CHUNK_SIZE)

# -----------------------------
# Structured extraction
# -----------------------------
def extract_rows(chunk: str, mim_id: int):
    prompt = f"""
You are an information extraction system for human genetics.
Extract ONLY facts that are EXPLICITLY stated in the text.

Output ONLY valid CSV rows.
Each row must have EXACTLY these 8 columns, in this order:
1. Gene / Locus
2. Species
3. Ancestry / Population
4. Chromosome
5. Associated Variant
6. Functional Mechanism / Biological Effect
7. Significance / Role
8. MIM Number

STRICT RULES (follow exactly):
- DO NOT infer, summarize, or generalize
- DO NOT merge information across different sentences
- DO NOT invent variants, genes, or effects
- Use only exact names appearing in the text
- Associated Variant MUST be one of:
  - rsID if explicitly stated (e.g., rs123456) or
  - HGVS DNA/protein notation (c., g., p.)
  - Otherwise NA if no variant mentioned
- Functional Mechanism / Biological Effect:
  - Use ≤12 words
  - No full sentences
- Significance / Role:
  - Use ≤10 words
  - No full sentences
- Species:
  - Use "Human" if Homo sapiens is implied
  - Or "mice" if mice is implied
  - Otherwise NA
- Ancestry / Population:
  - Only if explicitly stated (e.g., European, Japanese)
  - Otherwise NA
- Chromosome:
  - Use chr1-chr22, chrX, chrY
  - Otherwise NA
- Use "{mim_id}" for MIM Number
- Multiple rows allowed
- NO header
- NO commentary
- NO quotes
- Fields separated by commas only

IMPORTANT:
If a variant is mentioned without a gene, output NA for Gene.
If a gene is mentioned without a variant, DO NOT create a row.


TEXT:
{chunk}
"""
    raw = ollama_generate(prompt)

    rows = []
    for line in raw.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) == 8:
            rows.append(cols)

    return rows

# -----------------------------
# Main pipeline
# -----------------------------
if __name__ == "__main__":
    all_rows = []

    for mim_id in MIM_IDS:
        print(f"Fetching OMIM {mim_id}")
        text = fetch_omim_text(mim_id)

        for i, chunk in enumerate(chunk_text(text)):
            print(f"  Processing chunk {i+1}")
            rows = extract_rows(chunk, mim_id)
            all_rows.extend(rows)

    # -----------------------------
    # Write XLSX
    # -----------------------------
    wb = Workbook()
    ws = wb.active
    ws.title = "OMIM Summary"

    ws.append(COLUMNS)
    for row in all_rows:
        ws.append(row)

    output_path = Path("omim_summary.xlsx")
    wb.save(output_path)

    print(f"Written: {output_path}")
