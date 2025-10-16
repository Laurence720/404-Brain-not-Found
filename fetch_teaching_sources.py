import pathlib, requests, textwrap, html, re, time
from bs4 import BeautifulSoup

print(">>> script started")  # Log when the script begins

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "data" / "teaching_corpus"
for p in ["01_investopedia","02_cfa","03_morningstar","04_kaggle"]:
    (OUT / p).mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (educational use)"}

def clean_text(t:str)->str:
    t = html.unescape(t)
    t = re.sub(r"\s+"," ", t).strip()
    return t

def fetch_to_md(name: str, url: str, selector: str|None, title_selector: str|None, outfile: pathlib.Path):
    print(f"[fetch] {name}: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        title = soup.select_one(title_selector).get_text(strip=True) if title_selector and soup.select_one(title_selector) \
                else (soup.title.get_text(strip=True) if soup.title else url)
        node = soup.select_one(selector) if selector else soup.body
        text = clean_text(node.get_text(" ", strip=True)) if node else clean_text(soup.get_text(" ", strip=True))
        md = f"# {title}\n\nSource: {url}\n\n{textwrap.fill(text, width=100)}\n"
        outfile.write_text(md, encoding="utf-8")
        print(f"[ok] wrote -> {outfile}")
    except Exception as e:
        print(f"[error] {name}: {e}")

def main():
    fetch_to_md(
        "Investopedia",
        "https://www.investopedia.com/financial-term-dictionary-4769738",
        selector="[id^='article-body']",
        title_selector="h1",
        outfile=OUT/"01_investopedia"/"financial-term-dictionary.md"
    )
    fetch_to_md(
        "CFA",
        "https://www.cfainstitute.org/en/research/foundation",
        selector="main",
        title_selector="h1",
        outfile=OUT/"02_cfa"/"cfa-foundation-overview.md"
    )
    fetch_to_md(
        "Morningstar",
        "https://www.morningstar.com/lp/investment-glossary",
        selector="main",
        title_selector="h1",
        outfile=OUT/"03_morningstar"/"morningstar-investment-glossary.md"
    )
    print("[info] Kaggle skipped (no CSV).")
    print(">>> all done")

if __name__ == "__main__":
    main()
