"""Public-document fetchers for affine_docqa (generator side).

Every document comes back as {"id", "source", "title", "url", "licence",
"date", "text"} — plain text, HTML stripped. Sources and licences:

  * SEC EDGAR 10-K / 10-Q filings — US public domain (17 U.S.C. § 105 does
    not apply to filers, but SEC filings are public records with no
    copyright asserted; AA-LCR's own "Company Reports" category). The SEC
    asks for a descriptive User-Agent with a contact address.
  * GOV.UK consultations and their outcomes — Open Government Licence v3
    (attribution). AA-LCR's "Government Consultations" category.
  * Europe PMC open-access articles filtered to LICENSE:cc-by — CC-BY.
    AA-LCR's "Academia" category. (arXiv is not used: its Atom API does not
    expose the licence per paper.)

No AA-LCR document is used: titles listed in `exclude_titles.txt` (fill from
AA's AA-LCR page) are skipped case-insensitively.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from pathlib import Path

import httpx

UA = os.environ.get("AFFINE_DOCQA_UA", "AffineFoundation research (ops@affine.io)")
HERE = Path(__file__).resolve().parent
EXCLUDE = {l.strip().lower() for l in (HERE / "exclude_titles.txt").read_text().splitlines()
           if l.strip() and not l.startswith("#")} if (HERE / "exclude_titles.txt").exists() else set()

TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t\xa0]+")
NL_RE = re.compile(r"\n{3,}")


def html_to_text(s: str) -> str:
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"<(br|/p|/div|/tr|/li|/h\d)[^>]*>", "\n", s, flags=re.I)
    s = HTML_RE.sub(" ", s)
    s = html.unescape(s)
    s = WS_RE.sub(" ", s)
    return NL_RE.sub("\n\n", "\n".join(line.strip() for line in s.splitlines())).strip()


def _get(client: httpx.Client, url: str, **kw) -> httpx.Response:
    for attempt in range(4):
        r = client.get(url, **kw)
        if r.status_code in (429, 503) and attempt < 3:
            time.sleep(2.0 * (attempt + 1)); continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"unreachable: {url}")


def _doc(source: str, title: str, url: str, licence: str, date: str, text: str) -> dict | None:
    if title.strip().lower() in EXCLUDE or len(text) < 5_000:
        return None
    return {"id": hashlib.sha256(url.encode()).hexdigest()[:12], "source": source, "title": title.strip(),
            "url": url, "licence": licence, "date": date, "text": text}


# -- SEC EDGAR -------------------------------------------------------------------
# A fixed set of large filers (CIKs are public identifiers); 10-K = annual report.
EDGAR_CIKS = {
    "Apple": 320193, "Microsoft": 789019, "Alphabet": 1652044, "Amazon": 1018724, "NVIDIA": 1045810, "Meta": 1326801,
    "Tesla": 1318605, "Berkshire Hathaway": 1067983, "JPMorgan": 19617, "Johnson & Johnson": 200406,
    "Exxon Mobil": 34088, "Procter & Gamble": 80424, "Walmart": 104169, "Home Depot": 354950, "Pfizer": 78003,
    "Coca-Cola": 21344, "PepsiCo": 77476, "Intel": 50863, "Cisco": 858877, "Netflix": 1065280, "Adobe": 796343,
    "Salesforce": 1108524, "Nike": 320187, "McDonald's": 63908, "Boeing": 12927, "Caterpillar": 18230,
    "3M": 66740, "Ford": 37996, "General Motors": 1467858, "Starbucks": 829224, "Costco": 909832, "Oracle": 1341439,
    "AMD": 2488, "Qualcomm": 804328, "Micron": 723125, "Chevron": 93410, "UnitedHealth": 731766, "Merck": 310158,
    "AbbVie": 1551152, "Eli Lilly": 59478,
}


def edgar_filings(client: httpx.Client, company: str, cik: int, forms=("10-K",), limit: int = 4) -> list[dict]:
    sub = _get(client, f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    rec = sub["filings"]["recent"]
    out = []
    for form, acc, doc, date in zip(rec["form"], rec["accessionNumber"], rec["primaryDocument"], rec["filingDate"]):
        if form not in forms or not doc.lower().endswith((".htm", ".html")):
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
        text = html_to_text(_get(client, url).text)
        d = _doc("edgar", f"{company} {form} {date[:4]}", url, "US SEC public filing", date, text)
        if d:
            d["company"] = company
            out.append(d)
        if len(out) >= limit:
            break
        time.sleep(0.2)   # SEC rate guidance: <= 10 req/s
    return out


# -- GOV.UK ----------------------------------------------------------------------
def govuk_consultations(client: httpx.Client, count: int = 60, start: int = 0) -> list[dict]:
    search = _get(client, "https://www.gov.uk/api/search.json",
                  params={"filter_content_store_document_type": "consultation_outcome", "count": count, "start": start,
                          "fields": "title,link,public_timestamp,organisations"}).json()
    out = []
    for item in search.get("results", []):
        content = _get(client, f"https://www.gov.uk/api/content{item['link']}").json()
        details = content.get("details", {})
        parts = [details.get("body", "")]
        for key in ("final_outcome_detail", "public_feedback_detail"):
            if details.get(key):
                parts.append(details[key])
        text = html_to_text("\n\n".join(p for p in parts if p))
        d = _doc("govuk", item["title"], "https://www.gov.uk" + item["link"], "Open Government Licence v3.0",
                 (item.get("public_timestamp") or "")[:10], text)
        if d:
            d["organisation"] = ((item.get("organisations") or [{}])[0].get("title") or "")
            out.append(d)
    return out


# -- Europe PMC (CC-BY open access) -------------------------------------------------
def europepmc_articles(client: httpx.Client, query: str, page_size: int = 25) -> list[dict]:
    q = f'({query}) AND OPEN_ACCESS:y AND LICENSE:"cc by" AND HAS_FT:y AND SRC:MED'
    res = _get(client, "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
               params={"query": q, "format": "json", "pageSize": page_size, "resultType": "lite"}).json()
    out = []
    for r in res.get("resultList", {}).get("result", []):
        pmcid = r.get("pmcid")
        if not pmcid:
            continue
        xml = _get(client, f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML").text
        text = html_to_text(xml)
        d = _doc("europepmc", r.get("title", pmcid), f"https://europepmc.org/article/PMC/{pmcid}", "CC-BY",
                 r.get("firstPublicationDate", ""), text)
        if d:
            d["query"] = query
            out.append(d)
    return out


def client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": UA, "Accept-Encoding": "gzip"}, timeout=120, follow_redirects=True)


if __name__ == "__main__":
    with client() as c:
        docs = edgar_filings(c, "Apple", EDGAR_CIKS["Apple"], limit=1)
        print(json.dumps({k: (v[:200] if isinstance(v, str) else v) for k, v in docs[0].items()}, indent=1) if docs else "none")
