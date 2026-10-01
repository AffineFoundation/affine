"""Public-document fetchers for affine_docqa (generator side).

Every document comes back as {"id", "source", "title", "url", "licence",
"date", "text"} — plain text, HTML stripped. Sources and licences:

  * SEC EDGAR filings (10-K, 10-Q, 20-F, DEF 14A) — public records with no
    copyright asserted; AA-LCR's own "Company Reports" category. The SEC
    asks for a descriptive User-Agent with a contact address and <= 10
    requests/s. The filer list comes from the SEC's own ticker table
    (`company_tickers.json`, market-cap order) so the supply is ~400
    companies, not a hand-typed list; EDGAR_CIKS below is the fallback.
  * GOV.UK publications — Open Government Licence v3 (attribution):
    consultation outcomes, closed consultations, impact assessments,
    independent reports, research, policy papers, corporate reports, plus
    their HTML attachments (the body of most publications lives there).
    AA-LCR's "Government Consultations" category.
  * US Federal Register rules, proposed rules and long notices — US
    government works, public domain (17 U.S.C. § 105); plain text from the
    API's `raw_text_url`. Grouped by docket (proposed rule + final rule)
    or by agency.
  * Europe PMC open-access articles filtered to LICENSE:cc-by — CC-BY.
    AA-LCR's "Academia" category. (arXiv is not used: its Atom API does not
    expose the licence per paper.)

No AA-LCR document is used: titles listed in `exclude_titles.txt` (fill from
AA's AA-LCR page) are skipped case-insensitively.

Caching: JSON/listing responses are cached raw by URL; document bodies are
cached as STRIPPED TEXT (`text/` cache) so a restarted run re-downloads
nothing and a 10-K's 5-15 MB of HTML is kept as ~0.5 MB.
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
# Fetched bodies are cached by URL so a restarted generator does not re-download
# (the 17:54 restart rule: redeploy + restart must not lose the run's inputs).
CACHE_DIR = Path(os.environ.get("AFFINE_DOCQA_CACHE", Path.home() / ".cache" / "affine_docqa" / "http"))
TEXT_CACHE_DIR = CACHE_DIR.parent / "text"
HERE = Path(__file__).resolve().parent
EXCLUDE = {l.strip().lower() for l in (HERE / "exclude_titles.txt").read_text().splitlines()
           if l.strip() and not l.startswith("#")} if (HERE / "exclude_titles.txt").exists() else set()
MIN_DOC_CHARS = 5_000

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


class _Cached:
    """Minimal stand-in for httpx.Response: `.text` and `.json()` from a cached body."""

    def __init__(self, body: bytes) -> None:
        self.content = body
        self.text = body.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.text)


def _key(url: str, params) -> str:
    return hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()


def _fetch(client: httpx.Client, url: str, **kw) -> httpx.Response:
    for attempt in range(4):
        r = client.get(url, **kw)
        if r.status_code in (429, 503) and attempt < 3:
            time.sleep(2.0 * (attempt + 1)); continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"unreachable: {url}")


def _get(client: httpx.Client, url: str, **kw):
    """Raw-body cache (listings, JSON, small pages)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / _key(url, kw.get("params"))
    if cached.exists():
        return _Cached(cached.read_bytes())
    r = _fetch(client, url, **kw)
    tmp = cached.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.rename(cached)
    return r


def _get_text(client: httpx.Client, url: str, strip: bool = True) -> str:
    """Document-body cache: only the stripped text is kept on disk."""
    TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = TEXT_CACHE_DIR / _key(url, None)
    if cached.exists():
        return cached.read_text("utf-8")
    raw = _fetch(client, url).text
    text = html_to_text(raw) if strip else raw.strip()
    tmp = cached.with_suffix(".tmp")
    tmp.write_text(text, "utf-8")
    tmp.rename(cached)
    return text


def _doc(source: str, title: str, url: str, licence: str, date: str, text: str) -> dict | None:
    if title.strip().lower() in EXCLUDE or len(text) < MIN_DOC_CHARS:
        return None
    return {"id": hashlib.sha256(url.encode()).hexdigest()[:12], "source": source, "title": title.strip(),
            "url": url, "licence": licence, "date": date, "text": text}


# -- SEC EDGAR -------------------------------------------------------------------
# Fallback filer set (CIKs are public identifiers) if the SEC ticker table is
# unreachable. The live list is `edgar_companies()` — ~400 filers by market cap.
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
EDGAR_LICENCE = "US SEC public filing"
EDGAR_FORMS = ("10-K", "10-Q", "20-F", "DEF 14A")


def edgar_companies(client: httpx.Client, top: int = 400) -> list[tuple[str, int]]:
    """(name, cik) for the `top` filers of the SEC ticker table (market-cap
    order, one entry per ticker; duplicate CIKs from share classes dropped)."""
    try:
        table = _get(client, "https://www.sec.gov/files/company_tickers.json").json()
    except Exception:  # noqa: BLE001 - fall back to the hand list
        return list(EDGAR_CIKS.items())
    out, seen = [], set()
    for _, row in sorted(table.items(), key=lambda kv: int(kv[0])):
        cik = int(row["cik_str"])
        if cik in seen:
            continue
        seen.add(cik)
        out.append((str(row.get("title") or row.get("ticker")), cik))
        if len(out) >= top:
            break
    return out or list(EDGAR_CIKS.items())


def edgar_submissions(client: httpx.Client, cik: int) -> dict:
    """The filer's submissions JSON: name, sic, sicDescription, filings.recent."""
    return _get(client, f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()


def edgar_filing_index(sub: dict, cik: int, forms=EDGAR_FORMS) -> list[dict]:
    """Every filing of the wanted forms as {form, url, date} (newest first),
    no download yet."""
    rec = sub["filings"]["recent"]
    out = []
    for form, acc, doc, date in zip(rec["form"], rec["accessionNumber"], rec["primaryDocument"], rec["filingDate"]):
        if form not in forms or not doc.lower().endswith((".htm", ".html")):
            continue
        out.append({"form": form, "date": date,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"})
    return out


def edgar_fetch(client: httpx.Client, company: str, filing: dict) -> dict | None:
    text = _get_text(client, filing["url"])
    time.sleep(0.12)   # SEC rate guidance: <= 10 req/s
    d = _doc("edgar", f"{company} {filing['form']} {filing['date'][:4]}", filing["url"], EDGAR_LICENCE,
             filing["date"], text)
    if d:
        d["company"], d["form"] = company, filing["form"]
    return d


def edgar_filings(client: httpx.Client, company: str, cik: int, forms=("10-K",), limit: int = 4) -> list[dict]:
    """Newest `limit` filings of `forms` for one filer (kept for the e1 call shape)."""
    out = []
    for f in edgar_filing_index(edgar_submissions(client, cik), cik, forms):
        d = edgar_fetch(client, company, f)
        if d:
            out.append(d)
        if len(out) >= limit:
            break
    return out


# -- GOV.UK ----------------------------------------------------------------------
GOVUK_LICENCE = "Open Government Licence v3.0"
GOVUK_DOC_TYPES = ("consultation_outcome", "closed_consultation", "impact_assessment", "independent_report",
                   "research", "policy_paper", "corporate_report")


def govuk_search(client: httpx.Client, doc_type: str, count: int = 100, start: int = 0) -> list[dict]:
    res = _get(client, "https://www.gov.uk/api/search.json",
               params={"filter_content_store_document_type": doc_type, "count": count, "start": start,
                       "order": "-public_timestamp", "fields": "title,link,public_timestamp,organisations"}).json()
    return res.get("results", [])


def govuk_document(client: httpx.Client, item: dict, with_attachments: bool = True) -> dict | None:
    """One GOV.UK content item + its HTML attachments as a single document."""
    content = _get(client, f"https://www.gov.uk/api/content{item['link']}").json()
    details = content.get("details", {})
    parts = [details.get("body", "")]
    for key in ("final_outcome_detail", "public_feedback_detail"):
        if details.get(key):
            parts.append(details[key])
    if with_attachments:
        for att in details.get("attachments", []) or []:
            # HTML publications carry attachment_type "html" and no content_type;
            # PDFs / ODT / XLSX carry a content_type and are skipped.
            if att.get("attachment_type") != "html" or att.get("content_type") or not att.get("url"):
                continue
            url = att["url"] if att["url"].startswith("http") else "https://www.gov.uk" + att["url"]
            try:
                sub = _get(client, "https://www.gov.uk/api/content" + httpx.URL(url).path).json()
                body = (sub.get("details") or {}).get("body") or ""
            except Exception:  # noqa: BLE001 - attachment without a content item: strip the page
                body = ""
            parts.append(f"\n\n# {att.get('title', '')}\n\n" + (body if body else _get_text(client, url)))
    text = html_to_text("\n\n".join(p for p in parts if p))
    d = _doc("govuk", item["title"], "https://www.gov.uk" + item["link"], GOVUK_LICENCE,
             (item.get("public_timestamp") or "")[:10], text)
    if d:
        d["organisation"] = ((item.get("organisations") or [{}])[0].get("title") or "")
        d["doc_type"] = content.get("document_type") or ""
    return d


def govuk_consultations(client: httpx.Client, count: int = 60, start: int = 0) -> list[dict]:
    """Consultation outcomes page (kept for the e1 call shape)."""
    out = []
    for item in govuk_search(client, "consultation_outcome", count=count, start=start):
        d = govuk_document(client, item)
        if d:
            out.append(d)
    return out


# -- US Federal Register ---------------------------------------------------------------
FEDREG_LICENCE = "US Federal Register, public domain (17 U.S.C. 105)"
FEDREG_AGENCIES = (
    "environmental-protection-agency", "securities-and-exchange-commission", "food-and-drug-administration",
    "federal-communications-commission", "energy-department", "federal-aviation-administration",
    "centers-for-medicare-medicaid-services", "occupational-safety-and-health-administration",
    "consumer-financial-protection-bureau", "fish-and-wildlife-service",
    "national-highway-traffic-safety-administration", "internal-revenue-service", "federal-reserve-system",
    "education-department", "labor-department", "federal-trade-commission", "nuclear-regulatory-commission",
    "homeland-security-department", "federal-energy-regulatory-commission", "commodity-futures-trading-commission",
    "agriculture-department", "transportation-department", "health-and-human-services-department",
    "housing-and-urban-development-department", "national-oceanic-and-atmospheric-administration",
)
FEDREG_TYPES = ("RULE", "PRORULE")


def fedreg_documents(client: httpx.Client, agency: str, doc_type: str = "RULE", per_page: int = 100,
                     page: int = 1, min_pages: int = 6) -> list[dict]:
    """Listing only (no body): long documents of one agency and type."""
    res = _get(client, "https://www.federalregister.gov/api/v1/documents.json",
               params={"per_page": per_page, "page": page, "order": "newest",
                       "conditions[type][]": doc_type, "conditions[agencies][]": agency,
                       "fields[]": ["title", "raw_text_url", "publication_date", "docket_ids", "document_number",
                                    "html_url", "page_length", "agencies"]}).json()
    out = []
    for r in res.get("results", []):
        if not r.get("raw_text_url") or (r.get("page_length") or 0) < min_pages:
            continue
        out.append({**r, "agency": agency, "type": doc_type})
    return out


def fedreg_fetch(client: httpx.Client, meta: dict) -> dict | None:
    text = _get_text(client, meta["raw_text_url"], strip=False)
    d = _doc("fedreg", meta.get("title") or meta["document_number"], meta.get("html_url") or meta["raw_text_url"],
             FEDREG_LICENCE, meta.get("publication_date") or "", text)
    if d:
        d["agency"], d["docket_ids"], d["doc_type"] = meta["agency"], list(meta.get("docket_ids") or []), meta["type"]
    return d


# -- Europe PMC (CC-BY open access) -------------------------------------------------
PMC_LICENCE = "CC-BY"


def europepmc_articles(client: httpx.Client, query: str, page_size: int = 25, page: int = 1) -> list[dict]:
    q = f'({query}) AND OPEN_ACCESS:y AND LICENSE:"cc by" AND HAS_FT:y AND SRC:MED'
    res = _get(client, "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
               params={"query": q, "format": "json", "pageSize": page_size, "page": page, "resultType": "lite"}).json()
    out = []
    for r in res.get("resultList", {}).get("result", []):
        pmcid = r.get("pmcid")
        if not pmcid:
            continue
        text = _get_text(client, f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
        d = _doc("europepmc", r.get("title", pmcid), f"https://europepmc.org/article/PMC/{pmcid}", PMC_LICENCE,
                 r.get("firstPublicationDate", ""), text)
        if d:
            d["query"] = query
            out.append(d)
    return out


def client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": UA, "Accept-Encoding": "gzip"}, timeout=120, follow_redirects=True)


if __name__ == "__main__":
    with client() as c:
        print(len(edgar_companies(c)), "EDGAR filers")
        docs = edgar_filings(c, "Apple", EDGAR_CIKS["Apple"], limit=1)
        print(json.dumps({k: (v[:200] if isinstance(v, str) else v) for k, v in docs[0].items()}, indent=1) if docs else "none")
