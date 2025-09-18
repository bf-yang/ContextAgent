import requests, re, html
import xml.etree.ElementTree as ET
import config

UA = "med-lookup/0.1 (contextagent@gmail.com)"  

def _snippet(text: str, max_sent=2, max_chars=400) -> str:
    """Ultra-light summary: clean and take the first few sentences"""
    if not text:
        return ""
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    out = " ".join(p.strip() for p in parts if p.strip()[:max_chars])
    return out[:max_chars]

def get_medical_knowledge(query: str, limit: int = 3) -> str:
    """
    Get medical expert knowledge from the up-to-date medical knowledge database.

    Args:
        query (str): The query string containing the medical topic or symptoms.

    Returns:
        str: The medical expert knowledge as a string.
    """
    if config.is_sandbox():
        return (
            "Top PubMed evidence (information only, not medical advice):\n"
            "- Diabetes mellitus is a chronic condition characterized by high blood sugar levels due to insulin resistance or deficiency. Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision. (Source: PubMed)\n"
            "- Type 2 diabetes is the most common form of diabetes, often associated with obesity and lifestyle factors. Management includes lifestyle changes, oral medications, and sometimes insulin therapy. (Source: PubMed)\n"
            "- Complications of diabetes can include cardiovascular disease, neuropathy, nephropathy, and retinopathy. Regular monitoring and management are crucial to prevent these complications. (Source: PubMed)"
        )


    q = (query or "").strip()
    if not q:
        return "Error: query cannot be empty."

    try:
        # 1) Search for IDs
        es = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":q,"retmax":str(limit),"retmode":"json","sort":"relevance"},
            headers={"User-Agent": UA}, timeout=15
        ).json()
        ids = es.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No PubMed results for: {q}"

        # 2) Fetch details (including abstract)
        ef = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db":"pubmed","id":",".join(ids),"retmode":"xml"},
            headers={"User-Agent": UA}, timeout=20
        )
        root = ET.fromstring(ef.text)

        lines = ["Top PubMed evidence (information only, not medical advice):"]
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            a = art.find(".//Article")
            title = (a.findtext("ArticleTitle") if a is not None else "") or "(no title)"
            abs_text = " ".join([(t.text or "") for t in a.findall(".//AbstractText")]) if a is not None else ""
            journal = art.findtext(".//Journal/Title") or ""
            year = (art.findtext(".//JournalIssue/PubDate/Year")
                    or art.findtext(".//ArticleDate/Year") or "")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            snippet = _snippet(abs_text) or "(no abstract)"
            lines.append(f"- {title} — {journal} ({year})\n  {url}\n  Summary: {snippet}")
        return "\n".join(lines)

    except Exception as e:
        return f"Search failed: {e}"

FUNCTIONS = {
    "get_medical_knowledge": get_medical_knowledge,
}


if __name__ == "__main__":
    print(get_medical_knowledge("diabetes symptoms"))