from urllib.parse import urlsplit, urlunsplit
import requests, re, html
from ddgs import DDGS
import config

UA = "reddit-mini/0.1 (contextagent@gmail.com)"  # Custom UA is more stable

def _json_url(url: str):
    if "/comments/" not in url:
        return None
    p = urlsplit(url.replace("old.reddit.com", "www.reddit.com"))
    return urlunsplit((p.scheme, p.netloc, p.path.rstrip("/"), "", "")) + ".json"

def _snippet(json_url: str) -> str:
    r = requests.get(json_url, headers={"User-Agent": UA}, timeout=10)
    r.raise_for_status()
    data = r.json()
    post = data[0]["data"]["children"][0]["data"]
    txt = post.get("selftext", "")
    if not txt and len(data) > 1:  # If no main text, get the first top-level comment
        for c in data[1]["data"].get("children", []):
            if c.get("kind") == "t1":
                txt = c["data"].get("body", ""); break
    txt = html.unescape(re.sub(r"\s+", " ", txt)).strip()
    if not txt: return "(No preview text)"
    parts = re.split(r"(?<=[。！？.!?])\s+", txt)  # Take the first 2 sentences
    return " ".join(parts[:2])[:300]

def search_reddit(query: str, limit: int = 5, subreddit: str | None = None) -> str:
    if config.is_sandbox():
        return "This is going to be a SUBJECTIVE noise cancelling comparison between all the headphones I have tried, with subjective I mean Im going to tell you my experience and comparisons between those headphones, Im not going to take in count objective measurments, technology, mics quantity etc or db cancelled"
    
    q = f"site:reddit.com/r/{subreddit} {query}" if subreddit else f"site:reddit.com {query}"
    items = []
    with DDGS() as d:
        for r in d.text(q, max_results=limit * 3):  # Get more results, filter to ensure enough
            url = (r.get("href") or r.get("link") or "")
            if not url: continue
            jurl = _json_url(url)
            if not jurl: continue
            title = (r.get("title") or "(no title)").strip()
            try:
                snip = _snippet(jurl)
                url = url.replace("www.reddit.com", "old.reddit.com")
                items.append(f"- {title}\n  {url}\n  Snippet: {snip}")
            except Exception:
                continue
            if len(items) >= limit: break
    return "Top Reddit results:\n" + "\n".join(items) if items else f"No Reddit results for: {query}"


if __name__ == "__main__":
    query = "best noise cancelling headphones"
    result = search_reddit(query)
    print(result)