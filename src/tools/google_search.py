from googlesearch import search
import config

def google_search(query: str) -> str:
    """Performs a Google search for the given query, retrieves the top search
    result URLs and description from the page.

    Args:
        Input should be a search query.

    Returns:
        URL and description of the search result.
    """
    if config.is_sandbox():
        return "google search results"
    
    for result in search(query, num_results=1, sleep_interval=5, advanced=True):
        return f"url:{result.url},\ndescription:{result.description}"
    # return f"Search results for '{query}' in Google."
    return "No results found."

FUNCTIONS = {
    "google_search": google_search,
}

if __name__ == "__main__":
    print(google_search("Cyber physical systems"))