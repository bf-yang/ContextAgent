from langchain_community.utilities import WikipediaAPIWrapper
import config
wikipedia_api_wrapper = None

def wikipedia_search(query: str) -> str:
    """Tool that searches the Wikipedia API. Useful for when you need to answer
    general questions about people, places, companies, facts, historical
    events, or other subjects.

    Args:
        Input should be a search query.

    Returns:
        Wikipedia search result.
    """
    if config.is_sandbox():
        return "Cyber physical systems (CPS) are systems that integrate computation, networking, and physical processes. In CPS, embedded computers and networks monitor and control the physical processes, usually with feedback loops where physical processes affect computations and vice versa. Examples of CPS include smart grids, autonomous automobile systems, medical monitoring, industrial control systems, robotics systems, and automatic pilot avionics. CPS is a key technology in the development of the Internet of Things (IoT) and Industry 4.0."
    
    global wikipedia_api_wrapper
    if wikipedia_api_wrapper is None:
        wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1)

    return wikipedia_api_wrapper.run(query)

FUNCTIONS = {
    "wikipedia_search": wikipedia_search,
}

if __name__ == "__main__":
    print(wikipedia_search("Cyber Physical Systems"))

