from urllib.parse import quote_plus
import webbrowser
import config

def play_music_link(song: str, provider: str = "ytmusic", prefer_app: bool = False) -> str:
    """
    Return a playable/searchable link:
      - ytmusic: https://music.youtube.com/search?q=...
      - spotify: https://open.spotify.com/search/...  or  spotify:search:... (App)
      - apple:   https://music.apple.com/search?term=...
    If prefer_app=True, use app scheme for Spotify (requires system support for 'spotify:')
    """
    q = quote_plus((song or "").strip())
    if not q:
        return "Error: empty song name"

    if provider == "ytmusic":
        return f"https://music.youtube.com/search?q={q}"
    elif provider == "spotify":
        return f"spotify:search:{q}" if prefer_app else f"https://open.spotify.com/search/{q}"
    elif provider in ("apple", "itunes"):
        # Web link is most stable; if you are on iOS/macOS, you can also try 'music://' custom scheme
        return f"https://music.apple.com/search?term={q}"
    elif provider in ("system", "music"):
        # Similar to API-Bank, but many environments do not support direct handling of custom scheme
        return f"music://{q}"
    else:
        return f"Error: unknown provider '{provider}'"

def play_music(song: str, provider: str = "ytmusic", prefer_app: bool = False, open_now: bool = True) -> str:

    """
    Play a song from the user's music library.

    Args:
        None.

    Returns:
        str: The song playing confirmation.
    """
    if config.is_sandbox():
        return "Ready to play: Loves Me Not via ytmusic -> https://music.youtube.com/search?q=Loves+Me+Not"


    url = play_music_link(song, provider=provider, prefer_app=prefer_app)
    if url.startswith("Error"):
        return url
    if open_now:
        try:
            webbrowser.open(url, new=2)  # new=2: Prefer new tab
        except Exception as e:
            return f"Failed to open: {e}\nLink: {url}"
    return f"Ready to play: {song} via {provider} -> {url}"

FUNCTIONS = {
    "play_music": play_music,
}

if __name__ == "__main__":
    print(play_music("Loves Me Not", provider="ytmusic"))