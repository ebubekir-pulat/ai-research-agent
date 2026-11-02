
from ddgs import DDGS


def web_search(query: str) -> list[dict]:
    """Search the web and return relevant results."""

    results = DDGS().text(
        query,
        max_results=5
    )

    return [
        {
            "title": result.get("title"),
            "url": result.get("href"),
            "snippet": result.get("body")
        }
        for result in results
    ]
