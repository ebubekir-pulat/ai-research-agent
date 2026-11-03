from ddgs import DDGS


def web_search(query: str) -> list[dict]:
    """Search the web and return relevant results."""

    try:    
        results = DDGS().text(
            query,
            max_results=5
        )

        if not results:
            return []

        return [
            {
                "title": result.get("title"),
                "url": result.get("href"),
                "snippet": result.get("body")
            }
            for result in results
        ]

    except Exception as e:
        print(f"Search failed for '{query}': {e}")
        return []