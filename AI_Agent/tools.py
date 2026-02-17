from langchain_core.tools import tool
from datetime import datetime
import time

try:
    from ddgs import DDGS
except ImportError:
    raise ImportError("Please install ddgs: pip install ddgs")


# TOOL 1: EVENT SEARCH
@tool
def search_local_events(query: str) -> str:
    """
    Search for local events and news in Melbourne using DuckDuckGo.
    Args:
        query: Type of event or news (e.g., 'tech meetups', 'concerts', 'local news')
    Returns:
        String with search results including URLs
    """
    city = "Melbourne"
    
    search_queries = [
        f"{query} events in {city} Australia this week",
        f"{query} in {city} VIC upcoming events",
        f"things to do {city} {query} this weekend"
    ]
    
    results = []
    ddgs = DDGS()
    
    for search_query in search_queries:
        try:
            search_results = ddgs.text(search_query, max_results=5)
            formatted_results = "\n".join([
                f"- Title: {r['title']}\n  URL: {r['href']}\n  Description: {r['body']}"
                for r in search_results
            ])
            results.append(f"Search Query: {search_query}\n\nResults:\n{formatted_results}")
            time.sleep(1)
        except Exception as e:
            results.append(f"Search failed for '{search_query}': {str(e)}")
    
    return "\n\n" + "="*60 + "\n\n".join(results)


# TOOL 2: NEWS SEARCH
@tool
def search_melbourne_news(topic: str) -> str:
    """
    Search for local Melbourne news on a specific topic.
    Args:
        topic: News topic (e.g., 'local news', 'weather', 'traffic', 'politics')
    Returns:
        String with news results including URLs
    """
    news_queries = [
        f"Melbourne Australia {topic} news today",
        f"Melbourne {topic} latest updates"
    ]
    
    results = []
    ddgs = DDGS()
    
    for news_query in news_queries:
        try:
            search_results = ddgs.text(news_query, max_results=5)
            formatted_results = "\n".join([
                f"- Title: {r['title']}\n  URL: {r['href']}\n  Description: {r['body']}"
                for r in search_results
            ])
            results.append(f"News Search: {news_query}\n\nResults:\n{formatted_results}")
            time.sleep(1)
        except Exception as e:
            results.append(f"News search failed for '{news_query}': {str(e)}")
    
    return "\n\n" + "="*60 + "\n\n".join(results)


# TOOL 3: SAVE EVENTS
@tool
def save_events(event_data: str) -> str:
    """
    Save event and news information to a formatted text file.
    Args:
        event_data: The event/news information to save
    Returns:
        Success message with filename
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"melbourne_events_{timestamp}.txt"
    
    formatted_content = f"""
{'='*70}
                MELBOURNE EVENT AND NEWS DISCOVERY RESULTS
{'='*70}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Location: Melbourne, Victoria, Australia

{event_data}

{'='*70}
Note: Please verify event details and news on official sources before relying on this information.

Recommended Melbourne event sources:
- TimeOut Melbourne
- Eventbrite Melbourne
- What's On Melbourne
- The Age (local newspaper)
- Herald Sun (local newspaper)
{'='*70}
    """
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(formatted_content)
        return f"Events and news successfully saved to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"


# TOOL 4: BUDGET FILTER
@tool
def analyze_event_budget(event_info: str, max_budget: str = "100") -> str:
    """
    Analyze events and identify budget-friendly options.
    Args:
        event_info: Event information text
        max_budget: Maximum budget in AUD (default 100)
    Returns:
        Analysis of free and budget-friendly events
    """
    free_keywords = ["free", "no entry fee", "$0", "free entry", "no charge", "complimentary", "free admission"]
    budget_keywords = ["$", "aud", "dollar", "price", "ticket", "cost", "fee"]
    
    lines = event_info.lower().split('\n')
    free_events = []
    budget_events = []
    
    for line in lines:
        if any(keyword in line for keyword in free_keywords):
            free_events.append(line.strip())
        elif any(keyword in line for keyword in budget_keywords):
            budget_events.append(line.strip())
    
    result = f"""
BUDGET ANALYSIS (Maximum Budget: ${max_budget} AUD)

FREE EVENTS FOUND: {len(free_events)}
{chr(10).join(free_events[:5]) if free_events else "No clearly marked free events found in the search results"}

PAID EVENTS FOUND: {len(budget_events)}
{chr(10).join(budget_events[:5]) if budget_events else "No clearly marked paid events found in the search results"}

TIP: Always verify prices on official event pages as pricing may have changed.
    """
    
    return result


# Create tool list for export
event_search_tool = search_local_events
news_search_tool = search_melbourne_news
save_tool = save_events
budget_tool = analyze_event_budget

__all__ = ['event_search_tool', 'news_search_tool', 'save_tool', 'budget_tool']