from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from tools import event_search_tool, news_search_tool, save_tool, budget_tool

load_dotenv()


# STEP 1: DEFINE OUTPUT STRUCTURE
class MelbourneDiscoveryResponse(BaseModel):
    """Structured response for Melbourne event and news discovery"""
    query: str
    city: str
    events_found: list[str]
    news_highlights: list[str]
    recommendations: list[str]
    budget_friendly_options: list[str]
    friend_group_suggestions: list[str]
    tools_used: list[str]


# STEP 2: INITIALIZE CLAUDE WITH TOOLS
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
parser = PydanticOutputParser(pydantic_object=MelbourneDiscoveryResponse)

# Bind tools to the LLM
tools = [event_search_tool, news_search_tool, save_tool, budget_tool]
llm_with_tools = llm.bind_tools(tools)


# STEP 3: CREATE THE PROMPT
system_prompt = """
You are an intelligent Local Event and News Discovery Assistant for Melbourne, Australia.

Your mission:
1. Search for events and news based on user interests (concerts, meetups, festivals, local news, etc.)
2. Identify budget-friendly and free options
3. Organize events by date and relevance
4. Provide news highlights relevant to Melbourne
5. Suggest which friend groups to invite based on event type
6. Save results in an easy-to-read format

When suggesting friend groups, consider:
- Tech meetups -> Tech-savvy friends, colleagues, aspiring developers
- Concerts/Music -> Music lovers, party friends, concert buddies
- Food festivals -> Foodies, family, casual friend groups
- Art exhibitions -> Creative friends, art enthusiasts
- Sports events (AFL, cricket, tennis) -> Sports fans, active friends
- Networking events -> Professional contacts, entrepreneurs
- Comedy shows -> Friends with similar humor, casual groups
- Markets -> Shopping buddies, family, friends who love local goods

For news queries:
- Search for relevant Melbourne and Victoria news
- Summarize key points
- Include dates and sources when available

Use the available tools to:
1. Search for events using search_local_events
2. Search for news using search_melbourne_news
3. Analyze budget with analyze_event_budget
4. Save results using save_events

After gathering all information, provide a comprehensive response with event details, news highlights, recommendations, and friend group suggestions.

{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{query}"),
])

prompt = prompt.partial(format_instructions=parser.get_format_instructions())


# STEP 4: SIMPLE AGENT FUNCTION
def run_agent(query: str):
    """
    Simple agent that calls tools and processes results
    """
    print("\nStep 1: Analyzing query and selecting tools...")
    
    # First call to decide which tools to use
    response = llm_with_tools.invoke([
        ("system", system_prompt.format(format_instructions=parser.get_format_instructions())),
        ("human", query)
    ])
    
    results = []
    
    # Check if tools were called
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\nStep 2: Executing {len(response.tool_calls)} tool(s)...")
        
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('args', {})
            
            print(f"\n  - Calling tool: {tool_name}")
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = tool.func(**tool_args)
                        results.append(f"Tool: {tool_name}\nResult: {result}")
                        print(f"    Status: Success")
                    except Exception as e:
                        error_msg = f"Error with {tool_name}: {str(e)}"
                        results.append(error_msg)
                        print(f"    Status: Failed - {str(e)}")
                    break
    
    # Combine all results
    combined_results = "\n\n".join(results) if results else "No tools were executed."
    
    print("\nStep 3: Generating final response...")
    
    # Second call to generate final structured response
    final_prompt = f"""
Based on the following tool results, provide a comprehensive answer about Melbourne events and news.

Tool Results:
{combined_results}

Original Query: {query}

Please provide a detailed response in the following JSON format:
{parser.get_format_instructions()}
"""
    
    final_response = llm.invoke(final_prompt)
    
    return final_response.content


# STEP 5: RUN THE AGENT
def main():
    print("="*70)
    print("        MELBOURNE LOCAL EVENT AND NEWS DISCOVERY AGENT")
    print("="*70)
    
    # Get user input
    search_type = input("\nWhat would you like to search for?\n1. Events\n2. News\n3. Both\nEnter choice (1/2/3): ")
    
    if search_type == "1":
        event_type = input("\nWhat type of events are you interested in?\n(e.g., tech meetups, concerts, food festivals, AFL games, art shows, comedy shows)\n> ")
        budget_input = input("\nWhat is your budget in AUD? (press Enter for 'any')\n> ")
        
        if budget_input:
            query = f"Find {event_type} events in Melbourne Australia with budget under ${budget_input} AUD"
        else:
            query = f"Find {event_type} events in Melbourne Australia"
    
    elif search_type == "2":
        news_topic = input("\nWhat news topic are you interested in?\n(e.g., local news, weather, traffic, politics, community events)\n> ")
        query = f"Find latest news about {news_topic} in Melbourne Australia"
    
    else:
        event_type = input("\nWhat type of events are you interested in?\n(e.g., tech meetups, concerts, food festivals)\n> ")
        news_topic = input("\nWhat news topic are you interested in?\n(e.g., local news, weather, traffic)\n> ")
        budget_input = input("\nWhat is your budget in AUD? (press Enter for 'any')\n> ")
        
        if budget_input:
            query = f"Find {event_type} events in Melbourne Australia with budget under ${budget_input} AUD and also provide latest {news_topic} news in Melbourne"
        else:
            query = f"Find {event_type} events and latest {news_topic} news in Melbourne Australia"
    
    print(f"\nSearching for: {query}")
    print("="*70)
    
    # Execute the agent
    try:
        output = run_agent(query)
        
        # Try to parse structured response
        try:
            structured_response = parser.parse(output)
            
            print("\n" + "="*70)
            print("                    DISCOVERY COMPLETE")
            print("="*70)
            print(f"\nCity: {structured_response.city}")
            print(f"\nQuery: {structured_response.query}")
            
            print(f"\nEVENTS FOUND ({len(structured_response.events_found)}):")
            for i, event in enumerate(structured_response.events_found, 1):
                print(f"  {i}. {event}")
            
            if structured_response.news_highlights:
                print(f"\nNEWS HIGHLIGHTS ({len(structured_response.news_highlights)}):")
                for i, news in enumerate(structured_response.news_highlights, 1):
                    print(f"  {i}. {news}")
            
            print(f"\nTOP RECOMMENDATIONS:")
            for i, rec in enumerate(structured_response.recommendations, 1):
                print(f"  {i}. {rec}")
            
            if structured_response.budget_friendly_options:
                print(f"\nBUDGET-FRIENDLY OPTIONS:")
                for i, option in enumerate(structured_response.budget_friendly_options, 1):
                    print(f"  {i}. {option}")
            
            print(f"\nFRIEND GROUP SUGGESTIONS:")
            for i, suggestion in enumerate(structured_response.friend_group_suggestions, 1):
                print(f"  {i}. {suggestion}")
            
            print(f"\nTools Used: {', '.join(structured_response.tools_used)}")
            print("="*70)
            
        except Exception as parse_error:
            print("\nCould not parse structured response. Showing raw output:")
            print("="*70)
            print(output)
            print("="*70)
            print(f"\nParse Error Details: {parse_error}")
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify your ANTHROPIC_API_KEY in .env file")
        print("3. Ensure all packages are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()