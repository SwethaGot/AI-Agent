from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
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
    sources: list[str]
    tools_used: list[str]


# STEP 2: INITIALIZE CLAUDE WITH TOOLS
llm = ChatAnthropic(model="claude-sonnet-4-6")
parser = PydanticOutputParser(pydantic_object=MelbourneDiscoveryResponse)

# Bind tools to the LLM
tools = [event_search_tool, news_search_tool, save_tool, budget_tool]
llm_with_tools = llm.bind_tools(tools)


# STEP 3: CREATE THE PROMPT
system_prompt = """
You are an intelligent Local Event and News Discovery Assistant for Melbourne, Australia 
operating as a ReAct (Reasoning + Acting) agent.

REACT PROCESS:
You will go through multiple iterations of:
1. THOUGHT: Analyze what information you have and what you still need
2. ACTION: Choose which tools to call to get that information
3. OBSERVATION: See the results and decide if you need more information

AVAILABLE TOOLS:
1. search_local_events - Search for events in Melbourne
2. search_melbourne_news - Search for Melbourne news
3. analyze_event_budget - Analyze budget-friendly options
4. save_events - Save results to file

DECISION MAKING:
- If you have enough information to answer the query comprehensively, STOP calling tools
- If you need more specific information, call relevant tools again with refined queries
- You can call the same tool multiple times with different parameters
- You can call tools in any order that makes sense

WHEN TO STOP:
Stop calling tools when you have:
- Found relevant events or news
- Analyzed budget options if requested
- Gathered enough information to provide recommendations
- Have source URLs for the user to verify

Your mission:
1. Search for events and news based on user interests
2. Identify budget-friendly and free options
3. Organize events by date and relevance
4. Provide news highlights relevant to Melbourne
5. Suggest which friend groups to invite based on event type

IMPORTANT FOR SOURCES:
- Extract ALL URLs from tool results
- Include them in the sources field
- Format as: [Title](URL)
- Never make up URLs

Friend group suggestions:
- Tech meetups -> Tech-savvy friends, colleagues
- Concerts -> Music lovers, party friends
- Food festivals -> Foodies, family
- Art exhibitions -> Creative friends
- Sports events -> Sports fans, active friends
- Networking -> Professional contacts
- Comedy shows -> Friends with similar humor
- Markets -> Shopping buddies, family

After you decide you have enough information, I will ask you to provide a final
structured response.

{format_instructions}
"""

# STEP 4: AGENT FUNCTION
def run_agent(query: str, max_iterations: int = 10):
    """
    True ReAct agent with reasoning loop
    """
    print("\n" + "="*70)
    print("REACT AGENT STARTING")
    print("="*70)

    # Initialize conversation history
    messages = [
        SystemMessage(content=system_prompt.format(format_instructions=parser.get_format_instructions())),
        HumanMessage(content=query)
    ]

    iteration = 0

    # REACT LOOP - Agent keeps going until it's done
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- ITERATION {iteration} ---")

        # THOUGHT: Agent decides what to do
        print("Agent is thinking...")
        response = llm_with_tools.invoke(messages)

        # Check if agent is done (no more tool calls)
        if not response.tool_calls:
            print("\nAgent decided it has enough information. Generating final response...")

            final_prompt = f"""
Based on all the information gathered, provide a comprehensive answer about Melbourne events and news.

Original Query: {query}

Conversation History:
{format_conversation(messages)}

IMPORTANT: Extract all URLs from previous tool results and include them in the
sources field formatted as [Title](URL).

Please provide a detailed response in the following JSON format:
{parser.get_format_instructions()}
"""

            final_response = llm.invoke(final_prompt)
            return final_response.content

        # ACTION: Execute the tools agent chose
        print(f"\nAgent chose {len(response.tool_calls)} tool(s) to execute:")

        # Add assistant message with tool calls
        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('args', {})
            tool_call_id = tool_call.get('id', '')

            print(f"\n  Tool: {tool_name}")
            print(f"  Args: {tool_args}")

            # Find and execute the tool
            tool_result = f"Tool {tool_name} not found"
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        tool_result = tool.func(**tool_args)
                        print(f"  Status: Success")
                        print(f"  Result preview: {tool_result[:200]}...")
                    except Exception as e:
                        tool_result = f"Error: {str(e)}"
                        print(f"  Status: Failed - {str(e)}")
                    break

            # OBSERVATION: Add tool result back to conversation
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call_id))

        print(f"\nAgent will now process these results and decide next step...")
    
    # Max iterations reached
    print(f"\n⚠️ Max iterations ({max_iterations}) reached. Generating response with available data...")
    
    final_prompt = f"""
Based on the information gathered so far, provide the best possible answer.

Original Query: {query}

Conversation History:
{format_conversation(messages)}

Please provide a detailed response in the following JSON format:
{parser.get_format_instructions()}
"""
    
    final_response = llm.invoke(final_prompt)
    return final_response.content


def format_conversation(messages):
    """Helper to format conversation history for final prompt"""
    formatted = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        elif isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: Chose tools to execute")
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            formatted.append(f"Tool Result: {content[:500]}...")
    return "\n".join(formatted)


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

            if structured_response.sources:
                print(f"\nSOURCES & LINKS:")
                for i, source in enumerate(structured_response.sources, 1):
                    print(f"  {i}. {source}")

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