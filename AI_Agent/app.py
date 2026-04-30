import re
import json
import time
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from tools import event_search_tool, news_search_tool, save_tool, budget_tool
from datetime import datetime

try:
    from rag import RAGManager
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False
from anthropic._exceptions import OverloadedError

load_dotenv()


# Define output structure
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


# Initialize session state for conversation
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'user_context' not in st.session_state:
    st.session_state.user_context = {
        'preferences': {},
        'search_history': [],
        'current_results': None
    }

if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = []


def invoke_with_retry(llm_callable, messages, max_retries=3):
    """Retry an LLM call on OverloadedError with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return llm_callable(messages)
        except OverloadedError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            st.toast(f"API busy, retrying in {wait}s...", icon="⏳")
            time.sleep(wait)


# Initialize Claude
@st.cache_resource
def init_llm():
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    parser = PydanticOutputParser(pydantic_object=MelbourneDiscoveryResponse)

    system_prompt = """
    You are a conversational Local Event Intelligence Assistant for Melbourne, Australia.
    
    You are having a MULTI-TURN conversation with the user. This means:
    - You can ask clarifying questions
    - You remember what was said previously
    - You can refine searches based on user feedback
    - You help users iteratively find what they want
    
    CONVERSATION FLOW:
    1. If user query is vague, ask clarifying questions
    2. If you search and user wants refinement, search again with better parameters
    3. Build understanding of user preferences over time
    4. Suggest related events based on what they liked before
    
    AVAILABLE TOOLS:
    1. search_local_events - Search for events in Melbourne
    2. search_melbourne_news - Search for Melbourne news
    3. analyze_event_budget - Analyze budget-friendly options
    4. save_events - Save results to file
    
    WHEN TO ASK QUESTIONS vs SEARCH:
    - Vague query like "find events" → Ask what type of events
    - "tech meetups" → Search directly
    - After showing results, ask "Would you like to refine this?"
    - If user says "too expensive" → Search with lower budget
    - If user says "not interested" → Ask what they prefer instead
    
    IMPORTANT:
    - Extract URLs from search results
    - Format sources as [Title](URL)
    - Remember user preferences for future suggestions
    
    {format_instructions}
    """

    tools = [event_search_tool, news_search_tool, save_tool, budget_tool]
    llm_with_tools = llm.bind_tools(tools)

    return llm, llm_with_tools, parser, system_prompt, tools


@st.cache_resource
def init_rag():
    if not RAG_AVAILABLE:
        return None
    try:
        return RAGManager()
    except Exception:
        return None


llm, llm_with_tools, parser, system_prompt, tools = init_llm()
rag = init_rag()


def run_conversational_agent(user_input: str):
    """
    Run agent with full conversation context
    """
    # Build conversation context
    conversation_context = build_conversation_context()
    
    # Add user input to memory
    st.session_state.agent_memory.append({
        "role": "user",
        "content": user_input
    })
    
    # Agent decides: clarify, search, or respond from context
    agent_decision = make_agent_decision(user_input, conversation_context)
    
    if agent_decision['action'] == 'clarify':
        # Agent asks a question
        response = agent_decision['question']
        st.session_state.agent_memory.append({
            "role": "assistant",
            "content": response
        })
        return {'type': 'text', 'content': response}
    
    elif agent_decision['action'] == 'search':
        search_params = agent_decision['params']
        event_type = search_params.get('event_type', 'events')

        # 1. Check RAG cache first (only if RAG is available)
        cached = rag.retrieve_cached_search(user_input, search_params) if rag else None
        if cached:
            results = _parse_cached_result(cached, user_input)
            source_label = "memory"
        else:
            # 2. Live search
            results = execute_search_with_context(search_params)
            source_label = "live"
            # 3. Cache the result for next time
            if rag and results and results.events_found:
                rag.store_search_result(user_input, format_search_results(results), search_params)

        # 4. Always update user preferences
        if rag:
            rag.store_user_preference(event_type)

        st.session_state.user_context['current_results'] = results
        st.session_state.user_context['search_history'].append({
            'query': user_input,
            'params': search_params,
            'timestamp': datetime.now().isoformat(),
            'source': source_label
        })

        response_text = format_search_results(results)
        if source_label == "memory":
            response_text = "*(From recent search cache)*\n\n" + response_text

        st.session_state.agent_memory.append({
            "role": "assistant",
            "content": response_text
        })

        return {'type': 'results', 'content': results, 'text': response_text}
    
    elif agent_decision['action'] == 'refine':
        # Agent refines previous search
        refined_params = refine_search_params(agent_decision['refinement'])
        results = execute_search_with_context(refined_params)
        
        st.session_state.user_context['current_results'] = results
        response_text = format_search_results(results)
        
        st.session_state.agent_memory.append({
            "role": "assistant",
            "content": response_text
        })
        
        return {'type': 'results', 'content': results, 'text': response_text}
    
    else:
        # Agent responds from context
        response = agent_decision['response']
        st.session_state.agent_memory.append({
            "role": "assistant",
            "content": response
        })
        return {'type': 'text', 'content': response}


def truncate_at_line(text: str, limit: int) -> str:
    """Truncate text at a line boundary to avoid cutting URLs mid-string."""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_newline = truncated.rfind('\n')
    return truncated[:last_newline] if last_newline != -1 else truncated


def build_conversation_context():
    """Build context from conversation history"""
    context = {
        'conversation_history': st.session_state.agent_memory[-10:],  # Last 10 exchanges
        'user_preferences': st.session_state.user_context['preferences'],
        'previous_searches': st.session_state.user_context['search_history'][-3:],  # Last 3 searches
        'current_results': st.session_state.user_context['current_results']
    }
    return context


def make_agent_decision(user_input: str, context: dict):
    """
    Agent decides what to do based on input and context
    """
    decision_prompt = f"""
You are deciding what action to take based on user input.

User input: "{user_input}"

Previous conversation: {len(context['conversation_history'])} messages
Has current results: {context['current_results'] is not None}

RULES:
1. If user mentions specific event type (tech, food, music, sports, comedy, etc.) → SEARCH
2. If user mentions timeframe (this week, weekend, today, tomorrow) → SEARCH  
3. If user mentions budget (free, cheap, under $X) → SEARCH
4. If user says very vague things like just "events" or "find me something" → CLARIFY
5. If user is refining previous results (cheaper, different, not those) → REFINE
6. If user asks about current results → RESPOND

Based on the input "{user_input}", what should I do?

Respond with ONLY ONE of these exact formats:

For CLARIFY:
CLARIFY: What type of events are you interested in? (tech, food, sports, music, art, etc.)

For SEARCH:
SEARCH: {{"event_type": "tech", "budget": 100, "timeframe": "this week"}}

For REFINE:
REFINE: {{"aspect": "budget", "value": "50"}}

For RESPOND:
RESPOND: [your response here]

Your response:"""
    
    response = invoke_with_retry(llm.invoke, decision_prompt)
    content = response.content.strip()
    
    # Parse the response
    if content.startswith("CLARIFY:"):
        question = content.replace("CLARIFY:", "").strip()
        return {
            'action': 'clarify',
            'question': question
        }
    
    elif content.startswith("SEARCH:"):
        try:
            params_str = content.replace("SEARCH:", "").strip()
            params = json.loads(params_str)
            return {
                'action': 'search',
                'params': params
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback: extract keywords
            return {
                'action': 'search',
                'params': extract_search_params_from_text(user_input)
            }

    elif content.startswith("REFINE:"):
        try:
            refinement_str = content.replace("REFINE:", "").strip()
            refinement = json.loads(refinement_str)
            return {
                'action': 'refine',
                'refinement': refinement
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            return {
                'action': 'clarify',
                'question': "Could you clarify what you'd like to change about the search?"
            }
    
    elif content.startswith("RESPOND:"):
        response_text = content.replace("RESPOND:", "").strip()
        return {
            'action': 'respond',
            'response': response_text
        }
    
    else:
        # Default fallback
        return {
            'action': 'search',
            'params': extract_search_params_from_text(user_input)
        }


def extract_search_params_from_text(text: str):
    """
    Simple keyword extraction for search parameters
    """
    text_lower = text.lower()
    
    params = {
        'event_type': 'events',
        'budget': 100,
        'timeframe': 'this week'
    }
    
    # Extract event type
    event_keywords = {
        'tech': ['tech', 'technology', 'coding', 'programming', 'developer', 'startup'],
        'food': ['food', 'restaurant', 'dining', 'culinary', 'cooking'],
        'music': ['music', 'concert', 'band', 'DJ', 'festival'],
        'sports': ['sport', 'fitness', 'AFL', 'cricket', 'running', 'gym'],
        'art': ['art', 'gallery', 'exhibition', 'museum', 'culture'],
        'comedy': ['comedy', 'standup', 'funny', 'comedian'],
        'networking': ['networking', 'professional', 'business', 'meetup']
    }
    
    for event_type, keywords in event_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            params['event_type'] = event_type
            break
    
    # Extract budget
    if 'free' in text_lower:
        params['budget'] = 0
    elif 'cheap' in text_lower or 'budget' in text_lower:
        params['budget'] = 30
    elif '$' in text:
        match = re.search(r'\$(\d+)', text)
        if match:
            params['budget'] = int(match.group(1))
    
    # Extract timeframe
    if 'today' in text_lower:
        params['timeframe'] = 'today'
    elif 'tomorrow' in text_lower:
        params['timeframe'] = 'tomorrow'
    elif 'weekend' in text_lower or 'saturday' in text_lower or 'sunday' in text_lower:
        params['timeframe'] = 'this weekend'
    elif 'week' in text_lower:
        params['timeframe'] = 'this week'
    elif 'month' in text_lower:
        params['timeframe'] = 'this month'
    
    return params


def execute_search_with_context(params: dict):
    """
    Execute search with ReAct loop
    """
    # Build search query from params
    query_parts = []
    if params.get('event_type'):
        query_parts.append(params['event_type'])
    if params.get('timeframe'):
        query_parts.append(params['timeframe'])
    
    search_query = f"Find {' '.join(query_parts)} events in Melbourne"
    if params.get('budget') is not None:
        search_query += f" with budget under ${params['budget']}"
    
    # Build messages for ReAct agent
    messages = [
        SystemMessage(content=system_prompt.format(format_instructions=parser.get_format_instructions())),
        HumanMessage(content=search_query)
    ]

    # ReAct loop
    iteration = 0
    max_iterations = 5
    all_tool_results = []

    while iteration < max_iterations:
        iteration += 1

        response = invoke_with_retry(llm_with_tools.invoke, messages)

        # Check if done
        if not response.tool_calls:
            break

        # Add assistant message with tool calls
        messages.append(response)

        # Execute tools
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('args', {})
            tool_call_id = tool_call.get('id', '')

            tool_result = f"Tool {tool_name} not found"
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        tool_result = tool.func(**tool_args)
                        all_tool_results.append(tool_result)
                    except Exception as e:
                        tool_result = f"Error: {str(e)}"
                    break

            messages.append(ToolMessage(content=truncate_at_line(tool_result, 2000), tool_call_id=tool_call_id))
    
    # Generate final structured response
    if all_tool_results:
        final_prompt = f"""
Based on these search results, provide a structured response.

Search query: {search_query}
Tool results summary: {len(all_tool_results)} tools executed

Results:
{truncate_at_line(all_tool_results[0], 3000) if all_tool_results else "No results"}

Provide response in this JSON format:
{parser.get_format_instructions()}
"""
        
        try:
            final_response = invoke_with_retry(llm.invoke, final_prompt)
            parsed = parser.parse(final_response.content)
            return parsed
        except Exception as e:
            st.error(f"Error parsing response: {str(e)}")
            # Return minimal structure
            return MelbourneDiscoveryResponse(
                query=search_query,
                city="Melbourne",
                events_found=["Search completed but parsing failed. Raw results available in logs."],
                news_highlights=[],
                recommendations=["Try refining your search"],
                budget_friendly_options=[],
                friend_group_suggestions=[],
                sources=[],
                tools_used=["search_local_events"]
            )
    
    # No results
    return MelbourneDiscoveryResponse(
        query=search_query,
        city="Melbourne",
        events_found=["No events found matching your criteria"],
        news_highlights=[],
        recommendations=["Try broadening your search criteria", "Try different event types", "Check different timeframes"],
        budget_friendly_options=[],
        friend_group_suggestions=[],
        sources=[],
        tools_used=[]
    )

def refine_search_params(refinement: dict):
    """
    Refine previous search based on user feedback
    """
    # Get last search
    if st.session_state.user_context['search_history']:
        last_search = st.session_state.user_context['search_history'][-1]
        params = last_search['params'].copy()
        
        # Apply refinement
        if refinement['aspect'] == 'budget':
            params['budget'] = int(refinement['value'])
        elif refinement['aspect'] == 'time':
            params['timeframe'] = refinement['value']
        elif refinement['aspect'] == 'type':
            params['event_type'] = refinement['value']
        
        return params
    
    return {}


def _parse_cached_result(cached_text: str, original_query: str) -> MelbourneDiscoveryResponse:
    """Wrap cached text back into a MelbourneDiscoveryResponse."""
    lines = [l.strip() for l in cached_text.splitlines() if l.strip()]
    return MelbourneDiscoveryResponse(
        query=original_query,
        city="Melbourne",
        events_found=lines[:10],
        news_highlights=[],
        recommendations=["Results from cache — search again for the latest updates"],
        budget_friendly_options=[],
        friend_group_suggestions=[],
        sources=[],
        tools_used=["cache"]
    )


def format_search_results(results):
    """Format results as conversational text"""
    if not results:
        return "I couldn't find any events matching those criteria. Would you like to try different parameters?"
    
    text = f"I found {len(results.events_found)} events for you!\n\n"
    
    for i, event in enumerate(results.events_found[:5], 1):
        text += f"{i}. {event}\n"
    
    text += f"\nWould you like me to:\n"
    text += f"- Refine the search?\n"
    text += f"- See budget options?\n"
    text += f"- Get more details about any event?"
    
    return text


# Page configuration
st.set_page_config(
    page_title="Melbourne Event Intelligence",
    page_icon="🎭",
    layout="wide"
)

# Header
st.title("🎭 Melbourne Event Intelligence Assistant")
st.caption("Your conversational AI guide to Melbourne events")

# Sidebar - Conversation history & context
with st.sidebar:
    st.header("📊 Conversation Context")
    
    top_prefs = rag.get_top_preferences() if rag else []
    if top_prefs:
        st.subheader("Your Top Interests")
        for pref in top_prefs:
            st.caption(f"⭐ {pref}")

    if st.session_state.user_context['search_history']:
        st.subheader("Recent Searches")
        for search in st.session_state.user_context['search_history'][-3:]:
            source_icon = "💾" if search.get('source') == 'memory' else "🔍"
            st.caption(f"{source_icon} {search['query']}")
    
    if st.button("🔄 Start New Conversation"):
        st.session_state.messages = []
        st.session_state.user_context = {
            'preferences': {},
            'search_history': [],
            'current_results': None
        }
        st.session_state.agent_memory = []
        st.rerun()

# Main chat interface
st.divider()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and 'results' in message:
            # Show structured results
            results = message['results']
            
            tab1, tab2, tab3 = st.tabs(["Events", "Recommendations", "Sources"])
            
            with tab1:
                for i, event in enumerate(results.events_found, 1):
                    st.write(f"{i}. {event}")
            
            with tab2:
                for rec in results.recommendations:
                    st.write(f"• {rec}")
            
            with tab3:
                for source in results.sources:
                    st.markdown(source)
        else:
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about Melbourne events..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Agent processes
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_conversational_agent(prompt)
        
        if response['type'] == 'results':
            # Show results
            results = response['content']
            st.write(response['text'])
            
            tab1, tab2, tab3 = st.tabs(["Events", "Recommendations", "Sources"])
            
            with tab1:
                if results.events_found:
                    for i, event in enumerate(results.events_found, 1):
                        st.write(f"{i}. {event}")
            
            with tab2:
                for rec in results.recommendations:
                    st.write(f"• {rec}")
            
            with tab3:
                if results.sources:
                    for source in results.sources:
                        st.markdown(source)
            
            # Add to messages with results
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['text'],
                "results": results
            })
        
        else:
            # Simple text response
            st.write(response['content'])
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['content']
            })
    
    st.rerun()

# Footer
st.divider()
st.caption("💡 Tip: You can refine searches by saying things like 'cheaper', 'this weekend', or 'something different'")