import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from tools import event_search_tool, news_search_tool, save_tool, budget_tool

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


# Initialize Claude
@st.cache_resource
def init_llm():
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    parser = PydanticOutputParser(pydantic_object=MelbourneDiscoveryResponse)

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

    IMPORTANT FOR SOURCES:
    - Extract ALL URLs from the tool results
    - Include them in the sources field
    - Format each source as: [Title](URL)
    - Only include genuine URLs from search results
    - Never make up or guess URLs

    After gathering all information, provide a comprehensive response with event details,
    news highlights, recommendations, friend group suggestions and genuine source links.

    {format_instructions}
    """

    tools = [event_search_tool, news_search_tool, save_tool, budget_tool]
    llm_with_tools = llm.bind_tools(tools)

    return llm, llm_with_tools, parser, system_prompt, tools


llm, llm_with_tools, parser, system_prompt, tools = init_llm()


# Agent function
def run_agent(query: str):
    """Simple agent that calls tools and processes results"""

    # First call to decide which tools to use
    response = llm_with_tools.invoke([
        ("system", system_prompt.format(format_instructions=parser.get_format_instructions())),
        ("human", query)
    ])

    results = []

    # Check if tools were called
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('args', {})

            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = tool.func(**tool_args)
                        results.append(f"Tool: {tool_name}\nResult: {result}")
                    except Exception as e:
                        error_msg = f"Error with {tool_name}: {str(e)}"
                        results.append(error_msg)
                    break

    # Combine all results
    combined_results = "\n\n".join(results) if results else "No tools were executed."

    # Second call to generate final structured response
    final_prompt = f"""
Based on the following tool results, provide a comprehensive answer about 
Melbourne events and news.

Tool Results:
{combined_results}

Original Query: {query}

IMPORTANT: Extract all URLs from the tool results and include them in the
sources field formatted as [Title](URL). Only use URLs that appear in the
tool results, never make up URLs.

Please provide a detailed response in the following JSON format:
{parser.get_format_instructions()}
"""

    final_response = llm.invoke(final_prompt)

    return final_response.content


# Page configuration
st.set_page_config(
    page_title="Melbourne Event Finder",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    .source-link {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎭 Melbourne Event Discovery Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Find local events and news in Melbourne, Australia powered by AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI-powered agent helps you discover:
    - 🎪 Local events and activities
    - 📰 Melbourne news updates
    - 💰 Budget-friendly options
    - 👥 Friend group suggestions
    - 🔗 Genuine links to sources
    """)

    st.divider()

    st.header("🔧 How It Works")
    st.write("""
    1. Choose what to search for
    2. Enter your preferences
    3. Set your budget
    4. Click Search
    5. Get AI-powered recommendations!
    """)

    st.divider()

    st.header("💡 Tips")
    st.write("""
    - Be specific with event types
    - Try budget $0 for free events
    - Check 'Both' for complete info
    - Click source links to verify details
    """)

    st.divider()

    st.caption("Powered by Claude AI & LangChain")

# Main content
st.divider()

# Search type selection
st.subheader("What would you like to search for?")
search_type = st.radio(
    "Select search type:",
    ["Events", "News", "Both"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# Input fields based on search type
col1, col2 = st.columns(2)

with col1:
    if search_type in ["Events", "Both"]:
        event_type = st.text_input(
            "🎭 Event Type",
            placeholder="e.g., tech meetups, concerts, food festivals, AFL games, comedy shows...",
            help="What type of events are you interested in?"
        )
    else:
        event_type = ""

with col2:
    if search_type in ["News", "Both"]:
        news_topic = st.text_input(
            "📰 News Topic",
            placeholder="e.g., local news, weather, traffic, politics...",
            help="What news topics interest you?"
        )
    else:
        news_topic = ""

# Budget slider
if search_type in ["Events", "Both"]:
    st.subheader("💰 Budget")
    budget = st.slider(
        "Maximum budget in AUD",
        min_value=0,
        max_value=500,
        value=50,
        step=10,
        help="Set to $0 for free events only"
    )
else:
    budget = 0

st.divider()

# Search button
if st.button("🔍 Search", type="primary", use_container_width=True):

    # Validate inputs
    if search_type == "Events" and not event_type:
        st.error("Please enter an event type!")
    elif search_type == "News" and not news_topic:
        st.error("Please enter a news topic!")
    elif search_type == "Both" and (not event_type or not news_topic):
        st.error("Please enter both event type and news topic!")
    else:
        # Build query
        if search_type == "Events":
            query = f"Find {event_type} events in Melbourne Australia with budget under ${budget} AUD"
        elif search_type == "News":
            query = f"Find latest news about {news_topic} in Melbourne Australia"
        else:
            query = f"Find {event_type} events in Melbourne Australia with budget under ${budget} AUD and also provide latest {news_topic} news in Melbourne"

        # Show query
        with st.expander("📝 Search Query"):
            st.code(query)

        # Run agent with loading spinner
        with st.spinner("🤖 AI Agent is searching... This may take 10-20 seconds"):
            try:
                output = run_agent(query)

                # Parse response
                try:
                    structured_response = parser.parse(output)

                    # Success message
                    st.success("✅ Search complete!")

                    st.divider()

                    # Display results in tabs
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "📅 Events",
                        "📰 News",
                        "⭐ Recommendations",
                        "💵 Budget Options",
                        "👥 Friend Suggestions",
                        "🔗 Sources"
                    ])

                    with tab1:
                        st.subheader(f"Events Found ({len(structured_response.events_found)})")
                        if structured_response.events_found:
                            for i, event in enumerate(structured_response.events_found, 1):
                                st.write(f"{i}. {event}")
                        else:
                            st.info("No specific events found. Check recommendations!")

                    with tab2:
                        st.subheader(f"News Highlights ({len(structured_response.news_highlights)})")
                        if structured_response.news_highlights:
                            for i, news in enumerate(structured_response.news_highlights, 1):
                                st.write(f"{i}. {news}")
                        else:
                            st.info("No news highlights available.")

                    with tab3:
                        st.subheader("Top Recommendations")
                        for i, rec in enumerate(structured_response.recommendations, 1):
                            st.write(f"{i}. {rec}")

                    with tab4:
                        st.subheader("Budget-Friendly Options")
                        if structured_response.budget_friendly_options:
                            for i, option in enumerate(structured_response.budget_friendly_options, 1):
                                st.write(f"{i}. {option}")
                        else:
                            st.info("No specific budget options found.")

                    with tab5:
                        st.subheader("Friend Group Suggestions")
                        for i, suggestion in enumerate(structured_response.friend_group_suggestions, 1):
                            st.write(f"{i}. {suggestion}")

                    with tab6:
                        st.subheader("Sources & Links")
                        st.caption("Click any link to visit the original source")
                        if structured_response.sources:
                            for i, source in enumerate(structured_response.sources, 1):
                                # Render as clickable markdown link
                                st.markdown(f"{i}. {source}")
                        else:
                            st.info("No sources available for this search.")

                    # Footer info
                    st.divider()
                    st.caption(f"🔧 Tools used: {', '.join(structured_response.tools_used)}")
                    st.caption(f"📍 City: {structured_response.city}")

                except Exception as parse_error:
                    st.warning("⚠️ Could not parse structured response. Showing raw output:")
                    st.text_area("Raw Output", output, height=400)
                    st.error(f"Parse Error: {str(parse_error)}")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Troubleshooting tips:")
                st.write("- Check your internet connection")
                st.write("- Verify API keys in .env file")
                st.write("- Try a simpler search query")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #64748B; padding: 2rem;'>
        <p><strong>⚠️ Important:</strong> Always verify event details on official websites before attending.</p>
        <p>Recommended sources: TimeOut Melbourne, Eventbrite, What's On Melbourne</p>
    </div>
""", unsafe_allow_html=True)