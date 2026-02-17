# Melbourne Event Discovery Agent

AI-powered agent that discovers local events and news in Melbourne, Australia using Claude AI and LangChain.

🌐 **Live Demo:** [https://melbourneeventdiscoveryagent.streamlit.app/]

## Features

- 🎭 **Event Discovery** - Find concerts, meetups, festivals, sports events, and more
- 📰 **News Search** - Get latest Melbourne news on any topic
- 💰 **Budget Filtering** - Find free or budget-friendly options
- 👥 **Friend Suggestions** - AI recommendations on who to invite
- 💾 **Save Results** - Export findings to formatted text files
- 🌐 **Web Interface** - Easy-to-use Streamlit app
- 💻 **CLI Version** - Command-line interface also available

## How I Built This

### The Journey

This project started as a learning exercise to understand AI agents and evolved into a full-featured web application. It took me about a week to create this agent. Here's what went into building it:

### Phase 1: Understanding AI Agents

**What I Learned:**

- AI agents can use tools to interact with the real world
- LangChain provides a framework for building agent workflows
- Claude can decide which tools to call based on user queries
- Structured outputs ensure consistent, parseable responses

**Key Concepts:**

- **Tool Calling** - How LLMs can invoke external functions
- **Agent Orchestration** - Managing multi-step workflows
- **Prompt Engineering** - Crafting instructions for optimal AI behavior
- **Output Parsing** - Converting AI responses to structured data

**Challenges:**

- Understanding the difference between agents and simple LLM calls
- Learning when to use tools vs. when to use model knowledge
- Debugging tool call failures and parsing errors

### Phase 2: Building the Core Agent

**Tools & Technologies:**

```
Python 3.11          → Primary language
LangChain            → Agent framework
Anthropic Claude API → AI reasoning engine
Pydantic             → Data validation
DuckDuckGo (ddgs)    → Web search
```

**What I Built:**

1. **Search Tools** - Web search functions for events and news
2. **Budget Analyzer** - Tool to filter by price
3. **File Saver** - Export results to text files
4. **Agent Loop** - Orchestration logic to call tools and process results

**Technical Learnings:**

- How to create LangChain tools using the `@tool` decorator
- Binding tools to an LLM with `.bind_tools()`
- Parsing tool calls from LLM responses
- Handling errors gracefully when tools fail
- Managing API rate limits and retry logic

**Code Structure:**

```python
# Basic agent pattern I learned:
1. User asks a question
2. LLM decides which tools to use
3. Execute tools and gather results
4. LLM synthesizes final answer
5. Parse and display structured output
```

### Phase 3: Environment Management & Security

**What I Learned:**

- **Environment Variables** - Keeping API keys secure
- **Git Security** - Using `.gitignore` to prevent key leaks
- **API Key Rotation** - What to do if keys are exposed

**Key Files:**

```
.env           → Local secrets (never commit!)
.env.example   → Template for others
.gitignore     → Prevents accidental commits
```

**Security Practices:**

- Always use `.gitignore` for sensitive files
- Never hardcode API keys in code
- Use `python-dotenv` to load environment variables
- Rotate keys immediately if exposed

**Mistakes I Made:**

- Initially committed `.env` to git (had to clean git history)
- Forgot to add `.env` to `.gitignore` at first
- Learned about `git filter-branch` to remove sensitive data

### Phase 4: Web Interface with Streamlit

**Why Streamlit:**

- Zero frontend coding required (no HTML/CSS/JavaScript)
- Built specifically for data/AI apps
- Free hosting on Streamlit Community Cloud
- Python-native (no context switching)

**What I Learned:**

- Streamlit's reactive programming model
- Component library (buttons, sliders, tabs, etc.)
- Custom CSS styling with `st.markdown`
- Session state management
- Caching with `@st.cache_resource`

**UI/UX Decisions:**

```python
# Organized results in tabs for better readability
tab1, tab2, tab3 = st.tabs(["Events", "News", "Recommendations"])

# Used columns for side-by-side inputs
col1, col2 = st.columns(2)

# Loading spinner for better UX
with st.spinner("Searching..."):
    results = run_agent(query)
```

**Challenges:**

- Understanding Streamlit's top-to-bottom execution model
- Managing state between reruns
- Styling components to look professional
- Handling long-running operations (agent calls)

### Phase 5: Deployment

**Deployment Platform: Streamlit Community Cloud**

**What I Learned:**

- How to deploy Python apps to the cloud
- Managing secrets in production environments
- Reading deployment logs for debugging
- Continuous deployment from GitHub

**Deployment Process:**

```
1. Push code to GitHub
2. Connect Streamlit to repository
3. Configure secrets (API keys)
4. Deploy with one click
5. Get public URL
```

**Production Considerations:**

- Requirements.txt must include ALL dependencies
- Package versions matter (compatibility issues)
- Cloud environment differs from local (some packages may not work)
- Need to handle missing dependencies gracefully

**Deployment Challenges:**

- DuckDuckGo package compatibility issues
- Learning about TOML format for Streamlit secrets
- Understanding the difference between local `.env` and cloud secrets
- Debugging production errors without local access

### Tools & Skills I Developed

**Programming:**

- Python async/await patterns
- Error handling and exception management
- Working with external APIs
- Data parsing and validation with Pydantic
- File I/O operations

**AI/ML:**

- Prompt engineering for Claude
- Tool use / function calling
- Agent design patterns
- Structured output generation
- Context window management

**DevOps:**

- Git version control
- Environment management
- Secrets management
- Cloud deployment
- CI/CD basics (auto-deploy on push)

**Web Development:**

- Streamlit framework
- Basic UI/UX principles
- Responsive design
- User input validation
- Loading states and error messages

### Key Takeaways

**What Worked Well:**

- Starting with CLI before building web UI
- Using structured outputs (Pydantic) from the start
- Iterative development (MVP → features → polish)
- Free tools (DuckDuckGo, Streamlit Cloud)

**What I'd Do Differently:**

- Set up `.gitignore` before first commit
- Write tests for tools earlier
- Document as I build (not after)
- Consider error cases from the beginning

**Unexpected Learnings:**

- AI agents are harder to debug than regular code
- Web search results vary significantly by query phrasing
- User experience matters more than I expected
- Deployment always reveals hidden assumptions

### Resources That Helped

**Documentation:**

- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [Anthropic Claude API](https://docs.anthropic.com/en/docs/welcome)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Pydantic Docs](https://docs.pydantic.dev/)

**Tutorials:**

- LangChain Agent tutorials on YouTube
- Streamlit gallery for UI inspiration
- Anthropic cookbook for prompt engineering

**Community:**

- LangChain Discord
- r/LangChain subreddit
- Streamlit community forum
- Stack Overflow

## Project Structure

```
melbourne-event-agent/
├── app.py                 # Streamlit web interface
├── main.py                # CLI version (original)
├── tools.py               # Search and utility tools
├── requirements.txt       # Python dependencies
├── .env                   # API keys (local only, not in git)
├── .env.example          # Template for API keys
├── .gitignore            # Protects sensitive files
└── README.md             # This file
```

## Quick Start

### Use the Live App

Visit: [https://melbourneeventdiscoveryagent.streamlit.app/]

### Run Locally

1. **Clone and install**

```bash
   git clone https://github.com/SwethaGot/AI-agent.git
   cd AI-agent
   pip install -r requirements.txt
```

2. **Set up API key**

```bash
   cp .env.example .env
   # Edit .env and add: ANTHROPIC_API_KEY=sk-ant-your-key
```

3. **Run**

```bash
   streamlit run app.py
```

## Usage

**Web Interface:**

1. Choose search type (Events/News/Both)
2. Enter preferences and budget
3. Click Search
4. View organized results in tabs

**CLI Version:**

```bash
python main.py
```

## Technologies

- **Python 3.11** - Programming language
- **Claude Sonnet 4.5** - AI reasoning
- **LangChain** - Agent framework
- **Streamlit** - Web interface
- **DuckDuckGo** - Web search
- **Pydantic** - Data validation
- **Git & GitHub** - Version control
- **Streamlit Cloud** - Hosting

## Cost

- **Claude API**: ~$0.01-0.05 per search
- **DuckDuckGo**: Free
- **Streamlit Hosting**: Free
- **Total**: A few dollars/month

## Troubleshooting

**Import Errors:**

```bash
pip install -r requirements.txt
```

**API Key Issues:**

- Check `.env` file exists locally
- For Streamlit Cloud: Settings → Secrets
- Ensure key starts with `sk-ant-`

**Search Not Working:**

- Verify internet connection
- Try simpler search terms
- Wait if rate-limited

## Future Enhancements

- [ ] Add more cities
- [ ] Calendar export (.ics)
- [ ] Email notifications
- [ ] User preferences
- [ ] Event recommendations
- [ ] Mobile app

## Contributing

Contributions welcome! Fork, branch, commit, push, PR.

## License

MIT License - feel free to use and modify

## Acknowledgments

- [Anthropic Claude](https://www.anthropic.com/)
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [DuckDuckGo](https://duckduckgo.com/)

## Youtube tutorials that helped

**Video Tutorials I Followed:**

1. **[Build an AI Agent From Scratch in Python - Tutorial for Beginners](https://www.youtube.com/watch?v=5g-PKQHArKk)**
   - What it covered: Basic agent concepts, tool creation, LangChain fundamentals
   - Why it was helpful: Great introduction to agent architecture and practical examples
   - Key takeaway: Understanding how agents decide which tools to use

2. **[LangChain Crash Course - Build Apps with Language Models](https://www.youtube.com/watch?v=LbT1yp6quS8)**
   - Topics: LangChain basics, chains, agents, memory
   - Application: Helped structure the agent workflow

3. **[Streamlit Course - Build Data Science Web Apps in Python](https://www.youtube.com/watch?v=VqgUkExPvLY)**
   - Topics: Building web apps with Streamlit, components, deployment
   - Application: Created the web interface from scratch

4. **[Deploy Your Python App to the Cloud for FREE](https://www.youtube.com/watch?v=HKoOBiAaHGg)**
   - Topics: Streamlit Cloud deployment, managing secrets, CI/CD
   - Application: Deploying the app and making it publicly accessible

---

**Built as a learning project to understand AI agents and web deployment**
