# Melbourne Event Discovery Agent

AI agent that finds local events and news in Melbourne, Australia.

## What it does

- Searches for events (concerts, meetups, festivals, sports)
- Finds Melbourne news on any topic
- Filters by budget
- Suggests which friends to invite
- Saves results to text files

## Setup

1. **Install dependencies**
```bash
   pip install -r requirements.txt
```

2. **Add your API key**
```bash
   cp .env.example .env
```
   
   Edit `.env` and add your key:
```
   ANTHROPIC_API_KEY=sk-ant-your-key-here
```
   
   Get a key at: https://console.anthropic.com/

3. **Run**
```bash
   python main.py
```

## Usage

Choose what to search:
- **1** = Events only
- **2** = News only  
- **3** = Both

Enter your interests and budget. Results save to `melbourne_events_[timestamp].txt`

## Requirements

- Python 3.11+
- Anthropic API key

## Files

- `main.py` - Main agent
- `tools.py` - Search tools
- `.env` - Your API key (not tracked by git)
- `.env.example` - Template

## Cost

~$0.01-0.05 per search with Claude API

## Note

Always verify event details on official websites before attending.
