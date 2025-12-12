# AI Meeting Preparation Agent

An intelligent meeting preparation tool that uses multiple AI agents to research companies, analyze industries, and generate comprehensive meeting briefs.

## Overview

This application leverages CrewAI to orchestrate four specialized AI agents that work together to prepare you for important meetings. Simply enter your meeting details, and the agents will research the company, analyze industry trends, develop a meeting strategy, and create an executive brief.

## Features

- **Company Research**: Automatically researches recent news, products, competitors, and key developments
- **Industry Analysis**: Identifies trends, opportunities, threats, and market positioning
- **Meeting Strategy**: Creates time-boxed agendas with talking points and discussion questions
- **Executive Brief**: Generates a comprehensive document with Q&A preparation and recommendations
- **Download Option**: Export your meeting brief as a markdown file

## Requirements

- Python 3.11
- Anthropic API Key
- Serper API Key (for web search)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-meeting-prep-agent.git
   cd ai-meeting-prep-agent
   ```

2. Create a virtual environment:
   ```bash
   py -3.11 -m venv venv
   ```

3. Activate the virtual environment:
   
   Windows:
   ```bash
   .\venv\Scripts\Activate
   ```
   
   macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Enter your API keys in the sidebar

4. Fill in the meeting details:
   - Company Name
   - Meeting Objective
   - Attendees & Roles
   - Duration
   - Focus Areas / Concerns

5. Click **Prepare Meeting** and wait for the agents to complete their work

6. Review and download your meeting brief

## Getting API Keys

### Anthropic API Key
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key

### Serper API Key
1. Go to [serper.dev](https://serper.dev)
2. Sign up for a free account
3. Copy your API key from the dashboard

## Project Structure

```
ai-meeting-prep-agent/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## How It Works

The application uses four AI agents in sequence:

1. **Meeting Context Specialist**: Researches the company and gathers background information
2. **Industry Expert**: Analyzes industry trends and competitive landscape
3. **Meeting Strategist**: Develops a tailored agenda and talking points
4. **Communication Specialist**: Synthesizes everything into an executive brief

Each agent builds on the work of the previous agents, creating a comprehensive preparation package.

