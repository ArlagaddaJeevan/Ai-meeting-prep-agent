"""
AI Meeting Preparation Agent
A multi-agent system using CrewAI to prepare comprehensive meeting briefs.
"""

import streamlit as st
from dataclasses import dataclass
from typing import Optional
import os

from crewai import Agent, Task, Crew
from crewai.process import Process
from crewai_tools import SerperDevTool
from langchain_anthropic import ChatAnthropic


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MeetingConfig:
    """Configuration for meeting preparation."""
    company_name: str
    meeting_objective: str
    attendees: str
    duration_minutes: int
    focus_areas: str


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    model: str = "claude-3-haiku-20240307"
    temperature: float = 0.7


# =============================================================================
# Agent Factory
# =============================================================================

class AgentFactory:
    """Factory for creating specialized meeting preparation agents."""
    
    def __init__(self, llm: ChatAnthropic, search_tool: Optional[SerperDevTool] = None):
        self.llm = llm
        self.search_tool = search_tool
    
    def create_context_analyzer(self) -> Agent:
        return Agent(
            role="Meeting Context Specialist",
            goal="Analyze and summarize key background information for the meeting",
            backstory=(
                "You are an expert at quickly understanding complex business contexts "
                "and identifying critical information. You excel at research and synthesis."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.search_tool] if self.search_tool else []
        )
    
    def create_industry_expert(self) -> Agent:
        return Agent(
            role="Industry Expert",
            goal="Provide in-depth industry analysis and identify key trends",
            backstory=(
                "You are a seasoned industry analyst with deep expertise in market dynamics. "
                "You have a knack for spotting emerging trends and strategic opportunities."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.search_tool] if self.search_tool else []
        )
    
    def create_strategist(self) -> Agent:
        return Agent(
            role="Meeting Strategist",
            goal="Develop a tailored meeting strategy and detailed agenda",
            backstory=(
                "You are a master meeting planner known for creating highly effective "
                "strategies and agendas that drive productive outcomes."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_briefing_creator(self) -> Agent:
        return Agent(
            role="Communication Specialist",
            goal="Synthesize information into concise and impactful briefings",
            backstory=(
                "You are an expert communicator skilled at distilling complex information "
                "into clear, actionable insights that executives can quickly digest."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )


# =============================================================================
# Task Factory
# =============================================================================

class TaskFactory:
    """Factory for creating meeting preparation tasks with explicit chaining."""
    
    def __init__(self, config: MeetingConfig):
        self.config = config
    
    def create_context_analysis_task(self, agent: Agent) -> Task:
        return Task(
            description=f"""
Analyze the context for the meeting with {self.config.company_name}.

## Meeting Details
- **Objective**: {self.config.meeting_objective}
- **Attendees**: {self.config.attendees}
- **Duration**: {self.config.duration_minutes} minutes
- **Focus Areas**: {self.config.focus_areas}

## Research Requirements
Thoroughly research {self.config.company_name}:
1. Recent news and press releases (last 6 months)
2. Key products, services, and value proposition
3. Major competitors and market position
4. Leadership team and recent organizational changes
5. Financial performance indicators (if publicly available)

## Deliverable
Provide a comprehensive summary highlighting information most relevant to the meeting context.
Use markdown formatting with clear headings and subheadings.
            """,
            agent=agent,
            expected_output=(
                "A detailed markdown analysis of meeting context and company background, "
                "including recent developments, competitive position, and relevance to meeting objective."
            )
        )
    
    def create_industry_analysis_task(
        self, 
        agent: Agent, 
        context_task: Task
    ) -> Task:
        return Task(
            description=f"""
Based on the context analysis, provide an in-depth industry analysis for {self.config.company_name}.

## Analysis Areas
1. **Industry Trends**: Key developments shaping the industry
2. **Competitive Landscape**: Major players and their positioning
3. **Opportunities**: Growth areas and untapped potential
4. **Threats**: Risks and challenges facing the industry
5. **Market Positioning**: Where {self.config.company_name} fits in the ecosystem

## Requirements
- Ensure analysis is directly relevant to: {self.config.meeting_objective}
- Consider the perspectives of attendees: {self.config.attendees}
- Use markdown formatting with clear structure
            """,
            agent=agent,
            context=[context_task],  # Explicit task chaining
            expected_output=(
                "A comprehensive markdown industry analysis including trends, competitive landscape, "
                "opportunities, threats, and strategic insights relevant to the meeting."
            )
        )
    
    def create_strategy_task(
        self, 
        agent: Agent, 
        context_task: Task,
        industry_task: Task
    ) -> Task:
        return Task(
            description=f"""
Develop a tailored meeting strategy and detailed agenda for the {self.config.duration_minutes}-minute meeting.

## Agenda Requirements
Create a time-boxed agenda including:
1. **Opening** (5-10 min): Introductions and objective alignment
2. **Core Sections**: Key discussion topics with time allocations
3. **Closing** (5-10 min): Summary, action items, and next steps

## For Each Agenda Item Include
- Clear objective and expected outcome
- Key talking points (3-5 per section)
- Suggested lead speaker
- Discussion questions to drive conversation
- Potential objections and how to address them

## Strategic Considerations
- Address focus areas: {self.config.focus_areas}
- Align with objective: {self.config.meeting_objective}
- Optimize for attendee roles: {self.config.attendees}

Use markdown formatting with clear structure.
            """,
            agent=agent,
            context=[context_task, industry_task],  # Explicit task chaining
            expected_output=(
                "A detailed markdown meeting strategy with time-boxed agenda, talking points, "
                "discussion questions, and strategies for specific focus areas."
            )
        )
    
    def create_executive_brief_task(
        self, 
        agent: Agent,
        context_task: Task,
        industry_task: Task,
        strategy_task: Task
    ) -> Task:
        return Task(
            description=f"""
Synthesize all gathered information into a comprehensive executive brief for the meeting with {self.config.company_name}.

## Brief Components

### 1. Executive Summary (One Page)
- Meeting objective statement
- Key attendees and their roles
- Critical background points about {self.config.company_name}
- Top 3-5 strategic goals for the meeting
- Meeting structure overview

### 2. Key Talking Points
For each point include:
- Supporting data or statistics
- Relevant examples or case studies
- Connection to company's current situation

### 3. Q&A Preparation
- Anticipated questions based on attendee roles
- Data-driven response strategies
- Supporting context for complex questions

### 4. Strategic Recommendations
- 3-5 actionable recommendations
- Clear next steps with ownership
- Suggested timelines
- Risk mitigation strategies

## Formatting Requirements
- Use markdown with H1, H2, H3 headings
- Include bullet points and numbered lists where appropriate
- Bold key information for quick scanning
- Structure for easy navigation during the meeting

Ensure alignment with objective: {self.config.meeting_objective}
            """,
            agent=agent,
            context=[context_task, industry_task, strategy_task],  # Full context chain
            expected_output=(
                "A comprehensive markdown executive brief with summary, talking points, "
                "Q&A preparation, and strategic recommendations."
            )
        )


# =============================================================================
# Crew Manager
# =============================================================================

class MeetingPrepCrew:
    """Manages the meeting preparation crew and execution."""
    
    def __init__(self, anthropic_api_key: str, serper_api_key: str, llm_config: LLMConfig = None):
        self.llm_config = llm_config or LLMConfig()
        self._setup_environment(anthropic_api_key, serper_api_key)
        self._initialize_components(anthropic_api_key)
    
    def _setup_environment(self, anthropic_key: str, serper_key: str) -> None:
        """Set up environment variables for API access."""
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        os.environ["SERPER_API_KEY"] = serper_key
    
    def _initialize_components(self, api_key: str) -> None:
        """Initialize LLM and tools."""
        self.llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=self.llm_config.temperature,
            anthropic_api_key=api_key,
        )
        self.search_tool = SerperDevTool()
        self.agent_factory = AgentFactory(self.llm, self.search_tool)
    
    def prepare_meeting(self, config: MeetingConfig) -> str:
        """
        Execute the meeting preparation workflow.
        
        Args:
            config: Meeting configuration details
            
        Returns:
            Generated meeting preparation brief as markdown string
            
        Raises:
            RuntimeError: If crew execution fails
        """
        # Create agents
        context_analyzer = self.agent_factory.create_context_analyzer()
        industry_expert = self.agent_factory.create_industry_expert()
        strategist = self.agent_factory.create_strategist()
        briefing_creator = self.agent_factory.create_briefing_creator()
        
        # Create tasks with explicit chaining
        task_factory = TaskFactory(config)
        
        context_task = task_factory.create_context_analysis_task(context_analyzer)
        industry_task = task_factory.create_industry_analysis_task(industry_expert, context_task)
        strategy_task = task_factory.create_strategy_task(strategist, context_task, industry_task)
        brief_task = task_factory.create_executive_brief_task(
            briefing_creator, context_task, industry_task, strategy_task
        )
        
        # Assemble and run crew
        crew = Crew(
            agents=[context_analyzer, industry_expert, strategist, briefing_creator],
            tasks=[context_task, industry_task, strategy_task, brief_task],
            verbose=True,
            process=Process.sequential
        )
        
        try:
            result = crew.kickoff()
            # Handle different CrewAI versions
            return result.raw if hasattr(result, 'raw') else str(result)
        except Exception as e:
            raise RuntimeError(f"Meeting preparation failed: {str(e)}") from e


# =============================================================================
# Streamlit UI
# =============================================================================

def render_sidebar() -> tuple[str, str]:
    """Render sidebar with API key inputs and instructions."""
    st.sidebar.header("🔑 API Keys")
    
    anthropic_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        help="Required for Claude LLM access"
    )
    serper_key = st.sidebar.text_input(
        "Serper API Key",
        type="password",
        help="Required for web search capabilities"
    )
    
    st.sidebar.divider()
    st.sidebar.markdown("""
    ## 📖 How to Use
    
    1. Enter your API keys above
    2. Fill in meeting details
    3. Click **Prepare Meeting**
    
    ## 🤖 What Happens
    
    Four AI agents collaborate to:
    - Research company background
    - Analyze industry trends
    - Develop meeting strategy
    - Create executive brief
    
    *This may take 2-5 minutes.*
    """)
    
    return anthropic_key, serper_key


def render_meeting_form() -> Optional[MeetingConfig]:
    """Render meeting details form and return config if valid."""
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input(
            "Company Name",
            placeholder="e.g., Acme Corporation"
        )
        meeting_objective = st.text_input(
            "Meeting Objective",
            placeholder="e.g., Discuss partnership opportunities"
        )
        meeting_duration = st.slider(
            "Duration (minutes)",
            min_value=15,
            max_value=180,
            value=60,
            step=15
        )
    
    with col2:
        attendees = st.text_area(
            "Attendees & Roles",
            placeholder="John Smith - CEO\nJane Doe - VP Sales",
            height=100
        )
        focus_areas = st.text_input(
            "Focus Areas / Concerns",
            placeholder="e.g., Budget constraints, timeline"
        )
    
    # Validation
    if not company_name or not meeting_objective:
        return None
    
    return MeetingConfig(
        company_name=company_name,
        meeting_objective=meeting_objective,
        attendees=attendees or "Not specified",
        duration_minutes=meeting_duration,
        focus_areas=focus_areas or "General discussion"
    )


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="AI Meeting Agent",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 AI Meeting Preparation Agent")
    
    # Sidebar
    anthropic_key, serper_key = render_sidebar()
    
    # Check API keys
    if not (anthropic_key and serper_key):
        st.warning("⚠️ Please enter both API keys in the sidebar to continue.")
        st.stop()
    
    # Meeting form
    st.subheader("Meeting Details")
    meeting_config = render_meeting_form()
    
    if not meeting_config:
        st.info("Please fill in the company name and meeting objective to proceed.")
        st.stop()
    
    # Execute
    st.divider()
    
    if st.button("🚀 Prepare Meeting", type="primary", use_container_width=True):
        try:
            crew_manager = MeetingPrepCrew(anthropic_key, serper_key)
            
            with st.spinner("🤖 AI agents are preparing your meeting... This may take a few minutes."):
                result = crew_manager.prepare_meeting(meeting_config)
            
            st.success("✅ Meeting preparation complete!")
            st.divider()
            st.markdown(result)
            
            # Download option
            st.download_button(
                label="📥 Download Brief",
                data=result,
                file_name=f"meeting_brief_{meeting_config.company_name.replace(' ', '_')}.md",
                mime="text/markdown"
            )
            
        except RuntimeError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            st.info("Please check your API keys and try again.")


if __name__ == "__main__":
    main()