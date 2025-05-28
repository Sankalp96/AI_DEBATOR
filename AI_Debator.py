from agno.agent import Agent
import asyncio
from textwrap import dedent
import os
import phi
import agno.api
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
from agno.playground import Playground,serve_playground_app
load_dotenv()
agno.api=os.getenv("AGNO_API_KEY")
from agno.team.team import Team

for_agent = Agent(
    name="For Agent",
    role="You should support the Topic",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=[
        "You must support the Topic using all the facts and knowledge you have",
    ],
)

against_agent = Agent(
    name="Against Agent",
    role="You should counter the Topic",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=[
        "You must contradict and oppose the Topic using all the facts and knowledge you have",
    ],
)


agent_team = Team(
    name="Debate Team",
    mode="collaborate",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    members=[
        for_agent,
        against_agent,
    ],
    instructions=[
        "You are a debate master.",
        "You have to stop the discussion after 3 rounds of each agent.",
    ],
    # success_criteria="The team has reached a consensus.",
    enable_agentic_context=True,
    show_tool_calls=True,
    markdown=True,
    show_members_responses=True,
)

# if __name__ == "__main__":
#     agent_team.print_response(
#         message="Start debate on the topic: Polythene: Boon or Curse?'",
#         stream=True,
#         stream_intermediate_steps=True,
#     )

app=Playground(teams=[agent_team,]).get_app()

if __name__=="__main__":
    serve_playground_app("AI_Debator:app",reload=True)

