from langchain.tools import tool
from datetime import datetime
from langchain_ollama.llms import OllamaLLM
from langchain.agents import initialize_agent, AgentType


@tool
def get_current_time(dummy_ip: str) -> str:
    """Returns the current time as a string."""
    print("Dummy IP:", dummy_ip)
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


model = OllamaLLM(model="gemma3:4b")

# Don't wrap it again — already a tool
agent = initialize_agent(
    tools=[get_current_time],  # use the tool directly
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

while True:
    user_input = input("Ask a question (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    response = agent.run(user_input)
    print(response)
