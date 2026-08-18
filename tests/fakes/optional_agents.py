"""Test-only stub for the optional OpenAI Agents SDK."""

import sys
import types


def install_fake_agents() -> None:
    if "agents" in sys.modules:
        return
    agents = types.ModuleType("agents")

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AgentOutputSchema:
        def __init__(self, output_type, strict_json_schema=False):
            self.output_type = output_type
            self.strict_json_schema = strict_json_schema

    class Runner:
        @staticmethod
        async def run(*args, **kwargs):
            raise AssertionError("Tests must override the orchestrator planner")

    def function_tool(function=None, **kwargs):
        if function is None:
            return lambda decorated: decorated
        return function

    agents.Agent = Agent
    agents.AgentOutputSchema = AgentOutputSchema
    agents.Runner = Runner
    agents.function_tool = function_tool
    sys.modules["agents"] = agents
