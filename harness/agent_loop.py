from __future__ import annotations

import json
from dataclasses import dataclass

from .providers import Provider
from .thought_process import ThoughtProcessRecorder
from .tools import ToolHandler


@dataclass
class AgentLoopResult:
    reason: str
    iterations: int


def run_agent_loop(
    provider: Provider,
    tool_handler: ToolHandler,
    recorder: ThoughtProcessRecorder,
    system_prompt: str,
    initial_message: str,
    max_iterations: int,
) -> AgentLoopResult:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": initial_message})

    iteration = 0
    while True:
        iteration += 1
        response = provider.complete(messages, tool_handler.available_tools_for_llm())

        assistant_msg: dict = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args),
                    },
                }
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        if not response.tool_calls:
            recorder.record_iteration(iteration, response, [])
            return AgentLoopResult(reason="natural_termination", iterations=iteration)

        if iteration >= max_iterations:
            recorder.record_iteration(iteration, response, [])
            return AgentLoopResult(reason="iteration_limit", iterations=iteration)


     ############# GET THE TOOLS   #########################
        tool_results: list[dict] = []
        for tc in response.tool_calls:
            result = tool_handler.execute(tc)
            tool_results.append({"call_id": tc.call_id, "content": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": result,
                }
            )
        recorder.record_iteration(iteration, response, tool_results)
