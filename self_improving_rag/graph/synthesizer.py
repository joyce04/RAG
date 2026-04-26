"""
synthesizer.py
--------------
The Criteria Synthesizer: the final node in the Team graph.

Aggregates findings from all specialist agents and produces the formal
'Inclusion and Exclusion Criteria' document.

The model used here is controlled by `sop.synthesizer_model`, so the
self-improvement loop can swap it out as part of a mutation.
"""

from langchain_community.chat_models import ChatOllama

from graph.states import TeamState


def criteria_synthesizer(state: TeamState) -> dict:
    """
    LangGraph node: synthesize all agent findings into final criteria.

    Flow:
      1. Reads the synthesizer model and prompt from the active SOP.
      2. Concatenates every specialist's findings into a single context block.
      3. Calls the synthesizer LLM and writes the result to `final_criteria`.
    """
    sop = state['sop']

    # Instantiate the synthesizer LLM from the SOP's model setting.
    # Done here (not at import time) so mutations in the SOP model name
    # take effect on the next run without restarting the process.
    synthesizer_llm = ChatOllama(model=sop.synthesizer_model, temperature=0.2)

    # Build a labelled context block from every specialist's output
    context = "\n\n---\n\n".join(
        f"**{out.agent_name} Findings:**\n{out.findings}"
        for out in state['agent_outputs']
    )

    prompt = f"{sop.synthesizer_prompt}\n\n**Context from Specialist Teams:**\n{context}"
    print(f"[Synthesizer] Using model '{sop.synthesizer_model}' to generate final criteria.")

    response = synthesizer_llm.invoke(prompt)
    print("[Synthesizer] Final criteria generated.")

    return {**state, "final_criteria": response.content}
