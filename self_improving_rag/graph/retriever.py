"""
retriever.py
------------
The Retrieval agent: given a task description and a named retriever,
fetches relevant documents from the appropriate FAISS vector store.

Used for three specialist roles:
  - Regulatory Specialist  → fda_retriever
  - Medical Researcher     → pubmed_retriever
  - Ethics Specialist      → ethics_retriever

Uses a factory pattern so `knowledge_stores` is injected rather than
accessed as a global.
"""

from typing import Callable

from graph.states import TeamState, AgentOutput


def make_retrieval_agent(knowledge_stores: dict) -> Callable:
    """
    Factory that binds the `knowledge_stores` dict and returns a
    retrieval function with the signature:

        retrieval_agent(task_description, state, retriever_name, agent_name)
                        -> AgentOutput
    """

    def retrieval_agent(
        task_description: str,
        state: TeamState,
        retriever_name: str,
        agent_name: str,
    ) -> AgentOutput:
        """
        Retrieve documents relevant to `task_description` from
        `retriever_name` and return them as a single AgentOutput.

        For the Medical Researcher, `k` is overridden by the SOP's
        `researcher_retriever_k` setting, allowing the self-improvement
        loop to tune retrieval depth.
        """
        retriever = knowledge_stores[retriever_name]

        if retriever is None:
            return AgentOutput(
                agent_name=agent_name,
                findings=f"Retriever '{retriever_name}' is not available.",
            )

        # Allow the SOP to tune the Medical Researcher's retrieval depth
        if agent_name == "Medical Researcher":
            retriever.search_kwargs['k'] = state['sop'].researcher_retriever_k
            print(f"[{agent_name}] Using k={state['sop'].researcher_retriever_k} for retrieval.")

        retrieved_docs = retriever.invoke(task_description)

        # Concatenate docs with source metadata as context for the synthesizer
        findings = "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}"
            for doc in retrieved_docs
        )
        print(f"[{agent_name}] Retrieved {len(retrieved_docs)} documents.")
        print(f"[{agent_name}] Sample:\n{findings[:300]}...")

        return AgentOutput(agent_name=agent_name, findings=findings)

    return retrieval_agent
