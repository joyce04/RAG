from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from graph.chains.llm import default_llm

class GradeHallucinations(BaseModel):
    # if answer is grounded in facts then return boolean yes, else no
    binary_score: bool = Field(
        description="답변이 제공된 사실에 근거하고 있는지 여부. 'yes' 또는 'no'"
    )

structured_llm_grader = default_llm.with_structured_output(GradeHallucinations)

system = """당신은 생성된 답변이 제공된 문서(사실)에 근거하거나 뒷받침되는지 평가하는 평가자입니다.
답변이 제공된 문서의 내용에 잘 근거하고 있다면 이진 점수 'yes'를, 그렇지 않고 내용이 없거나 위배된다면 'no'를 부여하세요."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "제공된 문서(사실): \n\n {documents} \n\n 생성된 답변: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader