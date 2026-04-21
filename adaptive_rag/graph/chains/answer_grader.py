from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from graph.chains.llm import default_llm

class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="답변이 질문을 해결했는지 여부. 'yes' 또는 'no'"
    )

structured_llm_grader = default_llm.with_structured_output(GradeAnswer)

system = """당신은 평가자입니다. 주어진 답변이 원래의 질문을 온전히 해결하고 답변하는지 평가합니다.
주어진 답변이 질문을 잘 해결한다면 'yes', 해결하지 못하거나 관련이 없다면 'no'라는 이진 점수를 부여하세요."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "사용자 질문: \n\n {question} \n\n 생성된 답변: {generation}")
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader