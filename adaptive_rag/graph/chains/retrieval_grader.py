from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from graph.chains.llm import default_llm

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="문서가 질문과 관련이 있는지 여부. 'yes' 또는 'no'"
    )

structured_llm_grader = default_llm.with_structured_output(GradeDocuments)

system = """당신은 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 평가자입니다.
문서가 질문과 관련된 키워드 또는 의미적 맥락을 포함하고 있다면 관련이 있는 것으로 평가하세요.
문서가 질문과 관련이 있다면 'yes', 그렇지 않다면 'no'라는 이진 점수를 부여하세요."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "검색된 문서: \n\n {document} \n\n 사용자 질문: {question}"),
    ]
)

retrieval_grader: RunnableSequence = grade_prompt | structured_llm_grader