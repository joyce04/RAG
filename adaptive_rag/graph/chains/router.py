from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from graph.chains.llm import default_llm

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="사용자 질문을 입력받아 websearch 또는 vectorstore 중 어디로 라우팅할지 선택합니다.",
    )


structured_llm_router = default_llm.with_structured_output(RouteQuery)

system = """당신은 사용자 질문을 vectorstore 또는 websearch로 라우팅하는 전문가입니다.
vectorstore에는 한국 공정거래법, 경쟁법 등과 관련된 판례와 의결서 내용이 포함되어 있습니다.
공정거래 및 판례와 관련된 질문에는 vectorstore를 사용하세요. 그 외의 최신 정보나 일반적인 사항에 대해서는 websearch를 사용하세요."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "질문: {question}"),
    ]
)

question_router: RunnableSequence = route_prompt | structured_llm_router