"""
Generation chain — produces answers from retrieved documents via OpenRouter.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graph.chains.llm import default_llm

# ── Prompt: Korean competition law domain ──────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 한국 공정거래 판례를 분석하는 법률 전문가입니다.\n"
     "아래 제공된 문서를 근거로 질문에 답변하세요.\n\n"
     "규칙:\n"
     "1. 반드시 제공된 문서에 근거하여 답변하세요.\n"
     "2. 관련 판례명, 의결 번호, 날짜 등을 가능한 한 인용하세요.\n"
     "3. 문서에 답변 근거가 없으면 '제공된 문서에서 관련 정보를 찾을 수 없습니다'라고 답하세요.\n"
     "4. 답변은 명확하고 구조적으로 작성하세요."),
    ("human",
     "문서:\n{context}\n\n"
     "질문: {question}"),
])

from pydantic import BaseModel, Field
from typing import List

class Reference(BaseModel):
    source: str = Field(description="The source file name, eg 'case_1.pdf'")
    page: int = Field(description="The matching page number.")
    snippet: str = Field(description="A short quote from the document.")

class GenerationOutput(BaseModel):
    answer: str = Field(description="The generated answer in Markdown.")
    references: List[Reference] = Field(description="List of cited documents.", default_factory=list)

# ── LCEL chain ─────────────────────────────────────────────────────────────
# Input: {"context": <docs>, "question": <str>}
# Output: GenerationOutput
generation_chain = prompt | default_llm.with_structured_output(GenerationOutput)