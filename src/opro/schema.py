from pydantic import BaseModel


class PromptExample(BaseModel):
    prompt: str
    score: float


class ProblemExample(BaseModel):
    question: str
    answer: str
