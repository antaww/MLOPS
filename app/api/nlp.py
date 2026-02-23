from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.inference.qa import QAConfig, answer_question

router = APIRouter()


class QARequest(BaseModel):
	context: str
	question: str


@router.post("/nlp/qa")
def answer(request: Request, payload: QARequest) -> dict[str, str]:
	config: QAConfig | None = getattr(request.app.state, "qa_config", None)
	model = getattr(request.app.state, "qa_model", None)
	tokenizer = getattr(request.app.state, "qa_tokenizer", None)

	if config is None or model is None or tokenizer is None:
		raise HTTPException(status_code=503, detail="QA model not loaded")

	return answer_question(
		model,
		tokenizer,
		context=payload.context,
		question=payload.question,
	)
