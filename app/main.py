from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import sys

if __name__ == "__main__":
	sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI

from src.inference.qa import config_from_env as qa_config_from_env, load_model as load_qa_model
from src.inference.stt import config_from_env as stt_config_from_env, load_model as load_stt_model

from app.api.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
	qa_config = qa_config_from_env()
	qa_model, qa_tokenizer, qa_device = load_qa_model(qa_config)
	stt_config = stt_config_from_env()
	stt_model = load_stt_model(stt_config)

	app.state.qa_config = qa_config
	app.state.qa_device = qa_device
	app.state.qa_model = qa_model
	app.state.qa_tokenizer = qa_tokenizer
	app.state.stt_config = stt_config
	app.state.stt_model = stt_model

	yield


app = FastAPI(lifespan=lifespan, title="mlops")
app.include_router(api_router)

if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
