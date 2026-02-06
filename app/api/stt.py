from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.inference.stt import SttConfig, transcribe_file

router = APIRouter()


@router.post("/stt/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
	config: SttConfig | None = getattr(request.app.state, "stt_config", None)
	model = getattr(request.app.state, "stt_model", None)

	if model is None or config is None:
		raise HTTPException(status_code=503, detail="STT model not loaded")

	suffix = Path(file.filename or "audio").suffix

	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
		content = await file.read()
		temp_file.write(content)
		temp_path = Path(temp_file.name)

	try:
		result = transcribe_file(
			model,
			temp_path,
			beam_size=config.beam_size,
			language=config.language,
			vad_filter=config.vad_filter,
		)
	finally:
		try:
			os.remove(temp_path)
		except FileNotFoundError:
			pass

	return result
