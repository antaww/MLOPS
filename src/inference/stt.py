from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass(frozen=True, slots=True)
class SttConfig:
	beam_size: int = 5
	compute_type: str = "int8"
	device: str = "cuda"
	language: str | None = None
	model_name: str = "large-v3-turbo"
	vad_filter: bool = True


def config_from_env() -> SttConfig:
	beam_size_raw = os.getenv("STT_BEAM_SIZE", "5")
	compute_type = os.getenv("STT_COMPUTE_TYPE", "int8")
	device = os.getenv("STT_DEVICE", "cuda")
	language = os.getenv("STT_LANGUAGE", "")
	model_name = os.getenv("STT_MODEL_NAME", "large-v3-turbo")
	vad_filter_raw = os.getenv("STT_VAD_FILTER", "true")

	beam_size = int(beam_size_raw)
	language_value = language or None
	vad_filter = vad_filter_raw.lower() in {"1", "true", "yes"}

	return SttConfig(
		beam_size=beam_size,
		compute_type=compute_type,
		device=device,
		language=language_value,
		model_name=model_name,
		vad_filter=vad_filter,
	)


def load_model(config: SttConfig) -> WhisperModel:
	return WhisperModel(
		config.model_name,
		compute_type=config.compute_type,
		device=config.device,
	)


def transcribe_file(
	model: WhisperModel,
	audio_path: str | Path,
	*,
	beam_size: int,
	language: str | None,
	vad_filter: bool,
) -> dict[str, str]:
	path = Path(audio_path)
	if not path.exists():
		raise FileNotFoundError(f"Audio file not found: {path}")

	segments, info = model.transcribe(
		str(path),
		beam_size=beam_size,
		language=language,
		vad_filter=vad_filter,
	)

	text_parts = [segment.text for segment in segments]
	text = "".join(text_parts).strip()

	return {"language": info.language, "text": text}


def transcribe(audio_path: str | Path, config: SttConfig | None = None) -> dict[str, str]:
	resolved_config = config or config_from_env()
	model = load_model(resolved_config)

	return transcribe_file(
		model,
		audio_path,
		beam_size=resolved_config.beam_size,
		language=resolved_config.language,
		vad_filter=resolved_config.vad_filter,
	)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Transcribe an audio file.")
	parser.add_argument("audio_path", type=str)
	args = parser.parse_args()

	result = transcribe(args.audio_path)
	print(result["text"])
