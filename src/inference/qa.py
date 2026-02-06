from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True, slots=True)
class QAConfig:
	device: str = "cpu"
	max_chunk_chars: int = 2000
	max_new_tokens: int = 80
	model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
	temperature: float = 0.2


def config_from_env() -> QAConfig:
	device = os.getenv("QA_DEVICE", "cpu")
	max_chunk_chars_raw = os.getenv("QA_MAX_CHUNK_CHARS", "2000")
	max_new_tokens_raw = os.getenv("QA_MAX_NEW_TOKENS", "80")
	model_name = os.getenv("QA_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
	temperature_raw = os.getenv("QA_TEMPERATURE", "0.2")

	return QAConfig(
		device=device,
		max_chunk_chars=int(max_chunk_chars_raw),
		max_new_tokens=int(max_new_tokens_raw),
		model_name=model_name,
		temperature=float(temperature_raw),
	)


def load_model(config: QAConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
	device = config.device
	if device == "cuda" and not torch.cuda.is_available():
		device = "cpu"

	dtype = torch.float16 if device == "cuda" else torch.float32

	model = AutoModelForCausalLM.from_pretrained(
		config.model_name,
		torch_dtype=dtype,
	)
	tokenizer = AutoTokenizer.from_pretrained(config.model_name)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	model.to(device)

	return model, tokenizer, device


def answer_question(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	*,
	context: str,
	max_chunk_chars: int,
	max_new_tokens: int,
	question: str,
	temperature: float,
) -> dict[str, str]:
	chunks = split_into_chunks(context, max_chunk_chars)
	best_answer = ""

	for chunk in chunks:
		answer = generate_answer(
			model,
			tokenizer,
			chunk,
			max_new_tokens=max_new_tokens,
			question=question,
			temperature=temperature,
		)
		if is_valid_answer(answer):
			return {"answer": answer}

		if not best_answer:
			best_answer = answer

	return {"answer": best_answer}


def build_prompt(context: str, question: str) -> str:
	return (
		"Answer the question using only the context. If the answer is not in the context, "
		"reply with \"not found\".\n\n"
		f"Context:\n{context}\n\n"
		f"Question:\n{question}\n\n"
		"Answer:"
	)


def generate_answer(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	context: str,
	*,
	max_new_tokens: int,
	question: str,
	temperature: float,
) -> str:
	prompt = build_prompt(context, question)

	if hasattr(tokenizer, "apply_chat_template"):
		messages = [
			{
				"content": "You are a helpful assistant that answers questions using the provided context.",
				"role": "system",
			},
			{
				"content": prompt,
				"role": "user",
			},
		]
		prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

	inputs = tokenizer(prompt, return_tensors="pt")
	inputs = {key: value.to(model.device) for key, value in inputs.items()}

	outputs = model.generate(
		**inputs,
		do_sample=temperature > 0,
		max_new_tokens=max_new_tokens,
		temperature=temperature,
	)

	decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

	if "Answer:" in decoded:
		answer = decoded.split("Answer:")[-1].strip()
	else:
		answer = decoded.strip()

	return answer


def is_valid_answer(answer: str) -> bool:
	cleaned = answer.strip().lower()
	return bool(cleaned) and cleaned not in {"not found", "n/a", "unknown"}


def split_into_chunks(text: str, max_chunk_chars: int) -> list[str]:
	sentences = [segment.strip() for segment in text.replace("\n", " ").split(".") if segment.strip()]
	if not sentences:
		return [text.strip()]

	chunks: list[str] = []
	current = ""

	for sentence in sentences:
		candidate = f"{current} {sentence}.".strip() if current else f"{sentence}."
		if len(candidate) > max_chunk_chars and current:
			chunks.append(current)
			current = f"{sentence}."
		else:
			current = candidate

	if current:
		chunks.append(current)

	return chunks
