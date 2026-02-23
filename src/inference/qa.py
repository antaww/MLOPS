from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True, slots=True)
class QAConfig:
	device: str = "cuda"
	model_name: str = "Qwen/Qwen3-1.7B"


def config_from_env() -> QAConfig:
	device = os.getenv("QA_DEVICE", "cuda")
	model_name = os.getenv("QA_MODEL_NAME", "Qwen/Qwen3-1.7B")

	return QAConfig(
		device=device,
		model_name=model_name,
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
	print(f"[QA] Model loaded on device: {model.device}")

	return model, tokenizer, device


def answer_question(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	*,
	context: str,
	question: str,
) -> dict[str, str]:
	answer = generate_answer(
		model,
		tokenizer,
		context,
		question=question,
	)

	return {"answer": answer}


def build_prompt(context: str, question: str) -> str:
	return (
		f"Context:\n{context}\n\n"
		f"Question:\n{question}\n\n"
	)


def generate_answer(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	context: str,
	*,
	question: str,
) -> str:
	prompt = build_prompt(context, question)

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
	prompt = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
		thinking=True,
	)

	inputs = tokenizer(prompt, return_tensors="pt")
	inputs = {key: value.to(model.device) for key, value in inputs.items()}

	generated_ids = model.generate(
		**inputs,
		max_new_tokens=32768,
	)

	input_ids = inputs["input_ids"]
	output_ids = generated_ids[0][len(input_ids[0]) :].tolist()

	try:
		index = len(output_ids) - output_ids[::-1].index(151668)
	except ValueError:
		index = 0

	thinking_content = tokenizer.decode(
		output_ids[:index],
		skip_special_tokens=True,
	).strip("\n")
	content = tokenizer.decode(
		output_ids[index:],
		skip_special_tokens=True,
	).strip("\n")

	if "Answer:" in content:
		answer = content.split("Answer:")[-1].strip()
	else:
		answer = content.strip()

	return answer


def is_valid_answer(answer: str) -> bool:
	cleaned = answer.strip().lower()
	return bool(cleaned) and cleaned not in {"not found", "n/a", "unknown"}
