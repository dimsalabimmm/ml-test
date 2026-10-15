"""Run the saved sentiment classifier without training or internet access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "artifacts" / "rubert_tiny2_best"
MAX_LENGTH = 256


class SentimentPredictor:
    """Load one local checkpoint and reuse it for multiple predictions."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, device: str = "cpu"):
        model_dir = Path(model_dir)
        for filename in ("config.json", "model.safetensors", "tokenizer.json"):
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(
                    f"Не найден {model_dir / filename}. "
                    "Распакуйте архив rubert_tiny2_best.zip из Releases "
                    "в корень проекта."
                )

        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Apple MPS недоступен. Используйте --device cpu.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA недоступна. Используйте --device cpu.")
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Устройство должно быть cpu, mps или cuda.")

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="eager",
        ).float().to(self.device)
        self.model.eval()

    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        if batch_size < 1:
            raise ValueError("Размер батча должен быть положительным.")
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Передайте хотя бы один непустой текст.")

        results = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                probabilities = self.model(**inputs).logits.softmax(dim=-1).cpu()

                for text, scores in zip(batch_texts, probabilities):
                    label_id = int(scores.argmax().item())
                    results.append({
                        "text": text,
                        "label_id": label_id,
                        "sentiment": self.model.config.id2label[label_id],
                        "probabilities": {
                            self.model.config.id2label[i]: round(float(score), 6)
                            for i, score in enumerate(scores)
                        },
                    })
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Определение тональности текста")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", nargs="+", help="Один или несколько текстов")
    source.add_argument("--file", type=Path, help="UTF-8 файл, один текст на строку")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        texts = args.text
        if args.file is not None:
            texts = [
                line for line in args.file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        predictor = SentimentPredictor(args.model_dir, device=args.device)
        results = predictor.predict(texts, batch_size=args.batch_size)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Ошибка: {error}\n")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
