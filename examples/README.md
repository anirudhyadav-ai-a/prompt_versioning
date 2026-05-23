# Examples — Prompt Versioning & Drift Detection

> Runnable code samples demonstrating prompt lifecycle management for LLM systems.

---

## The Three Patterns

| Pattern | Folder | Description |
|---------|--------|-------------|
| Prompt Registry | [`prompt_registry/`](prompt_registry/) | Version prompts as code: store, diff, rollback |
| Drift Detection | [`drift_detector/`](drift_detector/) | Detect output drift after model updates using embedding similarity |
| A/B Testing | [`ab_tester/`](ab_tester/) | Route traffic between prompt versions, collect metrics, pick winner |

---

## Running Examples

Each pattern has a standalone `main.py`:

```bash
# Install from repo root
pip install -e ".[all]"

# Run individual patterns
python -m prompt_versioning.examples.prompt_registry.main
python -m prompt_versioning.examples.drift_detector.main
python -m prompt_versioning.examples.ab_tester.main
```

Set your `OPENAI_API_KEY` in a `.env` file (see `.env.example` at the repo root).

## Running Tests

```bash
pytest prompt_versioning/ -v
```
