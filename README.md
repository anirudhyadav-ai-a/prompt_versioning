# Prompt Versioning & Drift Detection

> Treating prompts as deployable artifacts — versioning, semantic diff, drift detection, and A/B testing for production LLM systems.

This folder contains all artifacts for the whitepaper *"Prompt Versioning & Drift Detection"* and companion implementation examples.

### Who This Is For

ML engineers, platform teams, and engineering leaders running LLM-powered features in production — especially teams managing prompts across multiple models or environments

---

## Start Here

| Document | What It Is |
|---|---|
| [`WORKING_PLAN.md`](WORKING_PLAN.md) | **Implementation guide** — skeleton sections, architecture diagrams, and research notes |
| [`examples/`](examples/) | Runnable code samples demonstrating prompt lifecycle management |

---

## Quick Start

```bash
# From the repo root
pip install -e ".[all]"

# Run the prompt registry demo
python -m prompt_versioning.examples.prompt_registry.main

# Run drift detection demo
python -m prompt_versioning.examples.drift_detector.main

# Run A/B testing demo
python -m prompt_versioning.examples.ab_tester.main
```

### Example Output (Prompt Registry)

```
Registered: classify-intent v1
Duplicate check: returned v1 (expected v1)
Registered: classify-intent v2
Registered: classify-intent v3

--- Version History ---
  v1: Initial version [39d5379246fd...]
  v2: Added feedback category, clearer instructions [4b657900a9f8...]
  v3: Added escalation category, JSON output [cacbcb6b7308...]

--- Text Diff (v1 → v3) ---
  - Classify the user's intent into one of: greeting, question, complaint, other.
  + You are an intent classifier. Given a user message, classify it into exactly
    one category: greeting, question, complaint, feedback, escalation, other.
    Respond with JSON: {"intent": "...", "confidence": 0.X}

--- Rollback to v2 ---
  Created: v4 with content from v2
```

---

## Folder Layout

```
prompt_versioning/
├── WORKING_PLAN.md              ← implementation guide (start here)
├── README.md                    ← you are here
│
└── examples/                    ← runnable code samples
    ├── README.md                ← index of examples
    ├── prompt_registry/         ← version prompts as code: store, diff, rollback
    ├── drift_detector/          ← detect output drift after model updates
    └── ab_tester/               ← route traffic between prompt versions, pick winner
```

---

## Sections Covered

1. **Introduction** — why prompts are code and should be versioned
2. **Prompt Registry** — storing, diffing, and rolling back prompt versions
3. **Drift Detection** — detecting output changes after model updates
4. **A/B Testing** — comparing prompt versions in production
5. **Decision Framework** — when to version, when to test, when to roll back
6. **Cross-Model Prompt Management** — per-model drift profiles, cross-model A/B testing, model-specific prompt variants
7. **CI/CD Integration** — automated drift detection via GitHub Actions, pre-deploy checks, alerting
8. **Prompt Templating** — variable substitution and Jinja2 support for prompt templates

---

## Known Limitations

- **In-memory registry** — the reference implementation stores prompt versions in a Python dict. Production use requires a persistent store (Postgres, DynamoDB, etc.).
- **Embedding model dependency** — drift detection accuracy depends on the embedding model. Pin the embedding model version to avoid false positives.
- **Single-turn prompts only** — multi-turn conversation prompt chains are not yet supported.
- **LLM-as-judge variance** — A/B test scoring uses an LLM judge, which introduces non-determinism. Temperature=0 reduces but does not eliminate this.

See the [Limitations section in WORKING_PLAN.md](WORKING_PLAN.md#limitations) for the full discussion.

---

## VS Code Extension

This phase includes a VS Code extension with native GitHub Copilot integration.

**Chat Participant:** `@prompts`
**Commands:** register drift test

### Setup

```bash
cd vscodebase && npm install && npm run compile
# Press F5 in VS Code to launch
```

### Features

- **Workspace scanning** — discovers source files using `vscode.workspace.findFiles()`
- **Context building** — extracts symbols and code snippets for LLM prompts
- **Native Copilot** — uses `vscode.lm.selectChatModels()`, no API keys needed
- **14 unit tests** — covering extension activation, LLM client, workspace scanner, chat participant

---

## Prompt Templates

Located in `prompts/`:

- [`prompt-registry.md`](prompts/prompt-registry.md)
- [`drift-detection.md`](prompts/drift-detection.md)
- [`ab-testing.md`](prompts/ab-testing.md)

Use these templates:
- In Copilot Chat: `#file:prompts/prompt-registry.md` then ask your question
- Via the VS Code extension: the `@prompts` participant loads them automatically
- Standalone: copy the template into any LLM tool and fill the `{{variables}}`

---

## Acknowledgements

Implementation scaffolding was generated with using Claude and Devinai; architecture, technical content, and design decisions are the author's own.

---

## License

License
Free to Use — With Eyes Open
Copyright (c) 2026 Anirudh Yadav

You are free to use, copy, adapt, remix, and build on anything in this repository — for personal projects, your own whitepapers, your team's workflows, or anything else you can think of. No permission needed. No strings attached.
If something here helps you, great. If you improve on it, even better — sharing back is appreciated but never required.

The Part You Should Actually Read
This repository — the playbook, the code, the scripts, the examples, the diagrams, the PDF, the Word documents — is provided as-is, based on what worked for one person in one context. If you run a script and something breaks, that's on you to debug. 
In plain terms: use good judgment, test before you publish, verify before you cite, and don't hold this repo responsible for outcomes in your project.
I will be happy to hear feedback, fix genuine errors, and improve the material — but accepts no liability for how it is used.

Happy Coding.

---

Feedback and corrections are welcome
