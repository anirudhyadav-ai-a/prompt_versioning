# Working Plan — Prompt Versioning & Drift Detection

> **Companion implementation guide for the whitepaper:**
> *"Prompt Versioning & Drift Detection"*

---

## Table of Contents

1. [Introduction](#1-introduction--why-prompts-are-code)
2. [Prompt Registry](#2-prompt-registry)
3. [Drift Detection](#3-drift-detection)
4. [A/B Testing](#4-ab-testing)
5. [Decision Framework](#5-decision-framework)
6. [Cross-Model Prompt Management](#6-cross-model-prompt-management)
7. [CI/CD Integration](#7-cicd-integration)
8. [Conclusion](#8-conclusion)
9. [Appendix](#9-appendix)

---

## 1. Introduction — Why Prompts Are Code

> **Who this is for:** ML engineers, platform teams, and engineering leaders running LLM-powered features in production — especially teams managing prompts across multiple models or environments.

Prompts are, in the author's experience, the most fragile component in an LLM system. A single word change can alter behavior dramatically. Yet most teams manage prompts as hardcoded strings — no versioning, no diffing, no rollback. When a model provider silently updates their model, prompts that worked yesterday break today with no way to diagnose what changed.

Several tools address pieces of this problem. Microsoft's PromptFlow [1] provides prompt management and evaluation workflows. Promptfoo [2] offers open-source prompt testing and comparison. LangChain's LangSmith [3] adds tracing, evaluation, and basic versioning. However, as of May 2026, none of these tools combine all four capabilities — versioning with content-hash dedup, embedding-based drift detection, statistically rigorous A/B testing, and cross-model prompt management — in a single, lightweight framework that integrates directly into existing CI/CD pipelines and VS Code workflows.

**Scope boundaries:** This paper covers prompt lifecycle management for text-based LLM systems. It does not cover image/multimodal prompts, fine-tuning workflows, prompt injection defense, or RAG-specific prompt patterns (covered in the companion RAG paper in this series).

*Uniqueness note: The author searched Google Scholar, arXiv, Medium, LinkedIn Pulse, and GitHub for "prompt versioning", "prompt drift detection", and "LLM prompt lifecycle management" in May 2026. The closest existing works are PromptFlow, Promptfoo, and LangSmith (referenced in Section 9). This paper's contribution is the combination of all four capabilities in a single framework with CI/CD integration and a VS Code extension.*

```
Prompt v1 ──► Model A ──► Output A (good)
Prompt v1 ──► Model B ──► Output B (degraded)  ← "drift"
Prompt v2 ──► Model B ──► Output C (good)      ← "fix via new version"
```

### What Goes Wrong Without Prompt Versioning

| Problem | Symptom | Root Cause |
|---|---|---|
| Silent regression | Quality drops after model update | No baseline to compare against |
| No rollback | Bad prompt deployed, can't revert | Prompts not versioned |
| No attribution | "Who changed the prompt and when?" | No audit trail |
| No testing | Prompt changes deployed without validation | No regression suite |

**Key insight:** "Prompts deserve the same lifecycle as source code — versioning, review, testing, and rollback."

[1] Microsoft PromptFlow — prompt management and evaluation framework
[2] Promptfoo — open-source prompt testing and evaluation (promptfoo.dev)
[3] LangSmith — LangChain's prompt tracing, evaluation, and versioning platform

---

## 2. Prompt Registry

### Architecture: Version, Diff, and Rollback

```
Developer ──► register(name, content, description) ──► Registry
                                                         │
                                                    ┌────┴────┐
                                                    │ v1      │
                                                    │ v2      │
                                                    │ v3 ←    │ (current)
                                                    └─────────┘
```

### Content Hashing (SHA-256) for Dedup

Every prompt version stores a SHA-256 hash of its content. When a developer calls `register()`, the registry checks whether the latest version already has the same hash. If so, it skips creating a new version — no duplicate entries, no wasted storage.

### Semantic Diff Between Versions

String diffs are noisy for prompts. Instead, embed both prompt versions — convert the text into a numerical vector using an embedding model — and compute cosine similarity (a measure of the angle between two vectors, where 1.0 means identical and 0.0 means completely unrelated). If the similarity is below 0.95, flag the change as semantically significant. The 0.95 threshold was chosen empirically: in testing, prompt rewrites that preserved intent typically scored above 0.95, while substantive changes scored below.

### Rollback

Rolling back creates a **new version** with the content of the target version. This preserves the full history — you can always see that a rollback happened and when.

### Storage

In-memory for examples, but production would use a database (Postgres, DynamoDB, etc.) with an immutable append-only table.

### Model Tracking

The `PromptVersion` model includes a `models_tested` field that records which models a prompt version has been validated against. The `mark_tested()` method on `PromptRegistry` updates this field:

```python
registry.mark_tested("classify-intent", version=3, model_label="claude")
```

This ensures every prompt version carries a record of which models it is known to work with — critical for multi-model deployments.

---

## 3. Drift Detection

### Architecture: Test Suite + Embedding Similarity

```
Test Suite: [(input_1, expected_1), (input_2, expected_2), ...]
                │
                ▼
Prompt v3 + Model ──► actual_output
                          │
                          ▼
              embed(expected) vs embed(actual) ──► similarity score
                                                       │
                                                  ≥ threshold ──► PASS
                                                  < threshold ──► DRIFT DETECTED
```

### Drift Detection Strategies

| Strategy | How it works | Pros | Cons |
|---|---|---|---|
| Exact match | Compare strings | Simple, deterministic | Too strict for LLMs |
| Embedding similarity | Cosine similarity of output embeddings | Captures semantic equivalence | Requires embedding model |
| LLM-as-judge | Ask an LLM if outputs are equivalent | Most flexible | Expensive, non-deterministic |

### When Drift Happens

Drift is typically caused by:
1. **Model updates** — the provider ships a new version of the model
2. **Infrastructure changes** — different hardware, quantization, or routing
3. **API behavior changes** — default parameters change silently

### Per-Model Drift Profiles

When running multiple models, drift should be tracked independently per model. The `CrossModelDriftDetector` runs the same test suite against each model and produces per-model drift reports:

```python
cross_detector = CrossModelDriftDetector(registry, multi_client, threshold=0.85)
results = await cross_detector.run_cross_model_suite("classify-intent", test_cases)
summary = cross_detector.cross_model_summary(results)
# summary["most_stable_model"], summary["most_drifted_model"]
```

This isolates which model regressed after a provider update, rather than treating all models as a single black box.

---

## 4. A/B Testing

### Architecture: Route, Score, Compare

```
Query ──► Router (50/50 split) ──► Prompt v3 ──► Output A ──► Score
                                  ──► Prompt v4 ──► Output B ──► Score
                                                                  │
                                                          Compare scores
                                                          ──► Winner
```

### A/B Test Metrics

| Metric | How to measure | What it tells you |
|---|---|---|
| Quality score | LLM-as-judge rates output 1-5 | Which prompt produces better answers |
| Latency | Wall-clock time per request | Whether the new prompt is slower |
| Token usage | Count input + output tokens | Cost impact of the new prompt |
| Consistency | Variance of quality scores | Whether the prompt is reliable |

### Scoring with LLM-as-Judge

Use a separate LLM call to rate each output on a 1–5 scale. The scoring prompt should be deterministic and focused: "Rate the following response for quality, relevance, and completeness."

### Statistical Significance with Welch's t-test

After collecting quality scores from both prompt versions, the framework applies Welch's t-test — a statistical test that compares two groups of measurements when they may have different variances and sample sizes. It returns a p-value: the probability that the observed difference occurred by chance. If p < 0.05, the difference is statistically significant and the winning prompt can be deployed with confidence. The implementation includes minimum sample size estimation using Cohen's d (effect size) to guide how many queries are needed before results are meaningful.

### Cross-Model Comparison

Beyond comparing prompt versions, A/B testing can compare the same prompt across different models. The `CrossModelABTester` runs each query against every model and uses LLM-as-judge scoring:

```python
tester = CrossModelABTester(registry, "summarize", version=2, models=multi_client)
comparison = await tester.run_model_comparison(queries)
# comparison["winner"], comparison["per_model"]
```

This answers the question: "For this specific prompt, which model performs best?"

---

## 5. Decision Framework

> Updated to include multi-model decision branches.

### When to Use Each Pattern

| Situation | Pattern | Why |
|---|---|---|
| Any production system | Prompt Registry | Baseline — always version your prompts |
| Model provider announces update | Drift Detection | Run test suite before and after to catch regressions |
| Rewriting a prompt | A/B Testing | Validate the new version produces better results |
| Debugging quality issues | Drift Detection + Registry | Check if prompt changed or model drifted |

### Decision Tree

```
            ┌──────────────────┐
            │ Are you changing  │
            │ the prompt?       │
            └────────┬─────────┘
                     │
            ┌────────┴────────┐
            │                 │
           Yes               No
            │                 │
            ▼                 ▼
     ┌────────────┐   ┌────────────┐
     │ A/B test   │   │ Is the     │
     │ new vs old │   │ model      │
     └────────────┘   │ changing?  │
                      └──────┬─────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                   Yes               No
                    │                 │
                    ▼                 ▼
             ┌────────────┐   ┌────────────┐
             │ Run drift  │   │ No action  │
             │ detection  │   │ needed     │
             └──────┬─────┘   └────────────┘
                    │
                    ▼
             ┌────────────────┐
             │ Multiple       │
             │ models?        │
             └───────┬────────┘
                     │
            ┌────────┴────────┐
            │                 │
           Yes               No
            │                 │
            ▼                 ▼
     ┌────────────────┐  ┌────────────┐
     │ Cross-model    │  │ Single     │
     │ drift suite    │  │ model run  │
     └────────────────┘  └────────────┘
```

### Combining Patterns

In practice, you use all three together:

```
1. Store all prompts in the Registry (always)
2. Run Drift Detection on a schedule (daily/weekly)
3. When rewriting a prompt, A/B test before deploying
4. If drift is detected, check Registry history to diagnose
5. Roll back to a known-good version if needed
```

---

## 6. Cross-Model Prompt Management

### Architecture: Per-Model Drift Detection

```
                    ┌─────────────────┐
                    │  Prompt Registry │
                    │  v3: "classify"  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Claude   │  │  GPT-4o  │  │  Gemini  │
        │  drift:2% │  │  drift:0%│  │  drift:5%│
        └──────────┘  └──────────┘  └──────────┘
```

### Cross-Model A/B Testing Workflow

1. Register the prompt in the registry
2. Configure `MultiModelClient` with all target models
3. Run `CrossModelABTester.run_model_comparison()` with representative queries
4. Review per-model quality, latency, and token usage
5. Select the best model for the task

### Model-Specific Prompt Variant Storage

Some prompts need model-specific tuning. The registry supports storing variants using naming conventions:

```
classify-intent/claude    → optimized for Claude
classify-intent/gpt-4o    → optimized for GPT-4o
classify-intent/gemini    → optimized for Gemini
```

### Code Reference

- `prompt_versioning/utils/llm_client.py` — `MultiModelClient` class
- `prompt_versioning/examples/drift_detector/detector.py` — `CrossModelDriftDetector`
- `prompt_versioning/examples/ab_tester/tester.py` — `CrossModelABTester`

This pattern is inspired by the per-role model preference approach used in the companion paper *"CopilotAs JAM"* (in this repository), which assigns preferred models per role (Judge → Claude, Advocate-ADR → GPT-4o, etc.).

---

## 7. CI/CD Integration

### Running Drift Detection as a GitHub Action

Prompt drift should be monitored continuously, not just when someone remembers to check. A GitHub Action can run drift detection on a schedule and alert the team when drift exceeds a threshold.

```yaml
name: Prompt Drift Detection
on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6am UTC
  workflow_dispatch:
    inputs:
      prompt_name:
        description: 'Prompt name to test (or "all")'
        default: 'all'

jobs:
  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e ".[all]"
      - name: Run drift detection
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python -m prompt_versioning.examples.drift_detector.ci_runner --test-cases prompt_versioning/examples/drift_detector/test_cases.json --output drift-report.json
      - name: Upload drift report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: drift-report
          path: drift-report.json
```

### Pre-Deploy Drift Check

Before deploying a new model version, run the drift detection suite against the candidate model. If drift rate exceeds the configured threshold (default 10%), the CI pipeline fails and blocks deployment.

The `ci_runner.py` script handles this workflow:
1. Loads test cases from `test_cases.json`
2. Runs drift detection against the live API
3. Outputs results as JSON to `drift-report.json`
4. Exits with code 1 if drift rate exceeds threshold

### Alerting on Drift

When drift is detected above threshold, the GitHub Action can be extended to notify the team:
- **Slack**: Use the `slackapi/slack-github-action` to post drift alerts to a channel
- **Email**: Configure GitHub Actions notifications for workflow failures
- **PagerDuty**: For critical prompts, trigger an incident when drift rate is high

### Decision Table: When to Run Drift Checks

| Trigger | Type | Use Case |
|---|---|---|
| Weekly schedule | Scheduled | Catch silent model provider updates |
| Model version change | Event-driven | Pre-deploy validation |
| Prompt version change | Event-driven | Validate new prompt against baseline |
| Manual dispatch | On-demand | Ad-hoc investigation |
| After incident | On-demand | Root cause analysis |

---

## 8. Conclusion

### Contributions

This paper presents a practical framework for treating prompts as versioned, testable, deployable artifacts. The key contributions are:

1. **Prompt Registry** with content-hash deduplication, semantic diff via embedding similarity, and full rollback history
2. **Drift Detection** using embedding-based similarity scoring to catch silent regressions after model provider updates — including per-model drift profiles for multi-model deployments
3. **A/B Testing** with LLM-as-judge scoring and Welch's t-test for statistical significance, enabling data-driven prompt iteration
4. **Cross-Model Management** — a unified interface for running drift detection and A/B tests across Claude, GPT-4o, Gemini, and other providers
5. **CI/CD Integration** — automated drift detection as a GitHub Action with configurable thresholds and alerting
6. **VS Code Extension** — native Copilot Chat integration via the `@prompts` participant for prompt registration, drift checking, and testing without leaving the editor

### Limitations

- **Embedding-based drift detection** depends on the quality of the embedding model. If the embedding model itself is updated, drift scores may shift without any actual change in LLM behavior. Teams should pin their embedding model version.
- **LLM-as-judge scoring** in A/B testing introduces non-determinism. While temperature=0 and structured prompts reduce variance, the judge's own biases (verbosity preference, position bias) can affect results. The framework does not currently implement judge calibration or multi-judge consensus.
- **In-memory storage** — the reference implementation uses an in-memory dictionary for the prompt registry. Production deployments would need a persistent store (Postgres, DynamoDB, etc.), which is documented but not implemented.
- **Cost** — running drift detection across multiple models with embedding similarity is not free. The framework does not currently estimate or cap API costs per run.
- **Single-turn prompts only** — the current implementation handles single-turn prompt versioning. Multi-turn conversation prompt management (system + user + assistant chains) is not covered.

### Future Work

- **Persistent storage backends** — Postgres and DynamoDB adapters for the prompt registry
- **Multi-turn prompt versioning** — extending the registry to handle conversation-level prompt chains
- **Judge calibration** — implementing multi-judge consensus and bias correction for A/B test scoring
- **Cost estimation** — per-run cost projections based on token counts and model pricing
- **Prompt optimization** — automated prompt rewriting guided by A/B test results (DSPy-style compilation)
- **Dashboard** — a lightweight web UI for browsing prompt versions, drift reports, and A/B test results

---

## 9. Appendix

### Glossary

| Term | Definition |
|---|---|
| Prompt Version | A specific revision of a prompt, identified by version number and content hash |
| Content Hash | SHA-256 digest of the prompt text, used to detect duplicate content |
| Semantic Diff | Comparison of two prompts using embedding similarity rather than string diff |
| Drift | Change in LLM output quality/behavior without any prompt change, typically caused by model updates |
| Baseline | A set of known-good (input, output) pairs used as reference for drift detection |
| A/B Test | Experiment comparing two prompt versions by routing traffic and measuring quality |

### References

**Prompt Management & Evaluation**
- "Prompt Engineering Guide" — DAIR.AI (2024)
- "PromptFlow" — Microsoft (prompt management and evaluation framework)
- "Promptfoo" — open-source prompt testing and evaluation framework
- "LangSmith" — LangChain (prompt tracing, evaluation, and versioning)

**LLM Evaluation Frameworks**
- Liang et al., "Holistic Evaluation of Language Models (HELM)" — Stanford CRFM (2023)
- Gao et al., "A Framework for Few-Shot Language Model Evaluation (lm-eval-harness)" — EleutherAI (2023)
- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2024)

**Prompt Sensitivity & Drift**
- Sclar et al., "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design" — ICLR 2024
- Lu et al., "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity" — ACL 2022
- Mizrahi et al., "State of What Art? A Call for Multi-Prompt LLM Evaluation" — TACL 2024

**Versioning & Operations**
- "Semantic Versioning 2.0.0" — semver.org
- Shankar et al., "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences" — 2023
- Ribeiro et al., "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList" — ACL 2020

**Industry Practices**
- Anthropic, "Effective context engineering for AI agents" — Anthropic Engineering Blog (2025) — covers prompt structuring patterns relevant to versioning
- OpenAI Cookbook — cookbook.openai.com — community-contributed examples of prompt evaluation, caching, and testing workflows

Feedback and corrections are welcome — [open an issue](https://github.com/anirudhyadav/whitepaper/issues) or connect on [LinkedIn](https://linkedin.com/in/anirudhyadav).

---

## VS Code Extension & Prompt Templates

### Extension Setup

```bash
cd vscodebase && npm install && npm run compile
# Press F5 in VS Code to launch the extension development host
```

### Chat Participant: `@prompts`

| Command | What It Does |
|---------|-------------|
| `/register` | Scans workspace, loads prompt template, calls Copilot LLM with injected context |
| `/drift` | Scans workspace, loads prompt template, calls Copilot LLM with injected context |
| `/test` | Scans workspace, loads prompt template, calls Copilot LLM with injected context |

### How It Works

1. User types `@prompts /<command>` in Copilot Chat
2. Extension calls `vscode.workspace.findFiles()` to discover source files
3. Extracts symbols (functions, classes, imports) from each file
4. Loads the matching prompt template from `prompts/`
5. Replaces `{{workspace_context}}` with the extracted code context
6. Sends the enriched prompt to Copilot via `vscode.lm.selectChatModels()`
7. Streams the response back to the chat panel

### Prompt Templates

Located in `prompts/`:
- `prompt-registry.md`
- `drift-detection.md`
- `ab-testing.md`

Each template uses these placeholders:
- `{{workspace_context}}` — injected codebase context from the scanner
- `{{user_query}}` — user's question or request
- `{{code}}` — code snippet to analyse
