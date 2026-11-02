# VagueBench

<h3 align="center">
  Evaluating Clarification Behavior in Large Language Models
</h3>

<p align="center">
  <img src="https://img.shields.io/badge/EMNLP-2026%20Oral-8A2BE2?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://zhangrongyu0101.github.io/VagueBench/paper.pdf">📄 Paper</a>
  &nbsp; · &nbsp;
  <a href="https://zhangrongyu0101.github.io/VagueBench/">🌐 Project Page</a>
  &nbsp; · &nbsp;
  <a href="https://huggingface.co/datasets/rongyu0101/VagueBench/tree/main">🤗 Hugging Face</a>
</p>

---

## Overview


VagueBench is a benchmark designed to evaluate the clarification capabilities of conversational assistants. It systematically examines how models handle ambiguous and underspecified user requests, focusing on three key dimensions:

- **Clarification Behavior** — Recognizing when clarification is necessary and asking appropriate questions.
- **Dialogue Efficiency** — Gathering missing information with minimal unnecessary interaction.
- **Intent & Detail Coverage** — Recovering the user's underlying intents and task-specific details.

VagueBench contains **750 records across 250 scenarios**, with three controlled variants per scenario to investigate assistant behavior under different levels of information availability.

## Benchmark


### Three Levels of Information Availability

| Group | Scenarios | Records | Information Withheld |
|:---:|:---:|:---:|:---|
| **A** | 250 | 250 | None |
| **B** | 250 | 250 | User intent(s) |
| **C** | 250 | 250 | User intent(s) + selected details |
| **Total** | **250** | **750** | |

The three variants of each scenario enable controlled evaluation under progressively increasing ambiguity.

## Quick Start

### Installation

```bash
git clone https://github.com/zhangrongyu0101/VagueBench.git
cd VagueBench

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp config.example.json config.local.json
```

### Configuration

Configure the target and judge models in `config.local.json`, including their model names, OpenAI-compatible Chat Completions endpoints, and API key environment variables.

By default, the judge model also serves as the simulated user. You can optionally configure a separate `simulator` model.

### Run Evaluation

```bash
export TARGET_API_KEY="<your-target-api-key>"
export JUDGE_API_KEY="<your-judge-api-key>"

python evaluate.py \
    --config config.local.json \
    --output outputs/results.jsonl
```

**Optional arguments**

| Argument | Description | Default |
|:---|:---|:---|
| `--input PATH` | Input JSONL dataset | `data/VagueBench_en.jsonl` |
| `--limit N` | Evaluate the first N records | All |
| `--max-turns N` | Maximum simulated-user replies | `5` |

For each run, specify a new output path to preserve previous results.

## Dataset

### Repository Structure

```text
VagueBench/
├── data/
│   └── VagueBench_en.jsonl
├── evaluate.py
├── config.example.json
├── requirements.txt
└── README.md
```

### Data Fields

Each JSONL record contains:

| Field | Description |
|:---|:---|
| `sample_id` | Scenario and variant identifier |
| `group` | Variant group (A/B/C) |
| `query_type` | Task category |
| `initial_query` | Opening query presented to the assistant |
| `source_query` | Full original user request |
| `needs_clarification` | Gold first-turn clarification label |
| `intents` | Annotated user intents |
| `details` | Annotated task details |
| `hidden_intent_ids` | Intents withheld from the opening |
| `hidden_detail_ids` | Details withheld from the opening |
| `simulator_prompt` | Simulated-user policy |

Details reference their corresponding intents through `intent_ids`. A detail's `value` can be either a string or a list.

### Important Notes

- Variants belonging to the same scenario share a UUID with `_A`, `_B`, or `_C` suffixes.
- Variants may appear far apart in the JSONL file. **Keep scenario variants together when splitting the dataset.**
- Always use each record's own annotations; some variants retain independently translated wording.

## Model Configuration

Each model role supports:

| Parameter | Description |
|:---|:---|
| `model` | Model identifier |
| `base_url` | OpenAI-compatible API endpoint |
| `api_key_env` | Environment variable containing the API key |
| `temperature` | Sampling temperature (`null` if unsupported) |
| `timeout` | Request timeout |
| `max_retries` | Maximum retry attempts |
| `json_mode` | Whether to enable JSON response mode |

If the judge endpoint does not support `response_format`, set its `json_mode` to `false`. The judge must still return valid JSON.

