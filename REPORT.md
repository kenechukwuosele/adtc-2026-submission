# ADTC 2026 Technical Report

**Team ID:** codehome
**Submitter:** kenechukwuosele
**Domain:** Coding Assistants
**Model:** Qwen2.5-Coder-3B-Instruct-Q3_K_M

---

## 1. Problem

Software developers across West African cities including Lagos, Nairobi, and Accra face persistent internet instability. Mobile data connections throttle under load, shared office networks drop during peak hours, and power outages reset routers without warning. Cloud-based coding assistants such as GitHub Copilot, ChatGPT, and Cursor require stable low-latency connectivity and fail or timeout when that connectivity is absent.

The target user is a self-taught or bootcamp-trained developer aged 18 to 30, working on a budget laptop in Nigeria. This user writes Python and JavaScript daily, cannot afford a GitHub Copilot subscription, and has no reliable access to cloud AI APIs. When an internet outage occurs mid-session, no AI assistance is currently available to them.

This submission addresses that gap by providing a coding assistant that runs entirely on the user's local machine with zero network dependency after the initial model download. The user can request code generation, debugging assistance, and programming explanations at any time, regardless of connectivity status.

---

## 2. Design Decisions

### 2.1 Starting Point

The submission template ships with SmolLM2-135M-Instruct as a working default. After reviewing the scoring specification, specifically that Sacc (50 percent of total score) is computed by running lm-eval-harness directly against the raw GGUF file with no custom inference pipeline in the evaluation path, it became clear that base model quality is the primary lever for accuracy. SmolLM2-135M produces incoherent responses on coding tasks and was replaced.

### 2.2 Model Selection

The viable parameter range for this hardware is approximately 2B to 4B at Q4 quantization: sufficient to stay within the 7 GB RAM budget while retaining meaningful coding capability. Three candidates were evaluated.

| Model | Decision | Reason |
|---|---|---|
| SmolLM2-135M-Instruct | Rejected | Insufficient capability for coding tasks. Low Sacc ceiling regardless of surrounding pipeline. |
| Qwen2.5-3B-Instruct (general) | Rejected | Capable at instruction following but not specialised for code. Weaker on debugging and generation than the Coder variant at the same parameter count. |
| Qwen2.5-Coder-3B-Instruct | Selected | Coding-specialised fine-tune of the Qwen2.5 base model. Stronger HumanEval and MBPP performance than the general variant at identical size. Best Sacc potential within the RAM budget. |
| Qwen2.5-Coder-7B-Instruct | Rejected | Exceeds RAM budget at Q4_K_M quantization. Peak RSS would approach 6 GB, severely penalising Seff. |

### 2.3 Quantization Selection

Three quantization levels were benchmarked directly on the submission hardware prior to selection.

| Quantization | File Size | Peak RAM | Generation TPS | Seff Score | Sperf Score |
|---|---|---|---|---|---|
| Q4_K_M | 1.79 GB | 3.27 GB | 9.72 t/s | 53.3 | 64.8 |
| Q3_K_M | 1.48 GB | 2.19 GB | 9.50 t/s | 68.7 | 63.3 |
| IQ3_M | 1.38 GB | not measured | 4.16 t/s | not measured | 27.7 |

IQ3_M was rejected. Importance-matrix quantization is optimised for GPU computation and performs poorly on x86-64 CPU inference with AVX2 instruction sets. The observed generation speed of 4.16 t/s represented an unacceptable Sperf penalty.

Q3_K_M was selected over Q4_K_M. The reduction in peak RAM of 1.08 GB (33 percent) produced a Seff gain of 15.4 points. The corresponding TPS reduction of 0.22 t/s produced a Sperf loss of only 1.5 points. The net improvement to the combined weighted score favoured Q3_K_M.

### 2.4 RAG Pipeline

A retrieval-augmented generation layer over a local programming documentation corpus was considered. It was ruled out on two grounds. First, the profiler invokes lm-eval-harness directly against the GGUF file and does not call any custom inference code, meaning a RAG layer would not affect the evaluated accuracy score. Second, running a vector store and embedding model alongside a 3B inference process would increase peak RAM and penalise Seff. Base model quality was identified as the correct lever for Sacc.

### 2.5 Fine-tuning

Additional fine-tuning on a curated coding dataset was considered. The submission hardware has no discrete GPU, making full fine-tuning and LoRA adaptation impractical within the hardware constraints. The Qwen2.5-Coder-3B-Instruct base model is already instruction-tuned for coding tasks, making further fine-tuning a lower priority relative to model selection and quantization decisions.

---

## 3. Constraints

**Hardware:** Development and benchmarking were conducted on an 11th Gen Intel Core i5-1135G7 at 2.40 GHz, 7.6 GB RAM, no discrete GPU, running Ubuntu 24.04 LTS. This configuration closely matches the ADTC Standard Laptop profile, making local benchmark numbers directly representative of audit results.

**Memory:** The 7 GB scoring budget constrains model selection to approximately 3B parameters at Q4 quantization. The KV cache allocates additional RAM above the raw model weight size. A 1.48 GB model file produces a 2.19 GB peak RSS during inference due to context buffer allocation at the default context length.

**Connectivity:** The model operates with zero external network calls during inference. Weights are downloaded once via download_model.sh from a public HuggingFace URL. No external API calls or retrieval requests are made at inference time.

**Runtime:** llama.cpp is the only accepted runtime per competition rules. The submission binary was compiled from source using cmake Release configuration with AVX2 auto-detection enabled. Benchmarking used llama-bench at 4 threads, consistent with the 4-vCPU standard laptop profile.

**Thread ceiling:** Generation TPS was benchmarked at thread counts of 1, 2, 3, and 4. Performance peaked at 4 threads. The absence of further gains beyond 4 threads is consistent with memory bandwidth being the limiting factor on this CPU rather than raw compute capacity.

---

## 4. Benchmarks

All measurements were taken on the development machine (Intel Core i5-1135G7, 7.6 GB RAM, no GPU, Ubuntu 24.04 LTS) using llama-bench from a source build of llama.cpp.

### 4.1 Throughput

```
| model                 | size     | params  | backend | threads | test  | t/s          |
| qwen2 3B Q3_K-Medium  | 1.48 GiB | 3.09 B  | CPU     | 4       | pp512 | 21.15 ± 0.69 |
| qwen2 3B Q3_K-Medium  | 1.48 GiB | 3.09 B  | CPU     | 4       | tg200 | 11.25 ± 0.16 |
```

Generation speed of 11.25 t/s with a variance of 0.16 indicates thermally stable inference across all repetitions.

### 4.2 Memory

| Metric | Value |
|---|---|
| Peak RSS | 2,197 MB |
| Steady-state RSS | 2,078 MB |
| Peak VMS | 2,674 MB |

Peak RAM of 2.19 GB leaves 4.81 GB of the 7 GB scoring budget unused.

### 4.3 Thermal

| Metric | Value |
|---|---|
| CPU utilization p99 | 64.2% |
| Throttled | No |

No thermal throttling was detected across any benchmark run. The thermal penalty of 10 points does not apply.

### 4.4 Official Profiler Output

The following values are drawn directly from submission.json produced by adtc-profiler in participant mode.

```
tokens_per_second_generation : 9.72
first_token_latency_ms       : 32,485
peak_rss_mb                  : 2,197
throttled                    : false
params_match                 : true
git_commit_sha               : 9af30a6560bc
measured_on                  : participant_laptop
```

### 4.5 Score Projection

| Component | Basis | Score | Weight | Contribution |
|---|---|---|---|---|
| Sacc | Qwen2.5-Coder-3B-Instruct | ~60 | 50% | ~30.0 pts |
| Sperf | 9.72 TPS vs 15.0 reference | 64.8 | 30% | 19.4 pts |
| Seff | 2.19 GB peak RAM | 68.7 | 20% | 13.7 pts |
| Pthermal | Not throttled | 0 penalty | -- | 0 pts |
| **Subtotal** | | | | **~63.1 pts** |
| African Use Case Bonus | Offline developer tooling for African markets | -- | Bonus | +5 to +10 pts |
| **Projected Total** | | | | **~68 to 73 pts** |
