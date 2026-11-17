# CortexGraph RAG Evaluation Scorecard

**Run ID:** `cortexgraph-eval-1794844973`  
**Timestamp:** `2026-11-16 16:02:53 UTC`  
**Total Benchmark Queries:** `35`  
**Evaluation Status:** `PASSED ALL TARGETS`  

## Summary Metrics vs Acceptance Targets

| Metric | Target | Actual Score | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Context Precision** | `>= 0.85` | **`0.9286`** | **PASS** | Proportion of expected ground truth note paths present in top-k retrieval |
| **Faithfulness** | `>= 0.90` | **`1.0000`** | **PASS** | Factual grounding score verifying claims derive strictly from verified context |
| **Answer Relevance** | `>= 0.85` | **`1.0000`** | **PASS** | Completeness and directness in answering the user question |
| **Abstention Accuracy** | `= 1.00` | **`1.0000`** | **PASS** | Accuracy on out-of-domain queries acknowledging absence of vault data |

## Performance by Query Category

| Category | Queries | Context Precision | Faithfulness | Answer Relevance | Abstention Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `single_note_lookup` | 10 | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `multi_note_synthesis` | 10 | `0.7500` | `1.0000` | `1.0000` | `1.0000` |
| `temporal_or_tag_filtered` | 8 | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `out_of_domain_abstention` | 7 | `1.0000` | `1.0000` | `1.0000` | `1.0000` |

## Detailed Test Case Results

| ID | Category | Query | Precision | Faithfulness | Relevance | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `eval-01` | `single_note_lookup` | What is the maximum retry limit enforced in ADR-00... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-02` | `single_note_lookup` | What are the two stages of document chunking defin... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-03` | `single_note_lookup` | Why did CortexGraph choose OpenAI Structured Outpu... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-04` | `single_note_lookup` | What caused the context window overflow incident i... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-05` | `single_note_lookup` | What circuit breaker and guardrail were introduced... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-06` | `single_note_lookup` | How does CortexGraph persist conversation state ac... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-07` | `single_note_lookup` | How does the rolling summarization algorithm manag... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-08` | `single_note_lookup` | What distance metric and collection name are confi... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-09` | `single_note_lookup` | How does CortexGraph achieve zero-redundancy incre... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-10` | `single_note_lookup` | What dimension and pricing tradeoffs were evaluate... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-11` | `multi_note_synthesis` | How do ADR-001 and cyclic recursion limits togethe... | `0.67` | `1.00` | `1.00` | **FAIL** |
| `eval-12` | `multi_note_synthesis` | How does the two-stage chunking strategy in ADR-00... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-13` | `multi_note_synthesis` | How does the conversation memory manager mitigate ... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-14` | `multi_note_synthesis` | What are the differences between Self-RAG reflecti... | `0.50` | `1.00` | `1.00` | **FAIL** |
| `eval-15` | `multi_note_synthesis` | How do the postmortems on context overflow and hal... | `0.67` | `1.00` | `1.00` | **FAIL** |
| `eval-16` | `multi_note_synthesis` | How does FastAPI SSE streaming coordinate with Lan... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-17` | `multi_note_synthesis` | How do Tenacity retry policies protect against Ope... | `0.50` | `1.00` | `1.00` | **FAIL** |
| `eval-18` | `multi_note_synthesis` | How does WikiLink normalization enable hybrid know... | `0.50` | `1.00` | `1.00` | **FAIL** |
| `eval-19` | `multi_note_synthesis` | How do RAG Triad evaluation metrics map onto LangG... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-20` | `multi_note_synthesis` | How does Transformer Self-Attention complexity inf... | `0.67` | `1.00` | `1.00` | **FAIL** |
| `eval-21` | `temporal_or_tag_filtered` | List the architectural decision records tagged wit... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-22` | `temporal_or_tag_filtered` | Find all debugging postmortems tagged with 'postmo... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-23` | `temporal_or_tag_filtered` | Retrieve reading notes tagged with 'reading-notes'... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-24` | `temporal_or_tag_filtered` | Find notes tagged with 'langfuse' regarding observ... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-25` | `temporal_or_tag_filtered` | Retrieve ADR notes with tag 'adr' discussing struc... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-26` | `temporal_or_tag_filtered` | Find notes tagged with 'fastapi' or 'sse' detailin... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-27` | `temporal_or_tag_filtered` | Retrieve notes tagged with 'resiliency' discussing... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-28` | `temporal_or_tag_filtered` | Find notes tagged with 'evals' regarding RAG triad... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-29` | `out_of_domain_abstention` | What is the capital of Australia and what is its c... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-30` | `out_of_domain_abstention` | How do you configure an Apache Cassandra multi-dat... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-31` | `out_of_domain_abstention` | What is the traditional recipe and hydration ratio... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-32` | `out_of_domain_abstention` | Who won the FIFA World Cup final in 1994 and what ... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-33` | `out_of_domain_abstention` | How do quantum key distribution protocols like BB8... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-34` | `out_of_domain_abstention` | What are the tax implications, like-kind criteria,... | `1.00` | `1.00` | `1.00` | **PASS** |
| `eval-35` | `out_of_domain_abstention` | What are the rules, board levels, and attack vecto... | `1.00` | `1.00` | `1.00` | **PASS** |

---
*Automated evaluation report generated by CortexGraph evaluation harness (`evals.run_evals`).*
