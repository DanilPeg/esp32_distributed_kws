# Research: Self-Improvement for a Long-Lived LLM Agent

Date: 2026-02-23

## Key Research Findings
- ReAct: interleaves reasoning traces and actions to improve interpretability and reduce error propagation in agentic tasks.
  https://arxiv.org/abs/2210.03629
- Reflexion: uses verbal reflection stored in episodic memory to improve future attempts without weight updates.
  https://arxiv.org/abs/2303.11366
- MemGPT: memory hierarchy + external memory to extend effective context and manage long-term state.
  https://arxiv.org/abs/2310.08560
- Tree of Thoughts: explores multiple reasoning paths and self-evaluates to choose better plans.
  https://arxiv.org/abs/2305.10601
- RAG: retrieval-augmented generation with non-parametric memory improves factuality and updatability.
  https://arxiv.org/abs/2005.11401

## Implications for Our Agent
1) Use a Plan -> Act -> Observe loop and separate "thought" from "action" logs.
2) Add short reflection after each task (what worked / what failed / how to adjust).
3) Memory tiers: BIBLE (constitution), Checkpoint (recent), Journal (details), Research (cold). Keep context small.
4) Retrieval step before big tasks: search Journal/Research for relevant keywords and include only needed excerpts.
5) For complex decisions, generate 2-3 candidate plans, evaluate quickly, pick one; record rejected options.

## Proposed Process Changes
- BIBLE update policy: only stable rules go into BIBLE; changes are small and versioned.
- Journal policy:
  - Progress entry per session (goal/decisions/actions/results).
  - Daily/weekly checkpoint summary.
  - Reflection snippet added after each task.
- Wrapper (Context Pack) design:
  - Input: task + constraints.
  - Load BIBLE + latest checkpoint + last 1-2 journal entries.
  - Retrieve relevant notes by keyword (rg) from Research/Journal.
  - Assemble a small context pack (bounded size) and run the task.
  - Log outcomes + reflection; propose BIBLE update if a rule repeats.

## Next Steps
- Add a short "Self-Improvement Protocol" section in BIBLE.md pointing here.
- Optional: implement code/scripts/context_pack.py to automate context assembly.