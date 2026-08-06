# TMLR Review — Requested Changes Checklist

Source: TMLR review of uTECH-GenUrban. Verdict on evidence: claims not yet sufficiently
supported — the paper reads as a system demonstration, not a controlled empirical study.
Core ask: isolate what the LLM agents and verification pipeline actually contribute, via
baselines and ablations, run on the existing 4-city / multi-model setup in
`MulitiCityStudy.ipynb`.

## Critical for acceptance

- [ ] **Non-LLM baselines for OD-flow prediction.**
  Add historical-average and gravity/radiation-model baselines, plus "GBR without
  LLM-generated features" (spatial + building + temporal features only, dropping the
  `MobilityPlanner` force outputs). Compare against the full pipeline on the same
  train/test splits already defined per city in `cities_db` (`MulitiCityStudy.ipynb`,
  Cell 7). Report R², RMSE, MAE, CPC for each baseline alongside the current results.

- [ ] **Component ablations on the generation agents.**
  Compare single-LLM generation (one prompt produces population+activity+mobility) against
  the current multi-agent decomposition (`PopulationGenerator` → `ActivityGenerator` →
  `MobilityPlanner`, `build_agents()` in Cell 3). Ablate/replace each of the three agents
  individually (e.g. population clusters from a naive ACS proportional split instead of the
  LLM) and measure downstream OD-flow accuracy, not just qualitative plausibility.

- [ ] **Quantitative verification ablation.**
  The hybrid verification pipeline (`hard_check_population` deterministic checks →
  `pop_self_verifier`/`act_self_verifier` self-repair → Advocate/Skeptic/Statistician/Referee
  debate, Cells 1–3) is currently evaluated only qualitatively. Add a 2×2-ish comparison:
  no verification vs. deterministic-checks-only vs. self-repair vs. full debate pipeline,
  measured on (a) rate of hard-constraint violations caught/fixed and (b) final OD-flow
  accuracy. This is the most novel claim in the paper and currently has zero quantitative
  support.

- [ ] **Clarify train/test protocol — no leakage from flow-derived features.**
  Confirm and document explicitly whether the MDS zone-topology embeddings
  (`HardDataEngine`, Cell 5) and any flow-derived spatial features are fit using training-fold
  flow data only, not full-dataset (train+test) flows. Given `cities_db` splits by
  time-of-day windows within the same day (`Flow 6-9` ... `Flow 18-21` train, `Flow 22-25`
  test) rather than by city, this needs a precise statement of what's held out and confirmation
  MDS/scalers are `.fit()` on train only and `.transform()` on test.

- [ ] **Narrow generalization claims to within-city, within-day temporal prediction.**
  Given the same-city train/test split structure above, rewrite any "generalization" language
  in the paper to state within-city temporal generalization (predicting a later time window
  from earlier ones in the same city), not cross-city transfer — unless a genuine cross-city
  holdout experiment (train on 3 cities, test OD flows on the 4th) is added.

## Would strengthen the paper

- [ ] **Move system description/prompts/extra figures to appendix**, keep main text focused on
  the empirical comparison (baselines, ablations, verification results above).

- [ ] **Report run-to-run variability.** The multi-model loop already iterates over
  `config_list` (`openai.gpt-5-mini`, `meta.llama-4-maverick-17b-instruct`,
  `google.gemini-2.5-flash`, `xai.grok-3-mini`) per city — repeat each city×model
  combination ≥3 times (temperature/seed variation) and report mean ± std for R²/RMSE/CPC
  instead of single-run numbers.

- [ ] **Report prompting details and compute cost.** Log/report LLM call counts, tokens, and
  wall-clock time per city run (the loop already sets `llm_config["timeout"] = 300`, so per-call
  latency is measurable); include full system prompts for each agent in the appendix.

- [ ] **Stronger quantitative evaluation of generated population cohorts**, beyond the existing
  ACS-total tolerance check in `hard_check_population` — e.g. distributional comparison
  (KL/Wasserstein) of generated cluster demographics vs. ACS marginals, not just aggregate
  population count.

## Notes

- "Broader Impact Concerns: N/A" — no action needed.
- Reviewer's core framing: novelty isn't the issue — the manuscript needs restructuring around
  controlled comparisons (baselines + ablations) rather than end-to-end demonstration.
