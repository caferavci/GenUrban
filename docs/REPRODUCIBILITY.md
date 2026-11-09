# Reproducibility and scope

## What is preserved

This release packages the original `MulitiCityStudy.ipynb` workflow with
environment-based configuration, a command-line runner, and limited runtime
repairs. The runner executes that notebook. It does not substitute a separate
generation or prediction implementation.

The pipeline retains the original sequence:

1. Generate demographic cohorts from ACS county marginals.
2. Generate five-domain hourly activity shares for each cohort.
3. Check and review population/activity outputs using the verification wrapper.
4. Ask PlannerAgent for commute, leisure, and home mobility forces.
5. Combine those forces with spatial, building, and temporal features for GBR
   prediction.

Population/activity generation prompts and the core prediction logic are retained.
Provider settings and private data must be supplied by the user. No generated
cohorts, schedules, predictions, or experiment logs are included in the release.

## Observation construction

The original notebook uses a **file-level OD representation**. For each input
file, it uses `flow_count` when present; otherwise it sums columns containing
`Trips`. Its temporal context is derived from the file name: the first matching
number is used as an hour-like feature, and the historical `10-13` batch is treated
as a weekend batch. These conventions are preserved for traceability to the
original workflow, not endorsed as a general-purpose timestamp parser.

The later controlled hourly evaluation constructs individual observations from
dated hourly columns. That is a substantive protocol difference. The code for
those additional comparisons is intentionally not part of this release. Do not
present a run of this original workflow as an exact reproduction of every result
in the current manuscript.

The Manhattan OD comparison shown in the README is a manuscript illustration. It
is not an output newly generated from this package. It juxtaposes observed TomTom
corridors and a representative complete-model prediction for qualitative spatial
inspection, not quantitative benchmarking.

## Prediction settings

| Component | Original setting |
|:--|:--|
| Spatial representation | Two-dimensional MDS using training-flow dissimilarities |
| MDS random state | `42` |
| Regressor | `GradientBoostingRegressor` |
| Number of estimators | `100` |
| Learning rate | `0.1` |
| Maximum depth | `5` |
| Regressor random state | `42` |
| Scaling | `StandardScaler`, fitted on training observations |
| Output constraint | Negative predictions clipped to zero |

The feature vector contains MDS distance, destination commercial capacity,
destination residential capacity, temporal context, a weekday indicator, and
three PlannerAgent mobility-force values. Topology and scaling are learned from
training files; the held-out observations are used for prediction evaluation.

Fixed numerical random states do not make the entire pipeline deterministic.
LLM generation, provider behavior, source-data revisions, and building subtype
assignment can introduce variability. Save your own configuration, data versions,
dependency versions, and execution logs when running a study.

## Verification is part of the original workflow

Population and activity outputs undergo deterministic checks, self-review and
repair, discussion by Advocate/Skeptic/Statistician roles, and a referee decision.
Acceptance requires both a passing deterministic report and an accepting referee
decision. The original wrapper permits up to three generation rounds.

Important implemented tolerances include:

- Population totals: within 2% of the ACS total.
- Activity values: a numerical bounds tolerance of `0.001`.
- Per-hour activity-share sums: within `0.05` of one.
- Activity coverage: all 24 hours and the expected cohort identifiers.

These are structural constraints. They do not demonstrate demographic fidelity,
activity realism, or improved OD predictive accuracy on their own. This release
does not run a comparison between alternative verification policies.

## Packaging and runtime repairs

The release removes saved notebook outputs and replaces private endpoint settings
and machine-specific paths with user configuration. It also makes narrowly scoped
repairs needed for a portable, inspectable execution:

- Replace an unfinished JSON-retry placeholder with a JSON-only correction
  request.
- Use consistent building-cache path capitalization across operating systems.
- Replace an unsupported pandas `Series.floor()` call with NumPy flooring.
- Disable LLM-generated code execution in the conversational agents.
- Record failed configurations explicitly instead of treating failure as zero
  predictive error.
- Align exported OD rows with the rows actually used for prediction.
- Make model labels safe for export filenames without changing provider IDs.
- Record one final status per configuration after its output-writing step.

These changes do not introduce alternative predictors, replacement agents, or
new comparison conditions. They also do not guarantee bit-for-bit reproduction
of historical runs, particularly when a repaired branch is exercised.

## Reading a run responsibly

Inspect run status, validation reports, and any fallback warnings before using
metrics. The original PlannerAgent can substitute a neutral force vector when a
response cannot be parsed, and building processing can use fallback densities.
Neither fallback should be described as a successful empirical observation.

R², RMSE, MAE, and CPC describe the held-out inputs used in that run. They do not
establish transfer to another city, season, or data source. Comparisons with other
systems require matched observations, preprocessing, and evaluation definitions;
this repository does not supply such a comparison suite.

Offline tests check implementation behavior without live model calls. A successful
test or preflight check does not verify provider access or replace an end-to-end
evaluation using appropriately licensed data.
