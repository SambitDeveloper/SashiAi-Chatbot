# Query-planning reliability implementation plan

## Status and scope

**Status: The previously approved reliability changes are applied. This document now proposes a further semantic-interpretation redesign; that redesign is not approved or implemented.**

Keep Qwen3-4B as the baseline model and keep the deterministic fast path. MCP remains an optional repository/execution adapter, not a prerequisite for planner work. Preserve existing behavior except where a validated request cannot safely be represented or answered.

The current `QueryPlan` already represents the reported date-filtered status breakdown: an admission-date start of 2026-08-02, status inclusion for Active and Discharged, grouping by status, and a room-number projection. Group counts are inherent in `breakdown`; resident identity/name is included in projected rows. Do not redesign `QueryPlan` for this case. If implementation reveals a genuinely unsupported generic operation, document the exact gap and propose the smallest compatible schema extension before changing it.

The working tree already contains uncommitted changes. Implementation must preserve them and make only changes approved in this plan.

## Architecture reassessment: compositional semantic interpretation

The latest `either Active or Away` case exposed a limitation in the current `ClauseScopeAnalyzer` and coverage ledger. The extractor correctly produced the `list` operation, `status IN [active, away]`, and `admission_date.start = 2025-01-02`, but its lexical audit marked `either` unresolved and made the requirements incomplete. Thus the backend rejected a semantically valid, schema-grounded plan before execution. This is a false rejection in the audit, not a Qwen planning failure.

The current flat `RequestRequirements` list plus surface-span coverage audit should **not remain the semantic completeness authority**. Do not repair it by adding `either`, more connective words, or a growing list of special cases. Keep useful deterministic analyzers as evidence providers and keep their precise date/value normalization, but replace token accounting as the gate with a compositional semantic representation and explicit supported-subset checks.

### Options and trade-offs

| Approach | Accuracy and limitations | Latency / deployment cost | Decision |
|---|---|---|---|
| Continue extending `ClauseScopeAnalyzer` and its word/config vocabulary | Low false positives on known patterns, but open-ended phrasing keeps producing false rejections; a list of covered words cannot establish scope or meaning. New exceptions also risk silently misreading coordination. | Very low runtime and no new dependency, but ongoing rule maintenance and unpredictable edge-case regressions. | Reject as the primary completeness method. Keep only deterministic, bounded analyzers for well-defined operations such as date normalization and live-vocabulary grounding. |
| Add a general syntactic parser (for example, dependency parsing or semantic-role labeling) | Can help identify clause structure and argument roles, but syntax is not the same as domain meaning; it does not by itself resolve status values, date operators, field roles, group-local scope, or unsupported relationships. A parse can also be confidently wrong. | Adds a package and parser/model assets, startup and CPU/memory cost, deployment/version management, and possibly language-model licensing review. Runtime must be measured on the production host. | Do not adopt as a stand-alone semantic verifier. Consider only if benchmarked evidence shows a specific structural-analysis gap it resolves. |
| Use the existing Qwen3-4B call to emit a typed semantic AST, then compile it deterministically to `QueryPlan` and validate it in the backend | Handles compositional meaning without requiring an ever-growing English token list. The AST remains a model interpretation and can still be wrong or omit meaning; provenance and schema grounding reduce fabrication but do not prove entailment. Backend checks can independently prove AST/plan consistency and execution safety, not arbitrary English correctness. | No additional model/service/package. Replacing the current plan response with a typed AST may add output tokens; measure single-call and end-to-end latency, timeout rate, and any retry cost. | Recommended first design to prototype and benchmark, with explicit limitations and fail-closed handling for unsupported or ambiguous ASTs. Keep Qwen3-4B as the baseline. |
| Add a separately trained local semantic verifier/parser | Potentially provides a genuinely separate source-to-meaning signal; independent errors can reduce correlated failures if measured. It is not a correctness oracle: model disagreement, shared training biases, and verifier false positives/negatives remain possible. A second interpretation from the same Qwen is not independent verification. | Requires an additional model/runtime, memory/CPU budget, deployment and update procedures, and at least one more inference or a substantial loading cost. On this CPU-bound deployment it may materially increase tail latency and overload risk. No external API is proposed. | Defer. Consider only if the AST prototype's independently reviewed benchmark shows unacceptable false executions and a candidate verifier measurably improves them at acceptable latency and resource cost. |

No technique can guarantee that arbitrary English has been understood without a semantic interpreter or a bounded formal language. The design must state this limitation instead of treating either all-token coverage or model confidence as proof. Use benchmarked interpretation quality, mechanically checkable invariants, safe rejection of unsupported semantics, and independently reviewed fixtures. If a request cannot be represented or its scope is uncertain, it remains non-executable.

### Proposed canonical representation

Introduce a typed `RequestIR` (the successor to flat `RequestRequirements`) whose semantic expression is compositional and whose nodes retain original-question evidence:

- A boolean expression tree for predicates: `And`, `Or`, `Not`, comparisons, set membership, missingness, and explicitly supported relationship predicates.
- Separate operation, grouping, aggregate, projection, ordering/limit, entity/reference, and conversation-context nodes so a field's filter/group/output roles cannot be conflated.
- Typed normalized values and operators, preserving source operator and date granularity (for example strict `after` versus inclusive `on or after`) until lowering is proven equivalent.
- Source spans and provenance for each semantic node, plus node-level status: supported, ambiguous, unsupported, or unresolved. A model-provided “complete” flag is evidence only, never authorization.
- Define span coordinates explicitly. If the prompt uses masked text, the server maintains and validates the offset map back to the original question; model-provided spans must be in bounds and their text must match the corresponding masked source. Never expose unmasked resident names to obtain provenance.
- Explicit unsupported-node representation. Never discard unknown clauses, connective scope, or relationships merely to produce a valid `QueryPlan`.

**Evidence is not semantic proof.** Keep these validation claims separate:

| Check | What it establishes | What it does not establish |
|---|---|---|
| Source evidence | A cited phrase/span exists in the original or masked question and is correctly mapped. | That the span was interpreted with the correct polarity, connective, scope, or role. |
| Schema/entity grounding | The field and canonical value exist in the permitted schema/vocabulary and references are authorized. | That the user intended that value or field. |
| AST validation | The expression uses known, well-formed node types and legal operands. | That the expression captures every requested clause or the intended logic. |
| IR-to-plan equivalence | Compilation preserves the AST's represented semantics. | That the AST itself means what the user intended. |
| Semantic accuracy | Agreement with independently reviewed expected meaning and results on held-out fixtures/minimal pairs. | A proof of correctness on every unseen natural-language question. |

Use the existing Qwen3-4B call as the candidate semantic interpreter: it returns one structured `RequestIR`, including evidence spans and any unresolved/unsupported material it identifies. The model's status labels and completeness claim are diagnostic only. The backend maintains an authoritative allowlist of executable IR node types/operators and independently rejects unknown, malformed, ungrounded, unauthorized, or unrepresentable constructs. A model cannot make a node executable by labeling it supported.

The backend can verify evidence alignment, schema validity, authorization, AST structure, supported-subset membership, and compilation equivalence. It cannot mechanically prove that Qwen has not omitted a clause or misread a valid span's meaning. Do not claim that it can. Measure semantic omissions and wrong scope against independently reviewed fixtures. If the required production risk tolerance demands per-request independent source-to-meaning verification beyond these checks, evaluate a separately trained local verifier as a distinct, explicitly approved option; a second output from the same Qwen is not independent.

Compile the validated IR to `QueryPlan` with a deterministic compiler. Check a normalized round trip: the compiled plan must preserve the executable IR's filters, boolean operators, grouping, aggregation, projection, and entity references exactly. The plan is not a second meaning source. Any mismatch, lossy lowering, or unsupported expression blocks execution. The deterministic fast path remains and must produce the same IR contract (or an equivalently validated representation) before using the same execution gate.

### Initial compilation boundary

The AST may describe more than the first executable subset. The compiler/backend—not the model—decides whether a request can execute:

| Initially executable when losslessly represented | Initially non-executable |
|---|---|
| AND across supported predicates/conditions. | Cross-field OR. |
| Same-field OR for supported set-valued fields, lowered to `include`. | Arbitrary nested boolean expressions or negation that cannot be reduced exactly to supported filter semantics. |
| Supported same-field exclusion/missingness and supported date/age comparisons, with strict/inclusive boundaries preserved at the field's granularity. | Arbitrary group-local filters or predicates whose scope cannot be represented by the plan. |
| Existing `group_by`, supported count/aggregate operations, and requested projections. | Unsupported relationship traversal or joins not implemented by the current repository/executor. |
| Existing resident lookup/reference behavior only when the resolver authorizes it and its plan semantics are preserved. | Any unknown IR node/operator, ungrounded value, ambiguous scope, or IR-to-plan lowering that drops or changes meaning. |

For example, `status = active OR status = away` may lower to one same-field `include` condition. `status = active AND status = away` must not be changed into that OR; if the current plan cannot represent the conjunction exactly, compilation rejects it. A generic operation outside the executable subset must clarify/return unsupported with no data query. Do not extend `QueryPlan` or the executor until an independently reviewed fixture demonstrates a valuable generic operation that cannot be represented and the extension is separately approved.

## Design invariants

1. Keep the original question authoritative; preserve its interpreted semantics and evidence through planning, retry, validation, and execution.
2. Keep the compositional `RequestIR` distinct from executable `QueryPlan`, but do not mislabel a Qwen-produced IR as independent from Qwen. Neither model output is trusted on its own; the deterministic compiler and backend validators must verify grounding, structure, supported semantics, and lossless equivalence.
3. Execute a data query only after the candidate IR and its compiled plan pass schema, entity-grounding, semantic-equivalence, and execution-safety validation. On uncertainty, clarify or refuse safely.
4. A retry is a repair attempt, not permission to discard previously validated semantics. Validate the entire repaired IR and compiled plan against the original question and evidence.
5. Do not add a question-specific handler, keyword rule, fabricated resident identity, external model, or required MCP dependency.
6. Preserve tenant/data scope, field whitelists, privacy rules, and read-only database access.
7. Keep request-level diagnostics useful while redacting resident/staff names and sensitive condition values from logs.

## Implementation stages

### Stage 0 — Capture the pre-change baseline

**Objective:** Record comparable results from the current implementation before changing application code, prompts, tests, configuration, or benchmark scoring.

**Current behavior:** `backend/scripts/benchmark.py` can run the deterministic path and Qwen3-4B against the existing gold, held-out, and paraphrase datasets. Its current breakdown signature does not score resident members or projections, so some requested metrics are not available from its report alone.

**Proposed behavior:**

- Before any implementation edits, run and preserve the current deterministic and Qwen3-4B benchmark reports, with the exact code revision/worktree state, model identity, prompt/schema version, sampling settings, timeout, concurrency, host, dataset, and fast-path settings recorded.
- Capture before-results for deterministic accuracy and Qwen3-4B plan accuracy, missing/invented requirements, safe rejection, grouped resident records, projections, latency, and timeout rates.
- For dimensions the existing benchmark cannot measure, capture a pre-change result using the current application and an independently prepared, isolated fixture/oracle. Do not first alter the benchmark harness and then call that the baseline.
- If an existing dimension cannot be measured before changes, record it as unavailable with the reason; do not imply a before/after improvement for it.
- Keep the baseline reports immutable and use the same datasets, model settings, test data, and scoring definitions for after-results.
- The stored Stage 0 reports predate the already-applied flat-ledger implementation and therefore are not a before-result for this proposed semantic-IR redesign. Before implementing this redesign, capture a separate immutable baseline of the current worktree, including the either/or false rejection, exact semantic accuracy, safe rejection/false execution, and latency/overload rates. Do not overwrite the original Stage 0 artifacts.

**Relevant files:**

- `backend/scripts/benchmark.py` — current baseline runner; use without modification for the initial report.
- `backend/tests/data/gold_questions.py`, `backend/tests/data/eval_50.py`, `backend/tests/data/paraphrases.py`, `backend/tests/data/residents.json` — current benchmark data and fixture source.
- New baseline report artifacts — store outside application code or in the agreed benchmark-artifact location; do not overwrite or regenerate the pre-change report during implementation.

**Regression risks:**

- A different model build, prompt, context, fixture, or hardware setting makes before/after comparisons invalid.
- The current scoring omissions can make baseline accuracy look higher than end-to-end grouped-detail accuracy.
- A baseline run may be affected by timeouts or environmental load; preserve raw reports and record those conditions rather than silently dropping failures.

**Tests / verification required:**

- Verify both current modes complete and record deterministic accuracy, Qwen3-4B plan accuracy, latency distribution, and timeout rate.
- Independently verify the grouped projection baseline, including expected members and fields, against a fixed isolated dataset.
- Confirm each requested metric is either captured or explicitly marked unavailable before any implementation change.
- Record a reproducible command/configuration manifest for the after-run.

### Stage 1 — Replace surface coverage with compositional `RequestIR`

**Objective:** Represent the original question's meaning compositionally, retain evidence, and separate interpretation from executable planning without treating token coverage as semantic proof.

**Current behavior:** `RequestRequirements` composes heuristic results into a flat list and uses `ClauseScopeAnalyzer` plus a lexical coverage audit to decide completeness. For the logged `either Active or Away` query, the audit correctly extracted the date and status predicates but rejected the question because the connective preface `either` was not in its recognized vocabulary. This demonstrates that source-token coverage is not semantic completeness and that adding vocabulary entries would not scale.

**Proposed behavior:**

- Replace the current flat list and `ClauseScopeAnalyzer`-as-completeness-gate with the typed compositional `RequestIR` defined above. Keep it separate from `QueryPlan`; it is the candidate interpretation contract and source of truth for deterministic compilation, not a model-authored authorization token.
- Use the existing Qwen3-4B call to produce the semantic IR instead of asking the model for a separate, parallel requirements interpretation and plan. Require typed nodes, normalized candidates, evidence spans, and explicit ambiguous/unsupported/unresolved material. Do not trust model confidence or a `complete` field as proof.
- Keep date parsing, live vocabulary/entity grounding, reference checks, and projection/structure detectors as backend evidence and normalization sources. They may confirm or constrain IR nodes; they may not prove completeness merely because their token spans were consumed.
- Remove the “every meaningful token must be in a vocabulary” gate. Unknown function words and connective realizations are not semantic omissions by themselves. A semantic node without evidence, a source span that cannot be aligned to its claimed evidence, a contradictory parse, ungrounded values, or material that the parser flags unresolved/unsupported prevents compilation.
- Require the IR to preserve meaning, not wording: capture boolean composition and scope in the tree; retain separate roles for filters, groups, aggregates, projections, entities, and relationships. Do not flatten nested/cross-field OR, negation, group-local predicates, or relationships to fit the current plan.
- Record provenance/status on each semantic node. Backend checks independently verify schema/type validity, live value grounding, reference authorization, date/operator normalization, expression well-formedness, supported executor semantics, and lossless IR-to-plan lowering. These checks do not claim to prove arbitrary English entailment; interpretation correctness is measured with independently reviewed fixtures and held-out minimal pairs.
- Preserve the original user question as authoritative input to interpretation and validation. Conversation context may satisfy an explicit follow-up reference only when resolved to the correct session/turn; it must not override explicit current-turn conditions.
- Keep resident names masked in model-facing data where identity can use placeholders. Evidence spans and values used for verification must be redacted from routine diagnostics.
- The table specifies how the candidate semantic parse, independent backend checks, uncertainty behavior, and tests cover each category:

| Category | Interpreter and backend mechanism | Mechanical checks vs. semantic measurement | Uncertainty behavior | Required tests |
|---|---|---|---|---|
| AND versus OR | Qwen3-4B emits `And`/`Or` nodes with operand evidence; backend validates tree shape and compiler equivalence. | Mechanical: preserve the represented tree; same-field OR may lower to `include`, and unsupported cross-field/nested OR cannot be flattened. Semantic omission or a wrong connective in the AST is measured against reviewed minimal pairs, not proven by spans. | Reject invalid/unsupported ASTs and lossy lowering. If the AST omits a connective or clause, backend checks alone cannot guarantee detection; measure that risk and use a separate verifier only if independently justified. | `either A or B` vs `A and B`; same-field vs cross-field OR; mixed/nested coordination and reordered phrases; no repository call for unsupported forms. |
| Before/after and greater-than versus greater-than-or-equal | IR carries comparison operator, typed date/number, granularity, and evidence; existing date/number parsers normalize supported values. | Mechanical: exact boundary-preserving lowering when the comparison is present. Whether Qwen selected the intended operator is measured against reviewed examples and boundary truth tables. | Reject conflicting/unsupported precision or ambiguous targets; evidence that a date exists does not prove the comparison is right. | Before/after, earlier/later, <, <=, >, >=; day/month/year; exact-boundary records. |
| Negation | IR represents `Not` over a predicate/subtree; missing-value and entity analyzers contribute grounded facts. | Mechanical: reject polarity/scope changes during lowering. Whether Qwen attached negation to the right clause/value is a semantic-accuracy measure, not a span-validation result. | Reject ambiguity/unsupported negation; backend cannot infer an omitted negation from valid positive nodes alone. | Positive/negative pairs, “not A but B,” “A and not B,” missing-value negation, double/ambiguous negation. |
| Group-specific conditions | IR attaches predicates to global scope or a group branch; live vocabulary resolves group values. | Mechanical: ensure represented scope is preserved; arbitrary per-group WHERE branches are not compiled. Omitted or misattached source conditions are measured with reviewed expected ASTs/results. | Reject ambiguous or unsupported scopes. A valid span does not prove the group scope is correct. | Global versus distinct group-local conditions; clause reorderings; unsupported cases make no repository query. |
| Filters versus grouping fields versus requested output fields | Distinct IR nodes represent predicate, group key, aggregate, and visible projection; backend schema/projection checks supply constraints. | Mechanical: validate roles present in the AST and prevent unrequested fields/internal fields from leaking. Whether Qwen omitted or misclassified a field role is measured with reviewed fixtures. | Reject conflicting/unrepresentable roles; do not claim source-span coverage proves role correctness. | Same field as filter/group/projection, supported combinations, internal-field handling, and missing/unrequested visible output. |
| Multiple clauses, entities, relationships, and follow-up context | IR represents clause composition and typed entity/relationship nodes; `ReferenceResolver` and session context authorize references. | Mechanical: verify entities, context scope, and preservation of represented clauses. Missing or mis-scoped clauses are measured independently; arbitrary relationship traversal remains unsupported. | Reject unresolved references, unauthorized entities, or unsupported relationships. Backend checks cannot guarantee detection of an entirely omitted model clause. | Multi-clause requests, references with/without history, relationships, add/replace/inherit follow-ups, cross-session isolation. |

**Relevant files:**

- `backend/app/nlu/requirements.py` — replace the flat ledger/lexical completeness gate with typed `RequestIR`, provenance, status, and conversion helpers.
- `backend/app/models/plan.py` — current restricted execution plan; change only for an approved, fixture-proven generic representational gap.
- `backend/app/nlu/qwen_translator.py`, `prompt.py`, `translator.py` — structured semantic-IR response contract and compiler integration; preserve Qwen3-4B and avoid a second self-check call.
- `backend/app/nlu/structure.py`, `grammar.py`, `projection.py`, `missing_values.py`, `references.py`, `vocabulary.py` — retain bounded evidence/normalization and deterministic fast-path support, not global semantic-completeness authority.
- `backend/app/engine/service.py`, `grounding.py` — construct/validate IR once and make lossless compilation plus backend safety checks the gate before execution.
- `backend/app/engine/executor.py`, `backend/app/data/repository.py` — inspect only for semantics the current plan/executor cannot represent; unsupported expressions must remain non-executable.
- `backend/tests/` — independently reviewed semantic fixtures, minimal pairs, AST/plan equivalence, grounding, and false-rejection/false-execution tests.

**Regression risks:**

- A model-generated AST can still be semantically wrong or incomplete; schema, provenance, and compiler checks cannot prove arbitrary English entailment. Measure errors and false acceptance explicitly; do not claim formal semantic certainty.
- If an AST's structure grows beyond the current `QueryPlan`, tempting lossy flattening would change results. Reject unsupported forms until a separately approved schema/executor extension is implemented and tested.
- Returning ASTs may increase completion tokens and latency; a second verifier would increase latency and resource contention. Compare measured p50/p95, timeout/overload rates, and memory before adopting extra components.
- A second model may share correlated errors or falsely reject valid plans; require evidence of improvement on independently reviewed held-out fixtures before adding it.
- Evidence-span alignment and live-vocabulary checks can reject fabrication but may not establish that the referenced words mean what the AST claims. Keep the limits explicit.
- Preserve privacy: resident names remain masked in model inputs where possible and must not appear in routine logs or benchmark reports.

**Tests required:**

- Independently reviewed semantic fixtures with provenance and expected IR, including the `either Active or Away` case. Reviewers must verify question, AST, plan semantics, and expected result separately; fixtures must not be generated solely by Qwen or the production executor.
- Add and retain this independently reviewed minimal-pair matrix before building the compiler:

| Question A | Question B | Expected semantic difference |
|---|---|---|
| Residents who are Active or Discharged | Residents who are Active and Discharged | OR versus AND; do not rewrite AND into same-field OR. |
| Admitted before August 1 | Admitted on or before August 1 | Strict versus inclusive date boundary. |
| Older than 80 | 80 or older | Strict versus inclusive numeric boundary. |
| Not Active | Active | Negated versus positive status predicate. |
| Active residents in each facility | Active residents in Skilled Nursing facilities only | Grouping by facility versus filtering to a facility value. |
| Active or Away and admitted after 2025 | Active and Away and admitted after 2025 | Boolean composition differs; if the conjunction is not representable, reject rather than flatten. |

- For every fixture, independently prepare and review the expected `RequestIR`, compiled query semantics, and expected result. Run the exact question through Qwen, compare its AST with the expected AST, compile only if backend checks pass, and compare execution with independently calculated expected results. Do not derive expected ASTs with the production prompt/model/compiler/executor.
- Minimal pairs should also cover negation scope, projection versus filter/group roles, group-local predicates, entity subject versus reference, and follow-up context.
- Property-based and hand-authored truth-table tests for boolean IR evaluation and lossless lowering of the supported subset.
- Tests that every IR node has valid source evidence, values/fields are grounded, unknown/unresolved semantic nodes block, and unsupported AST constructs never reach the repository.
- Tests comparing normalized IR semantics with compiled `QueryPlan` semantics; any AST-to-plan loss or mismatch blocks execution.
- A specific regression test proving the correct plan for the reported either/or request is not rejected because of an unrecognized connective token.
- Held-out paraphrase and adversarial tests measuring exact semantic accuracy, false rejection of correct parses, false execution, and safe rejection; keep tuning and evaluation sets separate.
- Report source-evidence validity, schema grounding, AST structural validity, IR-to-plan equivalence, and semantic accuracy as distinct metrics. Do not use any of the first four as a proxy for the fifth.
- Tests for date boundaries, aggregation, projections, masked entities, follow-ups, supported relationships, and deterministic fast-path parity.
- Privacy tests proving names are masked in model input and redacted in logs.
- Regression tests confirming deterministic fast-path questions still produce their existing plans.

### Stage 2 — Validate semantic IR and compiled plan

**Objective:** Reject detectable structural, grounding, representational, compilation, and safety errors without overstating the backend's ability to detect semantic omissions in a valid-looking model AST.

**Current behavior:** Pydantic validates `QueryPlan` shape; `resolve_plan` resolves database entities and rejects unsupported resident references; `check_plan` applies semantic checks. The flat requirement comparison and lexical completeness audit can reject correct plans for surface vocabulary omissions, while neither a model-authored requirements object nor a model-authored plan is an independent semantic oracle.

**Proposed behavior:**

- Validate the candidate `RequestIR`, then compile it deterministically to `QueryPlan`. Reject malformed ASTs, ungrounded or fabricated values/references, unresolved essential nodes, and semantics outside the executable subset.
- Independently verify that the compiled plan preserves the IR's operation, boolean filters/operators/values, grouping, aggregate, requested projection, relationship scope, and entity references. Treat plan/IR disagreement or lossy boolean lowering as a rejection, not a model-repair hint that can authorize execution.
- Identify missing, extra, or contradictory components relative to the accepted IR; reject semantically inconsistent combinations, including a lookup of a comparison-only entity. Missing meaning in an otherwise valid-looking IR is a semantic interpretation error measured by independent fixtures, not something source-span checks can guarantee to detect.
- Require every entity reference in a lookup to map to a resident reference explicitly present in the original question or valid conversation context. Never synthesize resident IDs or names.
- Distinguish user-visible output from internal execution requirements. Compare `QueryPlan.fields`/`output_fields` with the user's requested projection, and reject unnecessary user-visible fields; separately allow schema-grounded internal fields needed to filter, group, join, or calculate results. Internal fields must not be presented as requested output or leak into the response. Do not reject a valid plan merely because the executor needs an internal group key or join field.
- Run backend schema, grounding, provenance, source-span, AST integrity, authorization, supported-subset, semantic-equivalence, and execution-safety checks before any data query, for model, deterministic, and cached plans. These checks are independent of Qwen output as to execution correctness, but they do not prove arbitrary natural-language entailment or guarantee detection of an omitted source clause.
- If an operation is not representable by current `QueryPlan` or the repository, return a safe unsupported/clarification result. Document the exact generic capability gap and only then propose a compatible schema/executor extension.

**Tenant-isolation security verification (required before real resident data is exposed):**

- Do not assume tenant isolation exists because the current scope is preserved. The audit identified that tenant scope is not yet established in the plan-cache context; the actual request-to-tenant boundary and repository enforcement must be verified explicitly.
- Identify the trusted tenant/property identity source and prove that it is bound server-side to every vocabulary/entity index, resident query, reference resolution, cached/failure/session plan, and execution operation. Never trust a model-supplied tenant or property identifier.
- If a tenant identity or enforceable repository boundary is absent, mark tenant isolation as unverified and block exposure of real resident data in multi-tenant use until the boundary is implemented and tested.
- Verify cache keys and cache contents cannot be reused across tenants, including standalone plans, follow-ups, failed-plan entries, and any resident-name/ID resolution state.

**Relevant files:**

- `backend/app/engine/grounding.py` — replace or compose current dispersed semantic checks with IR grounding and IR-to-plan semantic-equivalence checks.
- `backend/app/engine/resolver.py` — retain exact entity resolution and pre-resolution reference checks; integrate requirement reference validation.
- `backend/app/engine/service.py` — make the validation result the explicit gate before execution; verify tenant identity propagation and tenant-aware cache/session scope rather than assuming it.
- `backend/app/models/plan.py` — change only if a verified unsupported generic operation requires it.
- `backend/app/data/resident_view.py` and `backend/app/data/repository.py` — change only if the verified operation is not executable by the current repository contract.
- `backend/app/api/routes.py`, `backend/app/data/mcp_repository.py`, repository implementations, and vocabulary/index construction — inspect the authentication/tenant boundary and enforce it at the trusted data-access boundary where required.
- `backend/tests/` — add cross-tenant isolation and cache/reference noninterference tests; identify the authoritative tenant source before implementation.

**Regression risks:**

- An overly strict comparison can reject valid synonyms or database aliases; canonicalization must happen before comparison.
- An overly permissive comparison can allow missing filters or fields to reach execution.
- The validator must distinguish a field used as a filter from the same field requested for display.
- Confusing internal execution fields with visible projection fields could either reject valid queries or disclose unrequested resident data.
- Existing cache and follow-up plans must be revalidated against the current question and context.
- Tenant identity may not currently be available to all repository/index/cache layers. Do not claim isolation until the trusted boundary is identified and enforced.

**Tests required:**

- Positive tests where all requirements are represented, including the reported date-filtered grouped query.
- Negative tests removing each required component in turn: filter, group, aggregation, projection, and entity reference.
- Negative tests for extra filters/fields, fabricated references, and comparison-only lookups.
- Tests showing legitimate internal group/join/filter fields remain usable but are absent from user-visible projections unless requested.
- Tests that a rejected plan produces no repository data-query call.
- Tests that direct and cached plans still pass through the same safety gate.
- Security tests that two tenants with overlapping names/IDs cannot resolve, query, execute, or reuse each other's cache/session/failure entries.
- Tests that missing or untrusted tenant identity fails closed before resident data is returned.

### Stage 3 — Make retries constrained repairs

**Objective:** Preserve valid semantic interpretation components and repair only what backend validation identifies.

**Current behavior:** A single retry receives the original masked question, date hints, previous candidate JSON, a requirements payload, and textual validation feedback. `preserve_verified_components` can retain source-verified components only when they were present in the prior candidate. It cannot recover an omitted component from an incomplete candidate. A repeated candidate may therefore consume the retry without fixing the omission.

**Proposed behavior:**

- Pass the original question, prior candidate `RequestIR`, and exact structured backend validation errors to the retry. Preserve source evidence and the same conversation context.
- Instruct the model to repair only identified IR nodes. Retain valid components after backend validation, but do not treat model-authored requirements or an unverified previous component as proof.
- Do not blindly merge the Qwen candidate with a second Qwen-generated requirements list. Any deterministic completion from independently grounded backend facts must be explicit, semantics-preserving, and revalidated; otherwise require a fresh valid candidate or clarify.
- Compare the repaired IR with the original question's available verified evidence and compile only when structurally complete, supported, and consistent. Do not claim that evidence alignment proves all source meaning was captured; test semantic omissions against the independent fixture set. Reject retries that drop represented requirements or add unsupported semantics.
- Re-run full entity resolution, schema validation, safety checks, IR-to-plan equivalence, and execution-safety validation after every retry.
- Execute no partial plan. A timeout, invalid retry, or unresolved requirement ends in a safe clarification/unavailable result.
- Retain the current one-retry limit initially to control latency. Consider additional retries only if measured benchmark results justify them and safe limits are defined.
- Measure latency separately for a single model call, the repair/retry call, end-to-end retry requests, concurrent requests, timeout, and cancellation. Record queue/slot wait and inference/request time separately where available.
- Test cancellation propagation: cancelling a request must not execute a partially planned query, leak a semaphore slot, or leave unbounded/abandoned model work. Verify timeout and cancellation are reported distinctly.

**Relevant files:**

- `backend/app/engine/service.py` — retry lifecycle, immutable original requirements, retry gate, and no-execution behavior.
- `backend/app/nlu/translator.py` — retry request contract.
- `backend/app/nlu/prompt.py` — retry instruction and formatting of requirements/errors.
- `backend/tests/test_retry_policy.py`, `backend/tests/test_service.py` — retry preservation and end-to-end gating.

**Regression risks:**

- Larger retry prompts can increase latency and timeout frequency on the CPU-hosted model.
- Retry concurrency and cancellation can expose slot leaks, abandoned HTTP requests, or misleading latency accounting.
- Requirements serialization can leak names if it bypasses masking.
- Blindly copying the prior candidate could preserve a bad component; only the full validator can authorize execution.
- Timeout handling must not be reported as a successful repair or as a model interpretation result.

**Tests required:**

- Retry receives the same original requirements and candidate, plus exact validation errors.
- A retry that fixes one error but drops a previously valid requirement is rejected.
- A retry that introduces an unsupported filter or entity is rejected.
- Successful repair executes only after complete revalidation.
- Timeout, malformed output, or repeated-invalid output does not invoke a data query.
- Existing retry count and model-call telemetry remain consistent.
- Measure and report single-call, retry-call, concurrent-request, timeout, and cancellation latency/outcomes on the target hardware.
- Stress-test cancellation while waiting for a model slot and during inference; verify slot release and no post-cancellation query execution.

### Stage 4 — Improve prompts and constrained output

**Objective:** Reduce misleading priors and constrain the proposed semantic-IR shape with balanced, schema-grounded examples.

**Current behavior:** The prompt and JSON schema describe valid fields and operations, but few-shots include caregiver filtering and a `resident_1` lookup. The current schema constrains plan shape but does not enforce semantic completeness. The question brief is optional and disabled by default.

**Proposed behavior:**

- Review and remove or rebalance caregiver and `resident_1`/`resident_2` examples that are not needed to teach the schema.
- Add a small, balanced set of general examples combining filters, grouping, per-group counts, and resident projections. Ensure examples are not copies of the reported question and use valid schema values.
- Keep explicit rules against inventing references and make requirements/errors visible during retry.
- Define a compact, typed output schema for `RequestIR`; constrain required node shapes where supported by llama.cpp JSON-schema grammar. Retain backend IR validation and deterministic compilation as authoritative.
- Keep prompts compact and cache-friendly. Do not add question-specific keywords or handlers.

**Relevant files:**

- `backend/app/nlu/prompt.py` — system instructions, few-shots, and constrained output schema.
- `backend/app/nlu/qwen_translator.py` — schema delivery and model-call behavior; no model change in this stage.
- `backend/tests/test_plan_and_prompt.py`, `backend/tests/test_qwen_translator.py` — prompt/schema contract.

**Regression risks:**

- Additional examples increase prompt-prefill time and can worsen timeouts.
- Overly restrictive JSON Schema features may not be supported consistently by llama.cpp grammar conversion.
- Examples can create new biases if imbalanced or too similar to evaluation questions.

**Tests required:**

- Prompt tests ensuring removed misleading examples do not reappear and examples remain balanced and valid.
- Schema tests for action/field compatibility and rejection of malformed outputs.
- Measure prompt and completion tokens; single model-call and end-to-end latency; p50 and p95 latency; timeout rate; retry call latency; and behavior under representative concurrent requests using Qwen3-4B on the target deployment hardware. Compare with the captured current-worktree baseline, and report unavailable/incomparable results explicitly.
- Keep semantic validation tests even when constrained decoding rejects some malformed shapes.

### Stage 5 — Strengthen benchmark ground truth and output scoring

**Objective:** Verify exact plan semantics and final grouped resident results independently of the application executor.

**Current behavior:** `backend/scripts/benchmark.py` compares expected and generated plans and executes both using `MemoryResidentRepository`. Breakdown signatures include group keys and counts, but not resident members or projections; plan normalization also omits projected fields. Expected plans and actual plans therefore share the same executor, and benchmark execution does not currently establish MongoDB result correctness.

**Proposed behavior:**

- Score exact filters/operators/values, operation, grouping, aggregation, projections, and entity references.
- For grouped projections, compare each group’s count and complete resident identity/name/requested fields.
- Check total matched count against returned members and explicitly test completeness, row caps, and pagination/truncation indicators.
- Add independently verified expected result fixtures for the benchmark dataset. Do not derive expected outputs solely by executing expected plans through the production executor.
- Add integration coverage against an isolated, seeded MongoDB test database (never production data) to verify aggregation results and completeness independently.
- Expand the set with unseen combinations and paraphrases; keep a held-out portion untouched during prompt/validator tuning.
- Compare Qwen3-4B against the same tests and settings first. Evaluate alternative models only after the architecture is stable.
- Compare every after-result for this redesign to the separate immutable current-worktree baseline captured immediately before its implementation. Keep the earlier Stage 0 reports as historical references only; they predate the flat-ledger implementation. Compare deterministic accuracy, Qwen3-4B semantic accuracy, omissions/inventions, false acceptance/rejection, safe rejection, grouped results, projections, latency, overloads, and timeouts with fixed scoring definitions. Report unavailable dimensions; do not imply a Qwen before/after improvement from incomplete data.

**Relevant files:**

- `backend/scripts/benchmark.py` — plan and result scoring, metrics, and model-run reporting.
- `backend/tests/data/gold_questions.py`, `backend/tests/data/eval_50.py`, `backend/tests/data/paraphrases.py` — curated inputs and expected plans.
- `backend/tests/data/residents.json` — isolated fixture source.
- `backend/app/engine/executor.py`, `backend/app/engine/responder.py`, `backend/app/data/resident_view.py` — behavior exercised by result and completeness tests; modify only if a defect is found.
- New benchmark ground-truth fixture or verifier, location to be selected before implementation.

**Regression risks:**

- Hand-authored ground truth can itself be wrong; use an independent review/verification process and version it with the dataset.
- A benchmark fixture may not capture production joins, missing values, or MongoDB date behavior.
- A test MongoDB environment must be isolated and seeded, with no production credentials/data.
- Larger benchmark sets increase runtime; split fast regression checks from full model evaluation.

**Tests required:**

- Unit tests for semantic plan scoring, including exact projections and entity references.
- Tests that fail if a group count is correct but its resident members or requested fields are wrong.
- Tests for zero-count groups, over-limit groups, completeness flags, and pagination.
- Independent expected-result checks on a fixed dataset.
- MongoDB integration tests on an isolated test collection, with expected results calculated independently of the application query builder/executor.
- Baseline benchmark report for deterministic routing and Qwen3-4B, including answer accuracy, safe rejection rate, latency, and timeout/failure rates.
- Before/after comparison report for every Stage 0 metric, with grouped-row/projection checks and timeout rates included.

### Stage 6 — Add normalized failure attribution

**Objective:** Separate model interpretation failures from system, validation, timeout, execution, and response failures.

**Current behavior:** Numbered pipeline steps, timings, model-call counts, validation results, and some problem strings are already logged. There is no single stable per-request failure category covering all stages; execution exceptions are primarily visible by exception type.

**Proposed behavior:**

- Define stable failure-stage and failure-code values, including:
  - model interpretation/invalid plan;
  - missing or invented requirements;
  - plan/schema validation;
  - entity resolution/grounding;
  - model timeout, overload, connection, and output errors;
  - database execution;
  - response generation.
- Attach the category to query trace/API diagnostics without logging sensitive names or raw prompt contents.
- Preserve existing detailed step timings and model attempt counts. Do not collapse transport failures into interpretation errors.
- Ensure successful safe clarifications are distinguishable from infrastructure failures.

**Relevant files:**

- `backend/app/engine/steps.py` — stage/failure code definitions and trace recording.
- `backend/app/engine/service.py` — classify pipeline outcomes and attach request-level attribution.
- `backend/app/nlu/qwen_translator.py` — retain precise transport/output error categories.
- `backend/app/api/routes.py` and response types only if an API field is needed; avoid unrelated API changes.
- `backend/tests/test_step_logging.py`, `backend/tests/test_model_timeout.py`, `backend/tests/test_service.py` — attribution and privacy.

**Regression risks:**

- Logging raw validation feedback could disclose resident/staff names or values.
- New category fields can break consumers if added incompatibly or if old fields change meaning.
- Misclassification can make model quality appear worse or infrastructure reliability appear better than it is.

**Tests required:**

- One test per failure category, including timeout versus semantic rejection versus execution exception.
- Tests that categorized diagnostics agree with the actual step and model-call counters.
- Redaction tests for names, identifiers, raw prompts, and condition values.
- API compatibility tests for existing diagnostic fields.

## Acceptance criteria

The previously approved reliability implementation is present, but this proposed redesign is not complete unless:

- A compositional `RequestIR` preserves boolean conditions, comparisons, negation, grouping, aggregation, projection, entities, relationships, and conversation scope with source evidence and explicit uncertainty statuses.
- The lexical coverage audit is no longer the completeness authority; a correct plan is not rejected solely because an English connective or other surface token is absent from a vocabulary.
- Model-produced IR is treated as a candidate, not an oracle. Backend checks independently validate evidence alignment, schema/value grounding, entity authorization, expression invariants, supported semantics, and lossless IR-to-QueryPlan compilation.
- Detectable missing, extra, inconsistent, lossy, ungrounded, or fabricated semantics prevent data execution. Semantic omissions inside a valid-looking Qwen AST cannot be claimed as mechanically eliminated; their frequency and impact must be measured independently and any unacceptable residual risk must trigger reconsideration of a separately trained verifier.
- Legitimate internal execution fields are permitted but never mistaken for requested output or exposed to users.
- Tenant isolation is explicitly verified across resident queries, reference resolution, caches, sessions, and execution; if the boundary is absent or unverified, real resident data is not exposed in multi-tenant use.
- Retries preserve all requirements, repair only validated errors, and are fully revalidated.
- The one-retry limit is retained initially, with single-call, retry, concurrency, timeout, and cancellation measurements captured.
- The deterministic fast path remains enabled and uses the same final validation gate.
- The reported grouped query returns both requested groups, correct counts, resident names, and room numbers—or explicitly reports incomplete details without implying completeness.
- The reviewed `either Active or Away` query compiles to same-field OR semantics plus the strict admission-date boundary and is not rejected because of the word “either”.
- Benchmark scoring covers the complete plan and grouped resident output using independently verified expected results.
- Independently reviewed semantic fixtures and minimal pairs measure false acceptance, false rejection, ambiguity handling, and safe rejection; expected labels are not generated solely by Qwen or the production executor.
- Prompt/completion tokens, one-call and end-to-end latency, p50/p95, timeout rates, retry latency, and concurrent-request behavior are measured against the current-worktree baseline; historical or incomplete Qwen results are not presented as comparable before/after evidence.
- The new redesign is compared with the separate current-worktree baseline captured immediately before its implementation. The older pre-change deterministic reports remain historical references, and the incomplete Qwen baseline remains unavailable rather than being replaced by the deterministic run or smoke test.
- Failure reporting separates interpretation, validation, model transport/timeout, execution, and response-generation failures.
- Qwen3-4B remains the baseline; no new model, third-party package, MCP dependency, unrelated behavior, or unapproved `QueryPlan` redesign is introduced.
- Focused tests, backend tests, and applicable frontend/type checks pass for the files changed.

## Implementation status and remaining limits

- The earlier reliability work was implemented before this redesign. The user subsequently approved the compositional `RequestIR` implementation.
- The approved redesign is now implemented in the current worktree as a prototype: Qwen returns one candidate `RequestIR`; the service compiles it deterministically through the existing `QueryPlan`; deterministic, cached, retry, and model-produced plans use the same validation path. The old `requirements.py` module remains for compatibility/tests, but its lexical coverage audit is no longer the service completeness gate.
- `RequestIR` has required status and evidence on its action, predicate/boolean/relationship, grouping, projection, aggregate, sort, lookup, and limit nodes, plus ambiguous/unsupported/unresolved diagnostic lists. Backend allowlists and compiler checks remain authoritative; relationship traversal and lossy boolean structures are rejected. Status labels and evidence do not prove semantic correctness.
- Independently authored compiler fixtures include the reported date/status breakdown and minimal pairs for OR versus AND, strict versus inclusive boundaries, negation, and grouping versus filtering. These fixtures measure supported semantics, not general Qwen accuracy.
- The complete backend test suite passes; backend-wide Ruff and Python compilation pass. The Qwen3-4B five-question `gold` smoke run also completed, but it is not a semantic benchmark.
- Same-dataset smoke comparison (n=5, model-only, same Qwen3-4B configuration): exact-plan/answer accuracy was 80% before and after, with one timeout in each run. After-run mean latency was 39.6s (p50 41.4s, p95 50.9s), versus 26.7s (p50 15.4s, p95 29.3s) before. The after-run used 6,274 input and 250 completion tokens versus 6,794 input and 68 completion tokens before. Warm-up timed out at 120s in both runs. These five simple questions do not establish compositional-query accuracy or a statistically meaningful trend.
- The reported grouped query itself did not produce an IR in three direct local-model attempts: two calls hit the configured 60s timeout and one diagnostic call hit 120s. It was never executed. This is an outstanding latency/semantic-validation blocker; do not treat the prototype as production-ready for this query until Qwen completes reliably within the configured deadline and the returned IR is checked against the independent fixture.
- The pre-change deterministic reports are preserved in the session files as `baseline_deterministic_gold.json`, `baseline_deterministic_eval50.json`, and `baseline_deterministic_para100.json`. The comparable pre-change Qwen3-4B benchmark remains unavailable: repeated warm-up and inference timeouts prevented a full result. The five-question Qwen smoke baseline and after-run are limited evidence only; reports are `baseline_request_ir_qwen.json` and `after_request_ir_qwen.json`. Do not claim general Qwen before/after accuracy improvement from these samples.
- Tenant isolation remains unverified because a trusted request-to-tenant identity and repository enforcement boundary have not been established. Do not treat the application as verified for multi-tenant real-resident-data exposure until the required security checks pass.
- Full retry, concurrency, cancellation, and compositional-Qwen accuracy measurements remain outstanding. The after-smoke records basic single-call latency/timeout behavior but not those other paths.

## Approval status

The user approved implementation of the RequestIR redesign. Qwen3-4B, the deterministic fast path, the existing `QueryPlan` schema, and the one-retry limit remain in scope. No second model, parser package, external service, MCP dependency, or question-specific handler was introduced. Any QueryPlan/executor schema expansion remains outside this approval and requires a separately justified change.
