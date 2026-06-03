# AUDIO_AGENT Tool Onboarding Harness (Phase 1)

**Version**: v1.0  
**Goal**: Onboard speech-processing tools as **reproducible local inference units**  
**Scope**: Phase 1 focuses on environment adaptation, local format management, and minimal inference invocation  

---

## 1. Goals and Scope

> **Constitution**: All harness components and agents must comply with the high-level invariant principles defined in the project-level constitution; see [`policies/constitution.md`](./policies/constitution.md) for details.

### 1.1 Core Goals

This document defines the standard workflow for Phase 1 tool onboarding, ensuring:

- **Reproducible environment**: through standardized backend selection and dependency management
- **Reproducible inference**: through a unified validation contract and artifact persistence
- **Auditable process**: through structured verdict and artifact records

### 1.2 Scope of Support

| Type | Support Status | Notes |
|------|----------|------|
| Local tools | ✅ Primary goal | Weights downloaded and loaded locally |
| API tools | ✅ Supported | Invoked via remote service |
| Full Benchmark | ❌ Out of scope | Phase 1 does only minimal inference validation |
| Heavy auto-repair | ❌ Out of scope | Manual intervention for diagnosis on failure |

### 1.3 Phase 1 Boundaries

**In scope**:
- Environment backend selection (uv/conda/docker/api)
- runtime identity convergence (display identity / runtime load identity / local dir name)
- Minimal validation (import/load/infer/contract)
- Wrapper skeleton generation
- Artifact archival (spec/verdict/log)

**Out of scope**:
- Full multi-agent team chat
- Free-form planner agent
- Large-scale benchmark/leaderboard
- Complex auto-repair loops

---

## 2. Phase 1 Workflow Overview

### 2.1 Main State Machine

```
DISCOVER
    ↓ (collect repo information)
CLASSIFY
    ↓ (determine tool type)
PLAN
    ↓ (generate spec, select backend)
VALIDATE_SPEC
    ↓ (validate spec completeness and evidence sufficiency)
BUILD_ENV
    ↓ (build isolated environment)
FETCH_WEIGHTS
    ↓ (fetch/validate weights)
VALIDATE_IMPORT
    ↓ (validate importability)
VALIDATE_LOAD
    ↓ (validate loadability)
VALIDATE_INFER
    ↓ (validate inference capability)
VALIDATE_CONTRACT
    ↓ (validate output satisfies io_contract)
GENERATE_WRAPPER
    ↓ (generate unified wrapper)
SAVE_ARTIFACTS
    ↓ (save all artifacts)
DONE
```

### 2.2 Failure-Handling State Machine

```
FAIL
    ↓
DIAGNOSE (Evaluator Agent classifies the failure)
    ↓
REPLAN (Builder Agent proposes a fix)
    ↓
RETRY_FROM_CHECKPOINT (return to the pre-failure state)
```

**Maximum retries**: 3; exceeding this marks the run as FAILED and exits

---

## 3. Role Responsibilities

Phase 1 defines three roles with clearly delineated responsibility boundaries.

### 3.1 Harness Controller

**Nature**: Fixed-flow orchestrator, not an LLM Agent

**Responsibilities**:
- Advance state-machine execution
- Invoke shell commands and capture logs
- Save artifacts to designated paths
- Determine validation results (PASS/FAIL)
- Before BUILD_ENV, check whether runtime identity, preflight, and build plan have converged

**Does not engage in**:
- Deciding backend selection
- Diagnosing failure causes
- Generating wrapper code

### 3.2 Builder Agent

**Engagement points**:
- **PLAN**: Recommend backend selection
- **BUILD_ENV**: Provide build-strategy suggestions
- **GENERATE_WRAPPER**: Identify inference-entrypoint candidates

**Inputs**:
- repo scan results
- dependency files (requirements.txt, environment.yml, etc.)
- failure logs

**Outputs**:
- backend_choice.json
- build_plan.json
- wrapper_skeleton.py

**Constraints**:
- Provides suggestions only; does not execute directly
- Every suggestion must have a recorded rationale

### 3.3 Evaluator Agent

**Engagement points**:
- **DIAGNOSE**: Interpret test logs
- **FAIL**: Classify the failure type
- **REPLAN**: Suggest a retry strategy

**Inputs**:
- validation.log
- build.log
- failure taxonomy

**Outputs**:
- failure_classification.json
- retry_recommendation.json

**Constraints**:
- Must reference standard categories from failure_taxonomy
- Must assess retryable likelihood

### 3.4 Phase 1 Agent Boundaries

**Agents are not allowed to directly replace**:
- Main-flow state advancement
- shell execution and log persistence
- Artifact archival
- contract test determination

---

## 4. Standard Inputs and Outputs

### 4.1 Artifact Classification

The artifacts that every tool onboarding must produce are grouped into three classes by importance:

#### Required Artifacts (mandatory every time)

| Artifact | Path | Notes |
|------|------|------|
| `model.spec.yaml` | `audio_agent/tools/catalog/{tool}/model.spec.yaml` | Tool spec (optional but strongly recommended) |
| `backend_choice.json` | `audio_agent/tools/catalog/{tool}/artifacts/backend_choice.json` | Backend selection record |
| `build.log` | `audio_agent/tools/catalog/{tool}/artifacts/build.log` | Build log |
| `validation.log` | `audio_agent/tools/catalog/{tool}/artifacts/validation.log` | Validation log |
| `verdict.json` | `audio_agent/tools/catalog/{tool}/artifacts/verdict.json` | Final verdict |
| `wrapper` | `audio_agent/tools/catalog/{tool}/model.py`, `audio_agent/tools/catalog/{tool}/server.py`, `audio_agent/tools/catalog/{tool}/__init__.py` | Tool wrapper file set |
| `artifact_manifest.json` | `audio_agent/tools/catalog/{tool}/artifacts/artifact_manifest.json` | Artifact manifest |

#### Conditional Artifacts (mandatory when the condition is met)

| Artifact | Path | Condition |
|------|------|------|
| `spec_validation.json` | `artifacts/spec_validation.json` | VALIDATE_SPEC executed |
| `preflight_summary.json` | `artifacts/preflight_summary.json` | preflight executed |
| `weights_manifest.json` | `artifacts/weights_manifest.json` | weights.required == true |
| `failure_classification.json` | `artifacts/failure_classification.json` | DIAGNOSE executed |
| `retry_recommendation.json` | `artifacts/retry_recommendation.json` | REPLAN executed |
| `escalation.json` | `artifacts/escalation.json` | Manual escalation triggered |
| `patch_report.json` | `artifacts/patch_report.json` | upstream/config patch applied |
| `uv.lock` | `uv.lock` | backend == 'uv' |

#### Optional Artifacts (optional)

| Artifact | Path | Notes |
|------|------|------|
| `performance_notes.md` | `artifacts/performance_notes.md` | Performance notes |
| `wrapper_notes.md` | `artifacts/wrapper_notes.md` | wrapper implementation notes |
| `diagnostic_outputs/` | `artifacts/diagnostic_outputs/` | Additional diagnostic outputs |

### 4.2 Template Locations

```
templates/
├── model.spec.yaml          # Tool spec template
├── verdict.json             # Verdict result template
└── artifact_manifest.json   # Artifact manifest template
```

### 4.3 Sub-document Index

| Topic | Document Path |
|------|----------|
| UV environment strategy | `playbooks/env_uv.md` |
| Failure taxonomy (incl. API tools) | `playbooks/failure_taxonomy.md` |
| Validation contract (spec + runtime gates) | `contracts/spec_validation.md` |

---

## 5. Backend Routing Rules

Phase 1 uses rule-based backend selection; every selection must record a rationale.

### 5.1 Decision Rules

```
1. If it is an API-only tool
   → api backend
   
2. If the repo has a Dockerfile and complex dependencies
   → docker backend
   
3. If the repo has a clear environment.yml / conda signal
   → conda backend
   
4. If the repo has only pyproject.toml / requirements.txt and is mostly pure Python
   → uv backend
   
5. If it involves CUDA compilation, custom C++/k2, or complex submodules
   → prefer docker backend
   
6. If the host-machine pollution risk is high
   → prefer docker backend

7. If the phase-1 goal is a Python-only minimal callable path and a lightweight backend suffices
   → do not abandon uv/conda merely because of a preferred_backend or requires_gpu hint
```

### 5.2 Recording Requirements

Every backend selection must produce a `backend_choice.json`:

```json
{
  "chosen_backend": "uv",
  "reason": "pure python dependencies, no cuda compilation needed",
  "evidence": ["pyproject.toml present", "no Dockerfile", "no conda env"],
  "rejected_options": [
    {"backend": "docker", "reason": "overkill for simple model"}
  ]
}
```

---

## 6. Success Criteria

The **minimum criteria** for a successful Phase 1 onboarding:

| Validation Item | Criterion | Artifact |
|--------|------|------|
| Reproducible environment | Can rebuild after deleting .venv | build.log |
| Tool can import | `from model import X` or `import package` raises no error | validation.log |
| Tool can load | Model object can be instantiated and weights loaded | validation.log |
| Minimal inference runs end-to-end | Produces a result for a given test sample | validation.log |
| Output satisfies the contract | Correct type, non-empty, required fields present | validation.log |
| Artifacts saved | spec/verdict/log/wrapper all exist | artifact_manifest.json |

---

## 7. State Definitions

### 7.1 DISCOVER

**Input**: repo URL / local repo, initial tool information  
**Actions**: Scan the repo file structure, collecting README, requirements, environment.yml, etc.; converge the runtime identity (display identity / runtime load identity / local dir name)  
**Output**: repo_summary.json

**Follow-up**: May run the [preflight checklist](./playbooks/preflight_checklist.md) to generate `preflight_summary.json`
  - host preflight: GPU/driver, disk, docker, system tools
  - runtime preflight: package manager, Python, TMPDIR/extract risk, CUDA initialization risk

### 7.2 CLASSIFY

**Actions**: Determine the tool type (local/api), determine the task type, determine the environment complexity, and confirm the runtime family and minimal callable path  
**Output**: classification.json

### 7.3 PLAN

**Actions**:
- Builder Agent selects the backend
- Generate model.spec.yaml
- Generate build_plan.json
- Specify runtime load identity, fixture selection, CPU fallback / GPU constraints (if applicable)

**Outputs**:
- model.spec.yaml
- backend_choice.json
- build_plan.json

### 7.4 VALIDATE_SPEC

**Actions**:
- Check whether `model.spec.yaml` is complete
- Check whether key fields are backed by evidence
- Check whether `backend_choice.json` records conflicts and rationale
- Check whether `build_plan.json` is executable
- Check whether the fixture is available
- Check whether `io_contract` is sufficient to support the subsequent contract test
- Check whether the preflight results are compatible with the backend selection
- Check whether the runtime identity has converged
- Check whether the temp-directory strategy for large-weight restore/extract is explicit
- Check whether GPU risk has been recorded as a requirement, warning, or fallback plan

**Output**:
- spec_validation.json

**Failure**:
- Enter DIAGNOSE / REPLAN
- **Not allowed** to enter BUILD_ENV directly

**Reference**: [Spec Validation contract](./contracts/spec_validation.md)

### 7.5 BUILD_ENV

**Actions**: Build an isolated environment using the selected backend, setting cache/tmp/runtime paths per the build plan  
**Output**: environment ready / failure  
**Artifact**: build.log

### 7.5 FETCH_WEIGHTS

**Actions**: Fetch or validate weights, recording the path and checksum information  
**Output**: weights ready / failure

### 7.6 VALIDATE_IMPORT

**Actions**: Run the import test  
**Output**: import result  
**Failure**: Enter DIAGNOSE (python_dependency_missing)

### 7.7 VALIDATE_LOAD

**Actions**: Run the load test  
**Output**: load result  
**Failure**: Enter DIAGNOSE (missing_weights / cuda_version_mismatch / config_not_set)

### 7.8 VALIDATE_INFER

**Actions**: Run the minimal inference test  
**Output**: infer result  
**Failure**: Enter DIAGNOSE (wrong_entrypoint / runtime_backend_incompatible)

### 7.9 VALIDATE_CONTRACT

**Actions**:
- Validate whether the output satisfies `model.spec.yaml.io_contract`
- Check whether required_fields are present
- Check whether nonempty_fields are non-empty
- Check whether primary_field is valid
- Check JSON serializability (if required)

**Output**: contract validation result (recorded to `validation.log`)

**Failure**:
- Enter DIAGNOSE (wrong_entrypoint / wrapper_contract_mismatch / io_contract_incomplete)

**Notes**:
- In Phase 1, the runtime validation target is the **repo-native entrypoint / minimal callable path**
- The wrapper is generated after the contract validation passes, for integration into the Audio Agent Framework
- If the runtime path has passed, the wrapper smoke validates only the tool-local wrapper and does not require the top-level `audio_agent` package extras to be fully available

### 7.10 GENERATE_WRAPPER

**Actions**: Generate the unified wrapper skeleton and fill in the minimal invocation logic  
**Output**: wrapper file set
  - `model.py`: Core model wrapper class (recommended but not mandatory)
  - `server.py`: MCP server implementation
  - `__init__.py`: Package export declaration
  - `config.yaml`: MCP tool configuration

**Reference**: For the actual scaffold see `audio_agent/tools/catalog/_template/` (`server.py` / `config.yaml` / `__init__.py`, etc.); you can `cp -r _template <tool>` as a starting point

**Constraints**:
- The wrapper should reuse the validated repo-native path
- If the wrapper smoke runs, it should avoid being blocked by unrelated global dependencies

### 7.11 SAVE_ARTIFACTS

**Actions**: Save the spec snapshot, log, lockfile, verdict, and wrapper  
**Output**: artifact_manifest.json

### 7.12 DIAGNOSE / REPLAN

**Actions**:
- The Evaluator Agent classifies the failure using the failure taxonomy
- The Builder Agent provides retry suggestions

**Outputs**:
- failure_classification.json
- retry_recommendation.json
- Decision: RETRY_FROM_CHECKPOINT / FAIL_STOP

**Constraints**: Retries must comply with the [retry and escalation policy](./policies/retry_and_escalation.md); blind retries are prohibited

---

## 8. Document Index

### 8.1 Constitution (high-level invariant principles)

- [Project Constitution](./policies/constitution.md) - the 10 core rules all components must comply with

### 8.2 Policies (decision policies)

- [Evidence priority policy](./policies/evidence_priority.md) - basis for judgment when multiple sources conflict
- [Retry and escalation policy](./policies/retry_and_escalation.md) - failure handling and manual-intervention rules

### 8.3 Playbooks (execution handbooks)

- [Preflight checklist](./playbooks/preflight_checklist.md) - environment checks before BUILD_ENV
- [UV environment strategy](./playbooks/env_uv.md)
- [Failure taxonomy](./playbooks/failure_taxonomy.md) - includes API tool strategy

### 8.4 Contracts (validation contracts)

- [Spec Validation contract](./contracts/spec_validation.md) - spec pre-validation + runtime validation gates
- [Fixture policy](./contracts/fixture_policy.md) - test sample specification

### 8.5 Templates (template files)

- [model.spec.yaml](./templates/model.spec.yaml)
- [verdict.json](./templates/verdict.json)
- [artifact_manifest.json](./templates/artifact_manifest.json)

---

## 9. Phase 1 Constraints Restated

**Not implemented**:
- Full multi-agent team chat
- Free-form planner agent
- Large-scale benchmark/leaderboard
- Complex auto-repair loops
- An overly large spec schema

**Allowed**:
- Agent + human collaboration
- Semi-automated workflows
- Manual intervention on failure

**Required**:
- Transparent process, states, and failure points
- All artifacts reviewable
- Reproducible environment

---

## Appendix: Changelog

### v1.0 (2024-03-30)

- Adapted from the SURE-EVAL harness to the Audio Agent Framework
- Updated paths to `audio_agent/tools/catalog/`
- Clarified that `model.spec.yaml` is optional but recommended
- Clarified that `model.py` is recommended but not mandatory
- Integrated the persistent uv and model_downloader conventions
