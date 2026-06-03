# Fixture Policy

**Version**: 1.0  
**Scope**: All validation tests (import/load/infer/contract)  
**Purpose**: Define minimal test fixtures and their usage

---

## Important Note

This document defines **fixture requirements and conventions**.

**Actual test execution** must follow:
- `model.spec.yaml.entrypoints.*_test` definitions
- Tool-specific fixture paths when contract requires
- Shared fixtures only when task semantics allow

Do not treat this document as prescribing fixed file paths for all tools.

---

## General Rules

### Fixtures Are Not Arbitrary Audio

A fixture **is not an arbitrary audio file**; it must satisfy:
- **Task-specific**: audio content must match the task semantics (ASR uses speech, VAD uses audio containing speech)
- **Format-compliant**: sample rate, channel count, and encoding format match the model's expectations
- **Deterministic**: the same input produces a predictable output

**Prohibited behaviors**:
- ❌ Using randomly generated noise as a fixture
- ❌ Using audio unrelated to the task (e.g., testing ASR with music)
- ❌ Using a language/format the model does not support

### Fixture Selection Priority

1. **Task-specific fixture** (preferred)
   - Dedicated fixtures under `tests/fixtures/{task_type}/`
   - For example, ASR models prefer `shared/asr/en_16k_10s.wav`

2. **Shared fallback** (explicitly recorded)
   - When a task-specific fixture is unavailable, use a general fixture under `shared/`
   - Must be recorded in `validation.log`: `"fixture_source": "shared"`

3. **Tool-specific** (when the contract requires it)
   - If `io_contract` specifies a particular format/language, a corresponding fixture must be created

### Handling Fixture Mismatch

A fixture mismatch **should not be immediately judged an integration failure**:
- First try using the shared fallback
- If the fallback is usable, record it as `"fixture_source": "shared"`
- Only when no fixture is available, mark it as `"fixture_missing"` and request a waiver

### Semantic Narrowing Rules

- If the runtime identity has already been resolved by evidence to a narrower target (e.g., a multilingual input resolved to an English-only checkpoint), the fixture may narrow accordingly
- If the runtime identity has not changed, the fixture fallback must preserve the original task semantics and must not implicitly narrow the validation target via the fallback
- Any semantic narrowing must be recorded in `model.spec.yaml` and `spec_validation.json`

---

### 1. Fixtures Must Be Short and Stable

- **Duration**: 3–10 seconds preferred
- **Max**: 30 seconds (avoid long files slowing iteration)
- **Stability**: Same input always produces same output (deterministic)
- **Versioning**: Fixtures tracked in git with version tags

### 2. Prefer Shared Fixtures When Possible

Use `tests/fixtures/shared/` when:
- Task type is standard (ASR, SD, SER)
- Input format is common (16kHz mono wav)
- Tool contract doesn't require domain-specific content

### 3. Tool-Specific Fixtures Allowed

Create `tests/fixtures/{tool_name}/` when:
- Tool requires specific language/domain (e.g., medical ASR)
- Tool has unique input format (e.g., 8kHz, stereo)
- Tool contract specifies specialized test cases

### 4. Fixture Location Convention

```
tests/fixtures/
├── shared/                    # Cross-tool fixtures
│   ├── asr/
│   │   ├── en_16khz_5s.wav
│   │   └── zh_16khz_5s.wav
│   ├── sd/
│   │   └── 4speakers_10s.wav
│   └── ser/
│       └── neutral_3s.wav
├── librispeech/              # Dataset-specific fixtures
│   └── sample_1.wav
└── {tool_name}/              # Tool-specific fixtures
    └── domain_specific.wav
```

---

## Per-Task Fixture Guidelines

### ASR (Automatic Speech Recognition)

**Input Requirements**:
- Format: WAV preferred (PCM)
- Sample rate: 16kHz (standard), 8kHz (telephony models)
- Channels: Mono
- Duration: 3–10 seconds
- Content: Clear speech, minimal noise

**Language Coverage**:
- English: `shared/asr/en_16khz_5s.wav`
- Chinese: `shared/asr/zh_16khz_5s.wav`
- Multilingual models: Test all claimed languages

**Output Validation**:
- Non-empty transcription
- Reasonable character count (not gibberish)

### SD (Speaker Diarization)

**Input Requirements**:
- Duration: 10–30 seconds (need multiple speaker turns)
- Speakers: 2–4 distinct speakers
- Content: Conversational speech

**Output Validation**:
- Speaker count matches
- Segment boundaries reasonable

### SA-ASR (Speaker-Aware ASR)

**Input Requirements**:
- Combine ASR + SD requirements
- Multi-speaker audio with transcription need

**Output Validation**:
- Transcription per speaker
- Speaker labels consistent

### SER (Speech Emotion Recognition)

**Input Requirements**:
- Duration: 2–5 seconds
- Content: Emotionally expressive speech
- Categories: Neutral, Happy, Angry, Sad (minimum)

**Output Validation**:
- Emotion label in expected set
- Confidence score present

### S2TT (Speech-to-Text Translation)

**Input Requirements**:
- Source language: Tool-specific
- Duration: 5–10 seconds
- Content: Translatable speech

**Output Validation**:
- Translation in target language
- Not just transcription

### SLU (Spoken Language Understanding)

**Input Requirements**:
- Content: Commands or queries with intent
- Duration: 3–8 seconds

**Output Validation**:
- Intent classification
- Slot values extracted

### VAD (Voice Activity Detection)

**Input Requirements**:
- Duration: 5–10 seconds
- Content: Mixed speech and silence

**Output Validation**:
- Speech segments detected
- Timestamps reasonable

### API-Only Tools

**Cost Consideration**:
- Use shortest viable audio (< 5 seconds)
- Cache responses when possible
- Document API costs in fixture metadata

**Offline Fallback**:
- If API unavailable, skip infer test
- Mark as "api_unavailable" in validation.log

---

## Fixture Metadata

Each fixture should have accompanying metadata:

```json
{
  "fixture_id": "asr_en_16khz_5s_v1",
  "path": "tests/fixtures/shared/asr/en_16khz_5s.wav",
  "format": "wav",
  "sample_rate": 16000,
  "channels": 1,
  "duration_sec": 5.0,
  "language": "en",
  "content_summary": "Clean English speech, minimal background noise",
  "ground_truth": "The quick brown fox jumps over the lazy dog",
  "source": "LibriSpeech dev-clean",
  "license": "CC BY 4.0"
}
```

---

## Fixture Selection for Validation

### Import Test
- **Fixture**: None (pure Python import)
- **Validation**: `from model import X` succeeds

### Load Test
- **Fixture**: None or minimal (model instantiation)
- **Validation**: Model object creates, weights load

### Infer Test
- **Fixture**: Use `model.spec.yaml` fixture or shared fixture
- **Validation**: Forward pass completes, output returned

### Contract Test
- **Fixture**: Same as Infer Test
- **Validation**: Output format matches `io_contract` in spec

---

## Fallback Strategy

If tool-specific fixture not available:

1. Check `tests/fixtures/shared/{task_type}/`
2. Check `tests/fixtures/{dataset_name}/` (if dataset known)
3. Use first available audio file in tool repo's examples/
4. Generate synthetic fixture (if generator available)
5. Skip infer/contract test with "fixture_missing" waiver

---

## Integration with Validation

The runtime validation gates (see `spec_validation.md`) reference fixtures by placeholder paths like `"path/to/tool_specific_fixture"`.

**Actual implementation** must:
- Resolve fixture path from `model.spec.yaml`
- Fall back to shared fixtures per this policy
- Record which fixture was used in `validation.log`

Example validation.log entry:
```json
{
  "test": "infer",
  "fixture_used": "tests/fixtures/shared/asr/en_16khz_5s.wav",
  "fixture_source": "shared",
  "result": "pass"
}
```
