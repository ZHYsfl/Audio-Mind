- Version: 0.2
- Design: compact_attention_steering

## Skills

### content_asr

Use when:
- Questions ask what is being said in the audio.
- Questions ask about semantic content, keywords, or transcription.

Focus:
- Keywords and core semantics.
- Language, dialect, and code-switching.
- Unclear key segments.

Watchouts:
- Do not guess the answer solely from the question text.
- Do not mistake dialects or accents for another language.
- Do not miss code-switching points.
- Do not ignore missing key words caused by noise or overlap.

Thinking pattern:
- First grasp the core content.
- Then mark difficulties that affect understanding.
- Finally list low-confidence segments.

Avoid:
- Do not expand into scene descriptions.
- Do not assume speaker analysis by default.

Cue: Prioritize understanding the content, then point out uncertain words caused by language switching, dialects, accents, or overlapping noise.

### speaker_structure

Use when:
- Questions ask how many people are speaking.
- Questions ask who is speaking.
- Questions ask who said what.
- Questions ask about dialogue structure.

Focus:
- Number of speakers.
- Turn-taking transitions.
- Overlapping segments.
- Who spoke which utterance.

Watchouts:
- Do not mistake emotional changes for speaker changes.
- Do not ignore short overlaps.
- Do not transcribe content without assigning it to a speaker.
- Do not lose character continuity across time.

Thinking pattern:
- First count active speakers.
- Then examine transitions and overlaps.
- Finally perform content attribution.

Avoid:
- Do not guess specific identities.
- Do not provide transcripts without speaker attribution.

Cue: First determine the number of speakers and transitions, mark overlaps, and only attribute utterances you are confident about.

### event_scene

Use when:
- Questions ask what sound events occurred.
- Questions ask about environment, scene, or location.
- Questions ask what is in the background.

Focus:
- Foreground events.
- Background environmental sounds.
- Key sound source combinations.

Watchouts:
- Do not summarize the entire scene with a single dominant sound.
- Do not ignore weak background clues.
- Do not use speech content as the only basis for scene inference.
- Do not confuse similar environmental sounds.

Thinking pattern:
- First list foreground events.
- Then add background environment.
- Finally derive scene hypotheses.

Avoid:
- Do not force transcription.
- Do not over-interpret irrelevant details.

Cue: Listen to foreground events first, then background environment, and only output scene judgments with acoustic evidence.

### temporal_count

Use when:
- Questions ask about order or sequence.
- Questions ask about duration or stage changes.
- Questions ask about quantity or repetition count.

Focus:
- Event boundaries.
- Sequential relationships.
- Repetition patterns.
- Counting units.

Watchouts:
- Do not treat overlap as sequential order.
- Do not miscount repeated units.
- Do not ignore weak but critical boundary signals.
- Do not confuse frequency with object count.

Thinking pattern:
- First define the counting or ordering unit.
- Then locate boundaries.
- Finally provide order or quantity.

Avoid:
- Do not guess the answer first and then reverse-engineer the unit.
- Do not make unsupported causal inferences.

Cue: Define what you are counting or ordering, find boundaries, output order or count, and mark ambiguous segments.

### emotion_pragmatics

Use when:
- Questions ask about emotion.
- Questions ask about attitude or intent.
- Questions ask whether someone is joking, being sarcastic, serious, threatening, or apologizing.

Focus:
- Tone and intensity.
- Speech rate and pauses.
- Laughter, sighs, and hesitations.
- Conflict between literal meaning and tone.

Watchouts:
- Do not look only at literal meaning.
- Do not treat exaggerated performance as genuine emotion.
- Do not ignore dialogue context.
- Do not miss sarcasm and indirect expressions.

Thinking pattern:
- First examine prosody.
- Then check against literal content.
- Finally give the most conservative pragmatic interpretation.

Avoid:
- Do not default to word-by-word transcription.
- Do not state uncertain tones as absolute facts.

Cue: Look at prosody, rhythm, and paralinguistic signals first, then check against literal meaning. Be especially alert to sarcasm, indirect expression, and context dependence.

### music

Use when:
- Questions ask about instruments, style, rhythm, harmony, or melody.
- Questions ask about musical structure.

Focus:
- Instruments.
- Beat and rhythm.
- Sectional structure.
- Mode or harmonic clues.

Watchouts:
- Do not mistake production effects for instruments.
- Do not conclude based only on genre impressions.
- Do not ignore rhythm and structural clues.
- Do not mix vocal or lyric issues with music structure questions.

Thinking pattern:
- First identify main voices or instruments.
- Then examine rhythm and structure.
- Finally make stylistic or theoretical judgments.

Avoid:
- Do not give vague emotional descriptions.
- Do not turn background-music scene questions into music analysis questions.

Cue: Identify the main instruments and voices first, then look at rhythm and sections, and finally give stylistic or theoretical judgments.

### quality_reliability

Use when:
- Questions ask whether the audio is clear or reliable.
- The main task is obviously affected by recording conditions.

Focus:
- Noise types.
- Far-field vs near-field.
- Reverb and echo.
- Distortion, clipping, or dropped frames.

Watchouts:
- Do not attribute all comprehension difficulties to the content itself.
- Do not ignore severe local degradation.
- Do not fail to report reliability drops caused by overlap.

Thinking pattern:
- First identify the main degradation.
- Then see where it occurs.
- Finally explain which task point it affects.

Avoid:
- Do not expand into irrelevant semantic analysis.

Cue: Briefly point out the main recording difficulty and explain which type of judgment it affects.

## Modifiers

### overlap

Trigger:
- Multiple people speaking simultaneously.
- Concurrent sound sources.

Add focus:
- Overlapping segments.
- Attribution conflicts.

Add watchouts:
- Do not force concurrent events into a linear sequence.
- Do not be overconfident in overlapping segments.

Add cue: Pay special attention to overlapping segments; distinguish concurrency from sequential order.

### long_context

Trigger:
- Long audio.
- Meetings or full conversations.
- Questions about the whole rather than a local part.

Add focus:
- Cross-segment consistency.
- Global structure.

Add watchouts:
- Do not represent the whole with only a local fragment.
- Do not lose cross-segment character or topic continuity.

Add cue: Do not focus only on local parts; supplement with cross-segment consistency and global structure.

### language_mix

Trigger:
- Multiple languages.
- Dialects.
- Code-switching.

Add focus:
- Language switching points.
- Dialect or accent clues.

Add watchouts:
- Do not mistake accents for another language.
- Do not miss short code-switches.

Add cue: Especially mark language switching points and low-confidence content caused by dialects or accents.

### dialogue_context

Trigger:
- Multi-turn dialogue.
- Pragmatics, intent, or sarcasm.

Add focus:
- Relationship between preceding and following turns.
- Referents and responding targets.

Add watchouts:
- Do not understand the current utterance in isolation.
- Do not ignore rhetorical questions, sarcasm, or indirect expressions.

Add cue: Put the current utterance back into context; do not interpret it only by literal meaning.

### low_evidence

Trigger:
- Very short audio.
- Weak evidence.
- Question goes beyond what is audible.

Add focus:
- Most direct evidence.
- Missing information.

Add watchouts:
- Do not fill in a definite answer with common-sense guesses.
- Do not write guesses as observations.

Add cue: When evidence is weak, only report what can be directly heard and explicitly state missing information.

### anti_hallucination

Trigger:
- All high-risk Q&A.
- Open-ended questions.

Add focus:
- Direct audio evidence.
- Auditory clues most relevant to the question.

Add watchouts:
- Do not default to transcription.
- Do not default to speaker ID.
- Do not use question prior knowledge in place of auditory evidence.

Add cue: Listen to the evidence first, then answer; do not turn the task into transcription, speaker identification, or common-sense guessing.
