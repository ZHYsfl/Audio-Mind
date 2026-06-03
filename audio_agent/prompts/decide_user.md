## Question
{question}

## Initial Plan
{initial_plan}

## Planner Reasoning Trace
Brief reasoning you stated alongside each prior round's tool call(s), in order:

{planner_reasoning_trace}

## Loop Budget
Step {step_count} of {max_steps}.

## Available Audio Files
{audio_list}

## Evidence Ledger
Chronological record of every evidence-producing event so far. Tool-derived
entries carry the call args and success status alongside the fuser-formatted
content.

{evidence_ledger}

## Decide
Emit BOTH of the following in this response — both are required:

1. **Message `content`** — 1–2 sentences of reasoning (why these call(s),
   what evidence is still missing or already sufficient). An empty
   `content` is a protocol violation; it makes your own Planner Reasoning
   Trace show "(no reasoning)" on the next round.
2. **Tool call(s)** for your chosen next action, following the rules
   above.
