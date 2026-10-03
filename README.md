# Noise Sensitivity of Agentic Misalignment Propensity

## What this tests
Does Best-of-N-style text noise (random capitalization, character
scrambling, character substitution — following Hughes et al.,
"Best-of-N Jailbreaking," 2024) change a model's rate of spontaneous
misalignment (blackmail, information leaking) in Anthropic's Agentic
Misalignment scenarios?

## Why
BoN-style noise reliably breaks refusal behavior when a model is
explicitly asked to do something harmful. Whether the same noise
shifts *propensity* — spontaneous behavior under pressure/opportunity,
with no explicit harmful instruction present to refuse — is untested.
This also serves as a check on whether current safety training
targeting this specific benchmark is robust to surface-level noise or
narrowly keyed to clean input.

## Base task
`agentic_misalignment` from UKGovernmentBEIS/inspect_evals, a port of
Anthropic's open-sourced Agentic Misalignment framework
(github.com/anthropic-experimental/agentic-misalignment).

## Method
- Noise applied only to natural-language portions of the prompt, not
  structural/functional content.
- Three intensity levels, anchored to fractions of Hughes et al.'s
  published augmentation probabilities.
- All other task config (scenario, goal_type, goal_value, urgency_type,
  extra_system_instructions) held fixed — only noise level varies.
- `test_eval_awareness=True` throughout, to check whether any effect
  is mediated by detected evaluation awareness.
- Nested sampling: several independent noise draws per level, each run
  across multiple epochs, separating the effect of noise level from
  the effect of one particular random draw and from ordinary model
  sampling variance.
- Primary outcome: `classifier_verdict`. Primary analysis:
  Cochran-Armitage trend test per model.

## Models
GPT-4.1 first (documented ~0.6 baseline on this eval). If an effect
appears, extending to a frontier model reported to have trained this
behavior away, to test whether noise degrades that suppression.

## Status
[in progress]

## Prior art
- Hughes et al., Best-of-N Jailbreaking (2024) — arxiv.org/abs/2412.03556
- Lynch et al., Agentic Misalignment (Anthropic, 2025) —
  anthropic.com/research/agentic-misalignment