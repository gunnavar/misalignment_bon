from inspect_ai import eval
from inspect_evals.agentic_misalignment import agentic_misalignment

eval(
    tasks=agentic_misalignment(
        grader_model="openrouter/deepseek/deepseek-v4.1-flash",
        test_eval_awareness=True,
    ), 
    model="openrouter/deepseek/deepseek-v3.2",
    epochs=10
)