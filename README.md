# selfbench

A model benchmark built from your own coding tasks, run on your own machine.

It turns commits in your repos into tasks, runs a coding agent on each one, and
judges the result with the tests the real commit added, which the agent never
sees. The point: an agent saying "done" is not evidence. The hidden tests are. No dependencies beyond Python 3.9+ and git.

## Status

Tested end to end with a stand-in agent (`python3 tests/test_pipeline.py`) and
run against a real headless agent CLI (Warp's `oz agent run`). With a new agent,
run `doctor` and `smoke` first.

## Results: Oz on Warp's own fixes

I ran Warp's Oz agent on 15 fixes merged into the open-source
[warpdotdev/warp](https://github.com/warpdotdev/warp) repo, once with Haiku 4.5
and once with Sonnet 5.5. Oz reported "done" in 30 of 30 runs. In 10 of them,
the tests that shipped with Warp's real fix failed. Full write-up, with every
failure read by hand: https://srikarchittemsetty.github.io/selfbench/

## Quick start

```bash
python3 -m selfbench doctor        # is the agent CLI installed and logged in?
python3 -m selfbench smoke         # one tiny task, end to end, real agent

python3 -m selfbench mine ~/code/my-repo -o tasks.json
# review tasks.json: fix prompts, setup_cmd and check_cmd
python3 -m selfbench validate tasks.json
python3 -m selfbench run --tasks tasks.json --configs configs.example.json --out bench-out
open bench-out/report.html
```

## How a task is built

A commit qualifies when it changes both test files and source files. Then:

- the workspace is the parent commit's files, with no git history
- the prompt is the commit message
- the check is the real commit's test files, copied in after the agent stops

`validate` keeps a task only if those tests fail before the real change and
pass after it.

## Task fields you will edit

- `prompt`: what the agent is asked to do
- `setup_cmd`: run in the workspace before the agent, e.g. create a venv
- `check_cmd`: the test command; `{test_files}` expands to the hidden tests
- `check_timeout_s`: default 120; a hang counts as a failure
- `timeout_s`: agent time limit, default 900

## Agent configurations

`kind: "warp"` runs `<bin> agent run --prompt ... --cwd ... --output-format ndjson`
with optional `model`, `profile` and `harness`. `kind: "command"` runs any other
headless agent from an argv template with `{prompt}`, `{cwd}` and `{task_id}`.

## What is measured

- pass or fail on the hidden tests
- wall time, tool calls, time spent inside tools, commands run and failed
- size of the agent's change, and whether it touched test files
- tokens, read from Warp's local database (including its write-ahead log)

Tokens show as n/a when a run cannot be matched; nothing is estimated.
