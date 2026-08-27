# Help-desk Scenario Harness

A small framework that runs scripted agent scenarios against a mock help-desk
service: an LLM agent works through a YAML-defined scenario, calling tools
against a Flask service; an evaluator then scores the run. This is the codebase
you'll be extending.

## Your assignment

Two tasks, either order - your choice is something we'll discuss.

**1. Add an LLM provider.** A third provider class beside `MockEchoProvider`
and `OpenAICompatibleProvider`, wrapping any LLM API you like. It should plug
into the existing abstraction (nothing else should care which provider is in
use), be selectable from a scenario's `provider:` block, handle chat +
tool-calling, and ship with a scenario that exercises it.

**2. Add a tool.** A new tool the agent can call — something useful for
help-desk work (HTTP fetch, file read, calculator, your call). Register it,
exercise it in a scenario, and handle errors consistently with the existing
tools.

Estimated time to complete ~3h. Prototype quality is fine; if you cut scope, note it.

## What we'll evaluate

Use AI freely — you will anyway, and we don't want you pretending otherwise :)
**The interview, not the code, is the evaluation.** We'll ask you to defend
your decisions and read code live, so know *why* every meaningful choice was
made and what you considered instead. "I let the AI write it and didn't think
about it" is a fair answer if it's true — we'll just move on and find code you
can speak to. The more you can defend in depth, the better.

## Getting started

Python 3.10+.

```bash
pip install -e .
python run.py scenarios/happy_path.yaml
```

Output lands in `output/`: `thought_process.json` (agent reasoning),
`honeypot.jsonl` (what the service recorded), `verdict.json` (the result). Add
your own scenarios alongside `scenarios/happy_path.yaml` — the schema is
straightforward; follow that file.

## What's inside

```
harness/     agent runner: loads scenarios, talks to an LLM, runs tools,
             captures reasoning
target/      mock help-desk service: Flask app + in-memory store + audit log
evaluator/   joins harness output with the audit log into a verdict
run.py       entry point
```

~1000 lines of Python; readable in well under an hour.

## Using a real LLM (optional)

`MockEchoProvider` is the default and needs no API key. To use a real model,
set the provider in your scenario YAML and export the key:

```yaml
provider:
  type: openai_compatible
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
```

Any OpenAI-compatible endpoint works. Not required for the assignment.

`ticket_stats` uses a third provider that talks to Ollama's native `/api/chat`
API (not the OpenAI-compatible `/v1` endpoint). Start a local server, then run
the scenario:

```bash
docker compose up ollama
python run.py scenarios/ticket_stats.yaml
```

```yaml
provider:
  type: ollama
  base_url: http://127.0.0.1:11434
  model: llama3.2
```

## Submission

1. Create a **private** repo from this code; make your changes there.
2. Invite `dadia@alice.io` as a collaborator.
3. Reply on the email thread with the repo URL and any setup notes.

Smoke check we'll run first — it should still pass after your changes (if you
broke it on purpose, say so in NOTES.md):

```bash
python run.py scenarios/happy_path.yaml
```

# #########################################################################################
##           HOW TO RUN THE SERVICE
# #########################################################################################

1. Install [Docker Desktop] as follow:
https://www.docker.com/products/docker-desktop 

2. Start the local Ollama server in terminal as follow:
   docker compose up ollama


3. Run the following command:
   python -m target

4. Open postman collection to get famniliar with apis 
   /postman/helpdesk.json


5. click on ctrl+c to stop the service


6. Run all 100 unit tests I added to full coverage exist code , new tool and new LLM :
   python -m pytest

7. Run the default (regression) scenario in terminal:
   python run.py scenarios/happy_path.yaml

8. Run the scenario that uses the new tool of stats which use new LLM of Ollama :
   python run.py scenarios/ticket_stats.yaml


9. Open the presentation as follow:
   /docs/presentation.html



