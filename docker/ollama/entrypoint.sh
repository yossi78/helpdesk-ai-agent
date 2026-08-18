#!/bin/sh
set -eu

# Official image entrypoint is `ollama`; we start the server, pull the
# model on first boot (no-op if already present), then stay in the
# foreground so Docker can manage the process.
ollama serve &
pid=$!

echo "Waiting for Ollama API..."
retries=0
until ollama list >/dev/null 2>&1; do
  retries=$((retries + 1))
  if [ "$retries" -ge 60 ]; then
    echo "Ollama failed to start" >&2
    exit 1
  fi
  sleep 1
done

echo "Ensuring model ${OLLAMA_MODEL} is available..."
ollama pull "${OLLAMA_MODEL}"

wait "$pid"
