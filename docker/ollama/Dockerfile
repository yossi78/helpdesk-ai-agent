# Local Ollama server for the `ollama` harness provider.
# Build and run from the repo root:
#   docker compose up ollama
FROM ollama/ollama:latest

ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODEL=llama3.2

EXPOSE 11434

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
