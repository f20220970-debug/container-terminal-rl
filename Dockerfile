# Reproduce every result with one command:
#   docker build -t terminal-rl . && docker run --rm -v "$PWD/results:/app/results" terminal-rl
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas gymnasium matplotlib pytest \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
CMD ["sh", "-c", "pytest -q && python evaluate.py && python plot.py && python bench.py"]
