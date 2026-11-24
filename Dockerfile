FROM python:3.13-slim
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY tests ./tests
RUN python -m pip install --no-cache-dir '.[research]'
CMD ["python", "-m", "pytest", "-q"]
