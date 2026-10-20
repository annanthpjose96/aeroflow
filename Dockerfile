FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd -m -u 10001 aeroflow
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY sops ./sops
RUN pip install .
RUN mkdir -p /data && chown aeroflow /data
USER aeroflow
WORKDIR /data
ENV AEROFLOW_CONFIG=/app/configs/default.yaml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["sh", "-c", "aeroflow run --config ${AEROFLOW_CONFIG} --host 0.0.0.0 --port 8000"]
