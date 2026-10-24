FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libosmesa6 ffmpeg git make && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /tidy
COPY requirements.lock pyproject.toml ./
RUN uv pip install --system -r requirements.lock
COPY . .
ENV MUJOCO_GL=osmesa PYTHONPATH=/tidy PY=python
CMD ["make", "reproduce-quick", "PY=python"]
