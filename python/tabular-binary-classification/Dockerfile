FROM amd64/python:3.9.17-slim-bullseye

WORKDIR /app

COPY inference_requirements.txt /app
COPY inference.py /app

RUN python3 -m venv venv \
    && . venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r inference_requirements.txt

ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8080

CMD ["python3", "inference.py"]
