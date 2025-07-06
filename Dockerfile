FROM python:3.13-alpine AS builder
RUN apk add --no-cache build-base
WORKDIR /dhaka-house-rent-predictor

COPY c_implementations/ c_implementations/
RUN pip wheel --no-deps --wheel-dir /wheels c_implementations/c_impl c_implementations/c_pthread

FROM python:3.13-alpine
WORKDIR /dhaka-house-rent-predictor

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

COPY app/ app/
COPY run.py .
COPY config.py .

EXPOSE 5000

ENV MODULE="c_pthread" PYTHONUNBUFFERED=1
CMD ["python", "run.py"]
