# Dockerfile for Developer/Reviewer Convenience
# This Dockerfile is provided for developers and reviewers to easily test
# the application with the high-performance C implementations.
# It is NOT intended for production deployment.

FROM python:3.13-alpine AS builder
RUN apk add --no-cache build-base
WORKDIR /dhaka-house-rent-predictor

# Build C implementations (c_impl and c_pthread)
COPY model_impls/ model_impls/
RUN pip wheel --no-deps --wheel-dir /wheels model_impls/c_impl model_impls/c_pthread model_impls/py_impl

FROM python:3.13-alpine
WORKDIR /dhaka-house-rent-predictor

# Install pre-built C implementation wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY app/ app/
COPY run.py .
COPY config.py .

EXPOSE 5000

# Use multithreaded C implementation by default for maximum performance
ENV MODULE="c_pthread" PYTHONUNBUFFERED=1
CMD ["python", "run.py"]
