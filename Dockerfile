FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system build deps required for some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy application code (including the model file `randomforest_model.joblib`)
COPY . /app

# Expose port
EXPOSE 5353

# Use gunicorn for production-like server
CMD ["gunicorn", "app:APP", "-b", "0.0.0.0:5353", "--workers", "2", "--threads", "2"]
