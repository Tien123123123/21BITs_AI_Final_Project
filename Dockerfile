# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements file before installing dependencies
COPY requirements.txt /app/

# Install uv and dependencies in a single RUN command
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gcc && \
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Failed to install uv"; exit 1; } && \
    export PATH="/root/.local/bin:$PATH" && \
    which uv || { echo "uv not found in PATH"; exit 1; } && \
    uv --version && \
    uv pip install --system -r requirements.txt && \
    apt-get remove -y curl gcc && \
    rm -rf /var/lib/apt/lists/* /root/.local/bin/.tmp*

# Ensure uv is available in the final container environment
ENV PATH="/root/.local/bin:$PATH"

# Copy application files
COPY . /app

# Set Flask environment variables
ENV FLASK_APP=Flask.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Expose port 5000
EXPOSE 5000

# Create a start script file
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run both the Flask app and Kafka consumer
CMD ["bash", "-c", "python -m kafka_server.consumer & python -m flask run"]