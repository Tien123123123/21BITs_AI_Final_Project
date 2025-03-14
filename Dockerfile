# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt /app/

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc && rm -rf /var/lib/apt/lists/*

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
