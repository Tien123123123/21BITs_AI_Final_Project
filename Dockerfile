# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Update system and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir --no-deps -r requirements.txt || true && \
    apt-get remove -y gcc && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . /app

# Set Flask environment variables
ENV FLASK_APP=Flask.py  
ENV FLASK_RUN_HOST=0.0.0.0

# Expose port 5000
EXPOSE 5000

# Run Flask app
CMD ["flask", "run"]
