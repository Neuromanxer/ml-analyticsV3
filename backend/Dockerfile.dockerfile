# Base Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . /app

# Install system build tools (for numpy, pandas, sklearn, etc.)
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command (optional — overridden by docker-compose)
CMD ["bash"]