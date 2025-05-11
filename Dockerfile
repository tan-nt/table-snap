FROM python:3.9-slim

# Install Nginx
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

# Expose ports
EXPOSE 8080

# Set the working directory
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the application code
COPY . ./

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv env

# Activate the virtual environment and install dependencies
RUN . env/bin/activate && env/bin/pip install -r requirements.txt

# Set the environment variables
ENV PATH="/app/env/bin:$PATH"
ENV TZ=Asia/Ho_Chi_Minh


# Run the startup script
CMD ["./startup.sh"]
