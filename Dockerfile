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
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update &&  apt-get install -y libgl1-mesa-glx && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install gdown to download files from Google Drive
RUN pip install gdown

RUN rm -rf ./model_weights

# Download the file from Google Drive using the file ID
RUN gdown https://drive.google.com/uc?id=1T-7oQDPWDM4BNa8Wl-2ThZJxptAS7ASW
# If the downloaded file is a ZIP archive, unzip it
# (You can replace "file_name.zip" with your actual downloaded file's name)
RUN unzip models.zip -d ./
RUN mv /models /model_weights


# RUN apt-get install libgl1 mesa-utils

# Create a virtual environment
RUN python -m venv env

# Activate the virtual environment and install dependencies
RUN . env/bin/activate && env/bin/pip install -r requirements.txt
RUN env/bin/pip install --upgrade onnx onnxruntime

# Set the environment variables
ENV PATH="/app/env/bin:$PATH"
ENV TZ=Asia/Ho_Chi_Minh


# Run the startup script
CMD ["./startup.sh"]
