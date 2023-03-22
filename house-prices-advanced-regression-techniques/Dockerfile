# Base image
FROM python:3.9-slim-buster

# Copy the current directory contents (Dockerfile) into the container at /app
COPY . /usr/src/myapp

# Set the working directory to /app
WORKDIR /usr/src/myapp

RUN apt-get update && apt-get install -y libgomp1

# Install dependencies for python virtual environment (for use of py script)
RUN pip install --no-cache-dir -r requirements.txt

# Copy pod_script.py to working directory
COPY /LGBM_regressor.py /usr/src/myapp

# Create a directory for output files
RUN mkdir /output

# Set up a volume mount to map /output inside the container to ./output on the host machine
VOLUME ["/output"]

# Run the python script on container launch and direct output to stdout
CMD ["python", "LGBM_regressor.py", "1>&1"]