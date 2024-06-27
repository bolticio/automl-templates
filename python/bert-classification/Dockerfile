FROM amd64/python:3.9.17-slim-bullseye

# Set the working directory
WORKDIR /app

RUN python3 -m venv venv

# Activate the virtual environment
ENV PATH="/app/venv/bin:$PATH"


# Copy all files to the working directory
COPY . /app

# Install requirements
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Run the Python file
CMD ["python3", "model.py"]
