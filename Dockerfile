# Use the official TensorFlow Docker image as a base
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port 80 to allow communication to/from the FastAPI application
EXPOSE 80

# Command to run your FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
