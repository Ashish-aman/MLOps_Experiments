# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy required files
COPY predict.py .
COPY lr_model.bin .

# Install dependencies
RUN pip install flask pandas scikit-learn==1.7.0

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["python", "predict.py"]
