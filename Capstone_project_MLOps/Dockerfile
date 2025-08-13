# Base image
FROM python:3.10-slim
 
# Set work directory
WORKDIR /
 
# Copy required files
COPY predict.py .
COPY best_model_pipeline.bin .
 
# Install dependencies
RUN pip install flask mlflow pandas scikit-learn==1.6.1
 
# Expose port
EXPOSE 8000
 
# Command to run the app
CMD ["python", "predict.py"]