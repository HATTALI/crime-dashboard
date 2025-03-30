# Base image with Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Streamlit uses
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "crime_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
