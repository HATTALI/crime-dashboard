# Crime Dashboard – Rihal Codestacker ML Challenge 2025

## Overview
This project was developed for the 2025 Rihal Codestacker Challenge – Machine Learning Track. The goal was to build a machine learning-powered system to help the CityX Police Department predict, visualize, and respond to crimes based on historical data and real-time PDF reports.

## Features by Level

### Level 1 – Exploratory Data Analysis
- Cleaned and analyzed crime data.
- Visualized crime trends by category, district, and time.

### Level 2 – Crime Classification & Severity Assignment
- Used TF-IDF + Logistic Regression to classify crime type (Category) from `Descript`.
- Assigned severity levels (1 to 5) based on crime category using a rule-based map.

### Level 3 – Geo-Spatial Visualization & Dashboard
- Created interactive heatmaps of crime locations using Folium.
- Dashboard includes filterable views and visual summaries of category/severity distributions.

### Level 4 – PDF Report Extraction and Real-Time Prediction
- Parsed key fields from uploaded PDF police reports (description, district, coordinates).
- Used trained model to predict crime category and severity in real-time.
- Displayed results directly in the dashboard.

### Bonus – Deployment
- Created a Dockerfile for full containerization.
- Deployed the app to Streamlit Cloud for public access.

## Known Limitations
- The Streamlit Cloud version may take longer to load due to model training and PDF extraction overhead.
- Some rare categories may return a severity of 0 (undefined), which is expected behavior.
- PDF upload is limited to one file at a time for simplicity.

## File Mapping by Challenge Level

| File/Folder                 | Purpose                                                                                                                                   |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `1_Data_Exploration.ipynb`  | Jupyter notebook for Level 1 (EDA), Level 2A (classification), Level 2B (severity)                                                        |
| `crime_dashboard.py`        | Streamlit web app for Level 3, 4A, 4B, and 4C                                                                                             |
| `predicted_data.csv`        | Output predictions from classification model used in dashboard                                                                            |
| `Competition_Dataset.csv`   | Original dataset provided for the competition                                                                                             |
| `processed_data.csv`        | Preprocessed dataset used for model training and evaluation                                                                               |
| `test_map.html`             | Sample output of the geo-spatial visualization (heatmap) generated during development                                                     |
| `Police Reports/`           | Contains sample crime reports in PDF format for testing Level 4A–4C                                                                       |
| `requirements.txt`          | List of Python dependencies                                                                                                               |
| `Dockerfile`                | Docker container setup for Bonus Deployment                                                                                               |
| `README.md`                 | Project documentation (this file)                                                                                                         |


## How to Run the Project

### Option 1: Run Locally with Docker (Recommended)
```bash
git clone https://github.com/HATTALI/crime-dashboard.git
cd crime-dashboard
docker build -t crime-dashboard .
docker run -p 8501:8501 crime-dashboard
```
Open your browser at: [http://localhost:8501](http://localhost:8501)

### Option 2: Streamlit Cloud (No Docker Required)
Visit the deployed app directly: [https://crime-dashboard-jhyngx...streamlit.app](https://crime-dashboard-jhyngx...streamlit.app)

## Requirements
Install dependencies manually if not using Docker:
```bash
pip install -r requirements.txt
```

Key Libraries:
- pandas, scikit-learn, streamlit
- folium, streamlit-folium
- fitz (PyMuPDF) for PDF parsing

## Acknowledgements
This project was submitted as part of the Machine Learning domain in the 2025 Rihal Codestacker Challenge to help support CityX law enforcement with predictive, data-driven insights.

