"""
crime_dashboard.py

This Streamlit dashboard implements the following components of the 
2025 ML Rihal Codestacker Challenge:

Level 3A – Geo-Spatial Visualization:
   - Interactive heatmaps using Folium and real crime coordinates.

Level 3B – Basic Web UI:
   - Dashboard built using Streamlit with tabs and filterable views.

Level 4B – Integration with Classifier:
   - Loads a trained ML model to predict crime category based on descriptions.
   - Maps predicted categories to severity levels.

Level 4C – Enhanced Web UI:
   - Allows users to upload PDF police reports.
   - Extracts key fields using regex (description, district, coordinates).
   - Runs real-time category and severity prediction using the trained model.
   - Displays predictions interactively in the dashboard.

Note:
This script retrains a lightweight model on already predicted data (from predicted_data.csv) 
to support real-time inference for uploaded PDF reports.

Submitted as part of the Machine Learning track of the Rihal Codestacker Challenge – CityX Crime Watch.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import fitz  # PyMuPDF
import re
import tempfile

# ==== Load processed crime data (Level 2 Output) ====
df = pd.read_csv("predicted_data.csv")
df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
df = df.dropna(subset=['Latitude', 'Longitude'])

# ==== Re-train model (Light model for dashboard predictions) ====
# This model is used for making predictions on uploaded PDF reports (Level 4C)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create and train a simple pipeline for real-time prediction
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(df['Descript'], df['Predicted_Category'])

# ==== Sidebar Filters ====
st.sidebar.header("Filters")
selected_district = st.sidebar.selectbox('Police District', df['PdDistrict'].unique())

# Filter the dataset based on selected district
filtered_df = df[df['PdDistrict'] == selected_district]

# ==== Main Layout: Tabs ====
tab1, tab2 = st.tabs(["📍 Heatmap", "📊 Charts"])

# ==== TAB 1: Interactive Crime Heatmap ====
with tab1:
    st.header("Crime Heatmap (Predicted)")
    
    # Prepare heatmap data
    heat_data = filtered_df[['Latitude', 'Longitude']].dropna().values.tolist()

    # Create Folium map centered on average location
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    # Add heatmap if data exists
    if heat_data:  # only add if there's data
        HeatMap(heat_data).add_to(m)
    else:
        st.warning("No valid coordinates to display heatmap.")

    # Display the map
    st_folium(m, width=1200, height=600)

     # Show sample of filtered data
    st.write(f"Showing data for {selected_district}")
    st.dataframe(filtered_df[['Descript', 'Predicted_Category', 'Predicted_Severity']].head())


# ==== TAB 2: Crime Charts ====
with tab2:
    st.header("Predicted Trends")

    st.subheader("Predicted Crime Categories")
    st.bar_chart(df['Predicted_Category'].value_counts())

    st.subheader("Predicted Severity Levels")
    st.bar_chart(df['Predicted_Severity'].value_counts())

    st.subheader("Original Crime Descriptions by District")
    st.bar_chart(df['PdDistrict'].value_counts())

# ==== PDF Upload and Real-time Prediction (Level 4C) ====
st.header("📂 Upload Police Crime Report (PDF)")

uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_pdf is not None:
    # Save the uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    # Function to extract fields from uploaded PDF (Level 4A logic reused)
    def extract_pdf_info(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        # Extract text fields using regular expressions
        date_time = re.search(r'Date & Time:\s+(.+)', text)
        description = re.search(r'Detailed Description:\s+(.+)', text)
        district = re.search(r'Police District:\s+(.+)', text)
        coordinates = re.search(r'Coordinates:\s+\(([-\d.]+),\s+([-\d.]+)\)', text)

        return {
            "Descript": description.group(1).strip() if description else "",
            "PdDistrict": district.group(1).strip() if district else "",
            "Latitude": float(coordinates.group(1)) if coordinates else None,
            "Longitude": float(coordinates.group(2)) if coordinates else None,
            "DateTime": date_time.group(1).strip() if date_time else ""
        }

    # Run the extractor on uploaded file
    info = extract_pdf_info(pdf_path)

    if info["Descript"]:
        # Predict category using the trained model
        predicted_category = model.predict([info["Descript"]])[0]

         # Map category to severity using predefined rule-based mapping
        severity_map = {
            "NON-CRIMINAL": 1, "SUSPICIOUS OCCURRENCE": 1, "SUSPICIOUS OCC": 1,
            "MISSING PERSON": 1, "RUNAWAY": 1, "RECOVERED VEHICLE": 1,
            "WARRANTS": 2, "OTHER OFFENSES": 2, "VANDALISM": 2, "TRESPASS": 2,
            "DISORDERLY CONDUCT": 2, "BAD CHECKS": 2,
            "LARCENY/THEFT": 3, "VEHICLE THEFT": 3, "FORGERY/COUNTERFEITING": 3,
            "DRUG/NARCOTIC": 3, "STOLEN PROPERTY": 3, "FRAUD": 3,
            "BRIBERY": 3, "EMBEZZLEMENT": 3,
            "ROBBERY": 4, "WEAPON LAWS": 4, "BURGLARY": 4, "EXTORTION": 4,
            "KIDNAPPING": 5, "ARSON": 5
        }
        predicted_severity = severity_map.get(predicted_category, 0)

        # Show results on the dashboard
        st.success("✅ Crime Report Analyzed:")
        st.markdown(f"**Description:** {info['Descript']}")
        st.markdown(f"**District:** {info['PdDistrict']}")
        st.markdown(f"**Coordinates:** ({info['Latitude']}, {info['Longitude']})")
        st.markdown(f"**Predicted Category:** `{predicted_category}`")
        st.markdown(f"**Predicted Severity:** `{predicted_severity}`")

    else:
        st.error("Could not extract description from the PDF.")
