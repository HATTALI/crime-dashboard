import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import fitz  # PyMuPDF
import re
import tempfile

# Load predicted data
df = pd.read_csv("predicted_data.csv")
df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
df = df.dropna(subset=['Latitude', 'Longitude'])

# ---- Train ML model on predicted data ----
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])
model.fit(df['Descript'], df['Predicted_Category'])

# Sidebar filters
st.sidebar.header("Filters")
selected_district = st.sidebar.selectbox('Police District', df['PdDistrict'].unique())

# Filtered data
filtered_df = df[df['PdDistrict'] == selected_district]

# TABS
tab1, tab2 = st.tabs(["📍 Heatmap", "📊 Charts"])

# ==== TAB 1: HEATMAP ====
with tab1:
    st.header("Crime Heatmap (Predicted)")
    
    # Filtered coordinates
    heat_data = filtered_df[['Latitude', 'Longitude']].dropna().values.tolist()
    
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
    
    if heat_data:  # only add if there's data
        HeatMap(heat_data).add_to(m)
    else:
        st.warning("No valid coordinates to display heatmap.")
    
    st_folium(m, width=1200, height=600)

    st.write(f"Showing data for {selected_district}")
    st.dataframe(filtered_df[['Descript', 'Predicted_Category', 'Predicted_Severity']].head())


# ==== TAB 2: CHARTS ====
with tab2:
    st.header("Predicted Trends")

    st.subheader("Predicted Crime Categories")
    st.bar_chart(df['Predicted_Category'].value_counts())

    st.subheader("Predicted Severity Levels")
    st.bar_chart(df['Predicted_Severity'].value_counts())

    st.subheader("Original Crime Descriptions by District")
    st.bar_chart(df['PdDistrict'].value_counts())

# ----- New Section: PDF Report Upload & Prediction -----
st.header("📂 Upload Police Crime Report (PDF)")

uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    # --- Extract text from PDF ---
    def extract_pdf_info(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

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

    info = extract_pdf_info(pdf_path)

    if info["Descript"]:
        # --- Predict using your model ---
        predicted_category = model.predict([info["Descript"]])[0]
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

        # --- Display results ---
        st.success("✅ Crime Report Analyzed:")
        st.markdown(f"**Description:** {info['Descript']}")
        st.markdown(f"**District:** {info['PdDistrict']}")
        st.markdown(f"**Coordinates:** ({info['Latitude']}, {info['Longitude']})")
        st.markdown(f"**Predicted Category:** `{predicted_category}`")
        st.markdown(f"**Predicted Severity:** `{predicted_severity}`")

    else:
        st.error("Could not extract description from the PDF.")
