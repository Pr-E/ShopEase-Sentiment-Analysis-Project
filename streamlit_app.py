import streamlit as st
import requests
import pandas as pd
import os

# API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# API_URL = os.getenv("API_URL", "http://backend:8000")  #3rd

API_URL = "https://shopease-backend.onrender.com"

#API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ShopEase Sentiment Dashboard", layout="wide")

st.title("ShopEase Sentiment Analysis Dashboard")
st.markdown("Analyse Customer Reviews")


# SINGLE PREDICTION

st.header("Single Review Prediction")

user_input = st.text_area("Enter customer review:", key="single_review_input")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict_sentiment", 
                    json={"text": user_input}
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("Prediction complete!")
                    st.write(f"**Sentiment:** {result['label']}")
                    st.write(f"**Confidence:** {result['confidence']:.2f}")

                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Connection error: {e}")


st.divider ()

# BATCH PREDICTION

st.header("Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload a CSV file with a 'review' column",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Preview file
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        # Validate column
        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column")
        else:
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):   # Getting API route of predict batch
                    try:
                        # Send file to FastAPI
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "text/csv"
                            )
                        }

                        response = requests.post(
                            f"{API_URL}/predict_batch",
                            files=files
                        )

                        if response.status_code == 200:
                            results = response.json()["predictions"]

                            result_df = pd.DataFrame(results)

                            st.success("Batch prediction completed!")

                            st.subheader("Results")
                            st.dataframe(result_df)

                            # Download button
                            csv = result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download Results CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )

                        else:
                            st.error(f"API Error: {response.text}")

                    except Exception as e:
                        st.error(f"Batch request failed: {e}")

    except Exception as e:
        st.error(f"Error reading file: {e}")


st.header("Model Training")

st.warning("Note: this may take some time.")

if st.button("Retrain Model", key = "retrain"):
    with st.spinner("Training model..."):
        try:
            response = requests.get(f"{API_URL}/train")

            if response.status_code == 200:
                st.success("Training triggered successfully")
            else:
                st.error("Training failed")

        except Exception as e:
            st.error(f"Error: {e}")            
