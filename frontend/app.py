import streamlit as st
import requests
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Chest X-Ray Classification",
    page_icon="🩻",
    layout="centered"
)


# =========================================================
# BACKEND URL
# =========================================================

API_URL = "http://127.0.0.1:8000/predict"


# =========================================================
# TITLE
# =========================================================

st.title("🩻 Chest X-Ray Classification")

st.write(
    "Upload a chest X-ray image to classify it as "
    "**NORMAL** or **PNEUMONIA**."
)

# =========================================================
# IMAGE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# DISPLAY IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded X-Ray")

    st.image(
        image,
        caption="Chest X-Ray",
        width=550
    )


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button(
        "🔍 Predict",
        type="primary",
        width="stretch"
    ):

        with st.spinner(
            "Analyzing X-Ray..."
        ):

            try:

                # -----------------------------------------
                # Reset file pointer
                # -----------------------------------------

                uploaded_file.seek(0)


                # -----------------------------------------
                # Prepare file for API
                # -----------------------------------------

                files = {

                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type
                    )

                }


                # -----------------------------------------
                # Send request to FastAPI
                # -----------------------------------------

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=60
                )


                # -----------------------------------------
                # Check response
                # -----------------------------------------

                if response.status_code == 200:

                    result = response.json()


                    prediction = result[
                        "prediction"
                    ]

                    confidence = result[
                        "confidence"
                    ]

                    probabilities = result[
                        "probabilities"
                    ]


                    # =====================================
                    # RESULT
                    # =====================================

                    st.divider()

                    st.subheader(
                        "Prediction Result"
                    )


                    # -------------------------------------
                    # Prediction
                    # -------------------------------------

                    if prediction == "PNEUMONIA":

                        st.error(
                            f"Prediction: {prediction}"
                        )

                    else:

                        st.success(
                            f"Prediction: {prediction}"
                        )


                    # -------------------------------------
                    # Confidence
                    # -------------------------------------

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )


                    # =====================================
                    # PROBABILITIES
                    # =====================================

                    st.subheader(
                        "Class Probabilities"
                    )


                    col1, col2 = st.columns(2)


                    with col1:

                        st.metric(
                            "NORMAL",
                            f"{probabilities['NORMAL']:.2f}%"
                        )


                    with col2:

                        st.metric(
                            "PNEUMONIA",
                            f"{probabilities['PNEUMONIA']:.2f}%"
                        )


                    # -------------------------------------
                    # Progress bars
                    # -------------------------------------

                    st.write("NORMAL")

                    st.progress(
                        int(
                            probabilities["NORMAL"]
                        )
                    )


                    st.write("PNEUMONIA")

                    st.progress(
                        int(
                            probabilities["PNEUMONIA"]
                        )
                    )


                else:

                    st.error(
                        f"Backend error: "
                        f"{response.status_code}"
                    )

                    st.write(
                        response.text
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the FastAPI backend."
                )

                st.write(
                    "Make sure the backend is running:"
                )

                st.code(
                    "uvicorn backend.main:app --reload"
                )


            except requests.exceptions.Timeout:

                st.error(
                    "❌ The prediction request timed out."
                )


            except Exception as e:

                st.error(
                    f"❌ Something went wrong: {str(e)}"
                )