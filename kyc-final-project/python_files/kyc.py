from face_verification import FaceVerifier
import streamlit as st

if __name__ == "__main__":
    st.set_page_config(
        page_title="KYC"
    )
    face_verifier = FaceVerifier()
    face_verifier.run()