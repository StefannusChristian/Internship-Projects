from face_verification import FaceVerifier
from ocr import KTPOCR
from gui import GUI
import streamlit as st

def run_kyc():
    gui = GUI()
    gui.show_title()
    face_verifier = FaceVerifier(False)
    ocr = KTPOCR(False)
    ktp_image, image_to_verify = face_verifier.run()
    verify_button = gui.make_center_button("VERIFY KTP!")
    if verify_button:
        is_error,is_verify = face_verifier.verify_face(ktp_image, image_to_verify)
        if not is_error and is_verify: ocr.run(ktp_image,True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="KYC",
        layout="wide",
        page_icon="../images/datalabs_logo/datalabs_logo_without_text.png"
    )
    run_kyc()