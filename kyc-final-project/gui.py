##################################################
#                Import module                   #
##################################################

import streamlit as st
from face_verification import FaceVerifier

class GUI:
    def __init__(self):
        self.face_verifier = FaceVerifier()

    def showUploadedFile(self, key):
        uploaded_file = st.file_uploader("Choose File", type=["jpg", "png"], key=key)
        return uploaded_file

    def make_center_button(self, button_text):
        col1, col2, col3 , col4, col5 = st.columns(5)
        with col1: pass
        with col2: pass
        with col4: pass
        with col5: pass
        with col3 : return st.button(button_text)

    def run(self):
        st.title("Know Your Customer Datalabs Internship Final Project")
        column1, column2 = st.columns(2)
        with column1: image1 = self.showUploadedFile("ktp_image_to_compare")
        with column2: image2 = self.showUploadedFile("image_to_compare")

        verify_button = self.make_center_button("Verify")
        if verify_button: self.face_verifier.verify_face(image1, image2)
