##################################################
#                Import module                   #
##################################################

import streamlit as st
from face_verification import FaceVerifier

class GUI:
    def __init__(self):
        st.title("Know Your Customer Datalabs Internship Final Project")
        self.face_verifier = FaceVerifier()
        self.valid_image_extensions = ["jpg", "jpeg", "png"]
        self.checkbox = st.checkbox("Stop Camera Input")

    def make_center_button(self, button_text):
        col1, col2, col3 , col4, col5 = st.columns(5)
        with col1: pass
        with col2: pass
        with col4: pass
        with col5: pass
        with col3 : return st.button(button_text)

    def run(self):
        col1,col2 = st.columns(2)
        with col1: image1 = st.file_uploader("Upload KTP Image", type=self.valid_image_extensions, key="image_1_key")
        if not self.checkbox:
            with col2: image2 = st.camera_input("Take A Picture")
        else:
            with col2: image2 = st.file_uploader("Upload Image To Verify", type=self.valid_image_extensions, key="image_2_key")

        verify_button = self.make_center_button("Verify")
        if verify_button: self.face_verifier.verify_face(image1, image2)
