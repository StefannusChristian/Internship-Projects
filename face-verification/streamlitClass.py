##################################################
#                Import module                   #
##################################################

from kyc import KYC
import streamlit as st

class StreamLit:
    def __init__(self):
        self.kyc = KYC()

    def showSelectBox(self):
        choices = {
            "1":"Face Detection",
            "2": "Face Detection 2",
            "3": "Face Verification",
            "4": "Extract Text From KTP"
        }

        choice = st.selectbox("Select Option", [
            choices["1"],
            choices["2"],
            choices["3"],
            choices["4"],
        ])

        return choice,choices

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

    def main(self):
        st.title("Know Your Customer")
        choice, choices = self.showSelectBox()

        if choice == choices["1"]:
            uploaded_file = self.showUploadedFile("face_detect_1")
            detect_button_1 = self.make_center_button("Detect")
            if detect_button_1: self.kyc.draw_bounding_boxes_on_face(uploaded_file)

        elif choice == choices["2"]:
            uploaded_file = self.showUploadedFile('face_detect_2')
            detect_button_2 = self.make_center_button("Detect")
            if detect_button_2: self.kyc.detect_face(uploaded_file)

        elif choice == choices["3"]:
            column1, column2 = st.columns(2)
            with column1: image1 = self.showUploadedFile("ktp_image_to_compare")
            with column2: image2 = self.showUploadedFile("image_to_compare")

            verify_button = self.make_center_button("Verify")
            if verify_button: self.kyc.verify_face(image1, image2)

        elif choice == choices["4"]:
            uploaded_file = self.showUploadedFile("ktp_image")
            self.kyc.extract_text_from_ktp(uploaded_file)
