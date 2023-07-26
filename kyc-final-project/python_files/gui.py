##################################################
#                Import module                   #
##################################################

import streamlit as st

class GUI:
    def __init__(self):
        st.image("../other_images/datalabs_logo.png",use_column_width=True)
        st.title("Know Your Customer Internship Final Project")

    def make_center_button(self, button_text):
        col1, col2, col3 , col4, col5 = st.columns(5)
        with col1: pass
        with col2: pass
        with col4: pass
        with col5: pass
        with col3 : return st.button(button_text)

    # Method to show error message
    def show_error(self,message): return st.error(message)

    # Method to show warning message
    def show_warning(self,message): return st.warning(message)

    # Method to show success message
    def show_success(self,message): return st.success(message)

    def show_error_in_face_verification(self, is_first):
        message = "Cannot find face in First Image. Please Upload Another Image With A Clear Face!" if is_first else "Cannot find face in Second Image. Please Upload Another Image With A Clear Face!"
        self.show_error(message)

