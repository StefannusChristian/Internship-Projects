from gui import GUI
import streamlit as st

if __name__ == "__main__":
    st.set_page_config(
        page_title="KYC"
    )
    gui = GUI()
    gui.run()