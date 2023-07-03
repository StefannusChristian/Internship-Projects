##################################################
#                Import module                   #
##################################################

# Image Processing and Manipulation Modules
import cv2  # computer vision tasks
from mtcnn.mtcnn import MTCNN  # MTCNN for face detection
from numpy import asarray  # numerical operations
from PIL import ExifTags

# Text Extraction Modules
import pytesseract  # OCR (Optical Character Recognition)
import pandas as pd  # data manipulation and analysis
import re  # pattern matching

# Import modules for visualization
import streamlit as st  # Streamlit library for building interactive web applications
from matplotlib.patches import Circle, Rectangle  # Matplotlib for plotting shapes on images
import matplotlib.pyplot as plt  # Matplotlib for visualization

# Import modules for face recognition
from scipy.spatial.distance import cosine  # SciPy for calculating cosine similarity
from keras_vggface.vggface import VGGFace  # VGGFace model for face recognition
from keras_vggface.utils import preprocess_input  # Preprocessing utility for VGGFace

# Import modules for Streamlit and PIL
import streamlit as st  # Streamlit library for building interactive web applications
from PIL import Image  # PIL library for image processing

import os


#########################################################
#             KYC (Know Your Customer) Class            #
#########################################################
class KYC:
    #########################################################
    #                    Constructor                         #
    #########################################################

    def __init__(self):
        self.face_not_found_error_msg = "Cannot Detect Faces For! Please Upload Another Image With A Clear Face!"
        self.empty_image_msg = "Please Upload An Image!"
        self.empty_image_1_msg = "Please Upload Image 1!"
        self.empty_image_2_msg = "Please Upload Image 2!"
        self.empty_ktp_file = "Please Upload Your KTP As Image!"
        self.face_verification_threshold = 0.5

        # Initialize the MTCNN face detection model
        self.face_detector = MTCNN()
        self.base_ktp_image_local_path = "database/ktp/"


    #########################################################
    #                 UI Related Methods                   #
    #########################################################

    # Method to show error message
    def show_error(self,message): return st.error(message)

    # Method to show warning message
    def show_warning(self,message): return st.warning(message)

    # Method to show success message
    def show_success(self,message): return st.success(message)

    def show_error_in_face_verification(self, is_first):
        message = "Cannot find face in First Image. Please Upload Another Image With A Clear Face!" if is_first else "Cannot find face in Second Image. Please Upload Another Image With A Clear Face!"
        self.show_error(message)

    #########################################################
    #                 Extract Face Method                   #
    #########################################################

    '''
    extracts and resizes the face region from an input image using the MTCNN (Multi-Task Cascaded Convolutional Neural Networks) face detection model
    '''

    def extract_face(self, file):
        # Convert the file to pixels (numpy array)
        pixels = asarray(file)
        # st.write(pixels)

        # Detect faces in the image using the MTCNN model
        try:
            results = self.face_detector.detect_faces(pixels)

            # Retrieve the coordinates and dimensions of the bounding box around the first detected face
            x1, y1, width, height = results[0]["box"]

            # Calculate the coordinates of the bottom-right corner of the bounding box
            x2, y2 = x1 + width, y1 + height

            # Extract the face region from the image based on the bounding box coordinates
            face = pixels[y1:y2, x1:x2]

            # Create a PIL Image object from the face region
            image = Image.fromarray(face)

            # Resize the image to 224x224 pixels
            '''
            why 224x224 px?
            1. Standardization: ensures that all images being processed have the same size

            2. Model Compatibility: Many pre-trained deep learning models have been trained on image datasets with an input size of 224x224

            3. Computational Efficiency: faster training and inference times

            4. Sufficient Information: retains sufficient information for many computer vision tasks. The downscaled image generally preserves important features and patterns necessary for tasks like face recognition, object detection, and image classification.
            '''
            image = image.resize((224, 224))

            # Convert the resized image back to a numpy array
            face_array = asarray(image)

            # Return the numpy array representing the resized face
            return face_array

        except: return None

    def draw_bounding_boxes_on_face(self, uploaded_file):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = self.fix_image_orientation(image)

            # Load the uploaded image file as a 3d numpy array
            data = asarray(image)

            # Create a new figure and axes
            fig, ax = plt.subplots()

            # Display the image on the axes
            ax.imshow(data)

            # Detect faces in the image
            faces = self.face_detector.detect_faces(data)
            if len(faces) == 0: self.show_error(self.face_not_found_error_msg)

            # Loop through each detected face
            for face in faces:
                # Extract the bounding box coordinates
                x, y, width, height = face['box']

                # Create a rectangle patch for the bounding box
                rect = Rectangle((x, y), width, height, fill=False, color='maroon')

                # Add the rectangle patch to the axes
                ax.add_patch(rect)

                # Draw keypoints on the face
                for _, value in face['keypoints'].items():
                    # Create a circle patch for each keypoint
                    dot = Circle(value, radius=2, color='maroon')

                    # Add the circle patch to the axes
                    ax.add_patch(dot)

            # Turn off the axis labels
            plt.axis("off")

            # Display the figure in the Streamlit app
            st.pyplot(fig)

        else:
            # Display a warning message if no image is uploaded
            self.show_warning(self.empty_image_msg)

    def fix_image_orientation(self, image):
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif is not None:
                for tag, value in exif.items():
                    if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                        if value == 3:
                            image = image.rotate(180, expand=True)
                        elif value == 6:
                            image = image.rotate(270, expand=True)
                        elif value == 8:
                            image = image.rotate(90, expand=True)
        return image

    def detect_face(self, uploaded_file):
        if uploaded_file is not None:
            column1, column2 = st.columns(2)
            image = Image.open(uploaded_file)
            image = self.fix_image_orientation(image)

            with column1:
                size = 450, 450
                image.thumbnail(size)
                image.save("thumb.png")
                st.image("thumb.png")
            pixels = asarray(image)
            fig, ax = plt.subplots()
            ax.imshow(pixels)

            try:
                results = self.face_detector.detect_faces(pixels)
                x1, y1, width, height = results[0]["box"]
                x2, y2 = x1 + width, y1 + height
                face = pixels[y1:y2, x1:x2]
                image = Image.fromarray(face)
                image = image.resize((224, 224))
                face_array = asarray(image)
                with column2:
                    plt.imshow(face_array)
                    plt.axis("off")
                    st.pyplot(fig)
            except: self.show_error(self.face_not_found_error_msg)
        else: self.show_warning(self.empty_image_msg)

    def verify_face(self, image1, image2):
        if (image1 is not None) and (image2 is not None):
            col1, col2 = st.columns(2)
            image1 = Image.open(image1)
            image2 = Image.open(image2)

            image1 = self.fix_image_orientation(image1)
            image2 = self.fix_image_orientation(image2)

            detected_face_1 = self.extract_face(image1)
            detected_face_2 = self.extract_face(image2)

            if detected_face_1 is None: self.show_error_in_face_verification(True)
            if detected_face_2 is None: self.show_error_in_face_verification(False)

            faces = [detected_face_1, detected_face_2]
            if all(face is not None for face in faces):
                with col1:
                    st.subheader("KTP Image")
                    st.image(image1)
                with col2:
                    st.subheader("Image To Verify")
                    st.image(image2)

                st.header("Detected Faces")
                col3, col4 = st.columns(2)
                with col3: st.image(detected_face_1)
                with col4: st.image(detected_face_2)

                samples = asarray(faces, dtype="float32")
                samples = preprocess_input(samples, version=2)
                model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg")
                embeddings = model.predict(samples)

                score = cosine(embeddings[0], embeddings[1])
                percent_match = (1 - score) * 100

                col5, col6 = st.columns(2)
                with col5:
                    if score <= self.face_verification_threshold:
                        st.success("> FACE MATCH! (%.4f <= %.1f)" % (score, self.face_verification_threshold))
                        st.balloons()
                    else:
                        st.error("> FACE UNMATCH! (%.4f > %.1f)" % (score, self.face_verification_threshold))

                with col6: st.info("> Face similarity: %.2f%%" % percent_match)

        else:
            col1,col2 = st.columns(2)
            with col1:
                if (image1 is None): self.show_warning(self.empty_image_1_msg)
            with col2:
                if (image2 is None): self.show_warning(self.empty_image_2_msg)

    def extract_text_from_ktp(self, ktp_file):
        if ktp_file is not None:
            ktp_table = self.extract_text_from_indonesian_ktp_as_pandas_dataframe(ktp_file)
            column1, column2 = st.columns(2)
            with column1:
                st.table(ktp_table)
            with column2:
                st.image(ktp_file, width=700, caption="User KTP", use_column_width=False)
        else: self.show_warning(self.empty_ktp_file)

    def extract_text_from_indonesian_ktp_as_pandas_dataframe(self, ktp_file):
        # Read Img
        image_file = self.base_ktp_image_local_path + ktp_file.name
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold For NIK
        th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)
        result = pytesseract.image_to_string((threshed), lang="ind")
        st.write(result)

        regex_patterns = {
            'NIK': r'NIK\s*:\s*(\d{16})',
            'Nama': r'Nama\s*:\s*([A-Z\s]+)',
            'Tempat/Tgl Lahir': r'Tempat/Tgi Lahir\s*:\s*([A-Z\s,]+),\s*(\d{2}-\d{2}-\d{4})',
            'Jenis Kelamin': r'Jenis kelamin\s*:\s*([A-Z\-]+)',
            'Alamat': r'Alamat\s*([A-Z0-9\s\.]+)',
            'RT/RW': r'RI/RW\s*([0-9\/]+)',
            'Kel/Desa': r'KelDesa\s*:\s*([A-Z\s]+)',
            'Kecamatan': r'Kecamatan\s*:\s*([A-Z\s]+)',
            'Agama': r'Agama\s*([A-Z\s]+)',
            'Status Perkawinan': r'Status Perkawinan\s*([A-Z\s]+)',
            'Pekerjaan': r'Pekerjaan\s*([A-Z\/]+)',
            'Kewarganegaraan': r'Kewarganegaraan\s*([A-Z\s]+)',
            'Berlaku Hingga': r'Berlaku Hingga\s*([A-Z\s]+)'
        }

        # Initialize a dictionary to store the extracted information
        extracted_info = {}

        # Extract information using regex patterns
        for key, pattern in regex_patterns.items():
            match = re.search(pattern, result)
            if match:
                extracted_info[key] = match.group(1)
            else:
                extracted_info[key] = 'Not found'

        # Create a DataFrame from the extracted information
        df = pd.DataFrame.from_dict(extracted_info, orient='index', columns=['Value'])
        df.index.rename('Information', inplace=True)

        return df
