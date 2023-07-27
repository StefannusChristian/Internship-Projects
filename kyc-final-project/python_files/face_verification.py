import streamlit as st  # Streamlit library for building interactive web applications
from mtcnn.mtcnn import MTCNN  # MTCNN for face detection
# PIL library for image processing
from PIL import ExifTags
from PIL import Image
from numpy import asarray  # numerical operations
from keras_vggface.utils import preprocess_input  # Preprocessing utility for VGGFace
# Import modules for face recognition
from scipy.spatial.distance import cosine  # SciPy for calculating cosine similarity
from keras_vggface.vggface import VGGFace  # VGGFace model for face recognition
from gui import GUI

class FaceVerifier:
    #########################################################
    #                    Constructor                         #m
    #########################################################
    def __init__(self):
        self.face_not_found_error_msg = "Face Not Detected! Please Upload Another Image With A Clear Face!"
        self.empty_image_msg = "Please Upload An Image!"
        self.empty_ktp_image_msg = "Please Upload KTP Image!"
        self.empty_image_to_verify_msg = "Please Upload Image To Verify!"
        self.empty_ktp_file = "Please Upload Your KTP As Image!"
        self.face_verification_threshold = 0.5
        self.gui = GUI()
        self.valid_image_extensions = ["jpg", "jpeg", "png"]
        self.checkbox = st.checkbox("Stop Camera Input")

        # Initialize the MTCNN face detection model
        self.face_detector = MTCNN()
        self.base_ktp_image_local_path = "database/ktp/"

    def fix_image_orientation(self, image):
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif is not None:
                for tag, value in exif.items():
                    if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                        if value == 3: image = image.rotate(180, expand=True)
                        elif value == 6: image = image.rotate(270, expand=True)
                        elif value == 8: image = image.rotate(90, expand=True)
        return image

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

    def verify_face(self, image1, image2):
        is_error, is_verify = False, False
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
                        is_verify = True
                        st.balloons()
                    else:
                        st.error("> FACE UNMATCH! (%.4f > %.1f)" % (score, self.face_verification_threshold))
                if not is_verify: st.error("OCR IS NOT RUN BECAUSE FACE IS UNMATCH!")

                with col6: st.info("> Face similarity: %.2f%%" % percent_match)

        else:
            col1,col2 = st.columns(2)
            with col1:
                if (image1 is None):
                    self.gui.show_warning(self.empty_ktp_image_msg)
            with col2:
                if (image2 is None):
                    self.gui.show_warning(self.empty_image_to_verify_msg)

            if image1 is None or image2 is None: is_error = True
        return is_error, is_verify

    def run(self):
        col1,col2 = st.columns(2)
        with col1: image1 = st.file_uploader("Upload KTP Image", type=self.valid_image_extensions, key="image_1_key")
        if not self.checkbox:
            with col2: image2 = st.camera_input("Take A Picture")
        else:
            with col2: image2 = st.file_uploader("Upload Image To Verify", type=self.valid_image_extensions, key="image_2_key")

        return image1,image2