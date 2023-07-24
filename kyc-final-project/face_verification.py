import streamlit as st  # Streamlit library for building interactive web applications
from mtcnn.mtcnn import MTCNN  # MTCNN for face detection
# PIL library for image processin
from PIL import ExifTags
from PIL import Image
from numpy import asarray  # numerical operations
from keras_vggface.utils import preprocess_input  # Preprocessing utility for VGGFace
# Import modules for face recognition
from scipy.spatial.distance import cosine  # SciPy for calculating cosine similarity
from keras_vggface.vggface import VGGFace  # VGGFace model for face recognition

class FaceVerifier:
    #########################################################
    #                    Constructor                         #
    #########################################################
    def __init__(self):
        self.face_not_found_error_msg = "Face Not Detected! Please Upload Another Image With A Clear Face!"
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