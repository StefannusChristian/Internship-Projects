import cv2
import json
import re
import pytesseract
import streamlit as st
import os
import pandas as pd
import numpy as np
from regex_patterns_file import regex_patterns
from form import KTPInformation
from test import Test
class KTPOCR():
    def __init__(self):
        self.pytesseract_path = r"C:\Users\chris\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        self.base_path = "./dataset/"
        self.file_path = None
        self.image = None
        self.original_image = None
        self.gray = None
        self.threshold_values = {
            'default': 127,
            'nik':110
        }
        self.informations = ["default","nik"]
        self.result = KTPInformation()
        self.image_name = None
        self.patterns_to_found = {
            "rtrw": ["RTRW", "RT/RW"],
            "kota_kabupaten": ["KOTA","KABUPATEN"],
            "tempat": ["Tempat","Tempa"]
        }
        self.regex_patterns = regex_patterns
        self.image_name_for_test = None
        self.tester = None
        self.choice = self.showSelectBox()
        self.ocr_threshold = st.slider(label='OCR Threshold', min_value=0.0, max_value=1.0, step=0.05, value=0.8)

    def process(self, information: str):
        pytesseract.pytesseract.tesseract_cmd = self.pytesseract_path
        if information == "nik": th, threshed = cv2.threshold(self.gray, self.threshold_values['nik'], 255, cv2.THRESH_TRUNC)
        elif information == "default": th, threshed = cv2.threshold(self.gray, self.threshold_values['default'], 255, cv2.THRESH_TRUNC)

        raw_extracted_text = pytesseract.image_to_string((threshed), lang="ind")

        return self.clean_special_characters_text(raw_extracted_text)

    def get_ktp_image_name(self, is_test: bool):
        full_file_name = os.path.split(self.file_path)[-1]
        if is_test: return full_file_name.split('_')[1].split('.')[0]
        else: return full_file_name

    def clean_special_characters_text(self, raw_text: str):
        things_to_clean = {
            "--": "",
            "—": "",
            "“": "",
            "~": "",
            "'": "",
            "+": "",
            "[": "",
            "\\": "",
            "@": "",
            "^": "",
            "{": "",
            "%": "",
            "(": "",
            '"': "",
            "*": "",
            "|": "",
            ",": "",
            "&": "",
            "<": "",
            "`": "",
            "}": "",
            "_": "",
            "=": "",
            "]": "",
            "!": "",
            ">": "",
            ";": "",
            "#": "",
            "$": "",
            ")": "",
        }

        return self.replace_letter_from_word_dict(raw_text, things_to_clean)

    def clean_semicolons_and_stripes(self, text: str):
        things_to_clean = {
            ":": "",
            "-": "",
        }

        return self.replace_letter_from_word_dict(text, things_to_clean)

    def replace_letter_from_word_dict(self, word: str, d: dict):
        res = ""
        for letter in word:
            if letter in d: res += d[letter]
            else: res += letter
        return res.strip()

    def word_to_number_converter(self, word):
        word_dict = {
            '|' : "1"
        }
        return self.replace_letter_from_word_dict(word, word_dict)

    def clean_nik(self, word):
        missed_extracted_letters = {
            'b' : "6",
            'e' : "2",
            '?':'7',
        }

        word = self.replace_letter_from_word_dict(word, missed_extracted_letters)
        word = re.sub(r"[^\d]", "", word)
        word = word.replace(" ","")
        return word.strip()

    def remove_dots_from_string(self, text: str):
        text = text.replace(".","")
        return text.strip()

    def extract(self, extracted_result:str, information:str):
        if information == "default":
            st.success(extracted_result)
            for word in extracted_result.split("\n"):
                # EXTRACT PROVINSI
                if "PROVINSI" in word: self.extract_provinsi(word)

                # EXTRACT KOTA KABUPATEN NON JAKARTA
                if any(keyword in word for keyword in self.patterns_to_found["kota_kabupaten"]): self.extract_kota_kabupaten(word,False)

                # EXTRACT KOTA KABUPATEN JAKARTA
                if "JAKARTA" in word: self.extract_kota_kabupaten(word,True)

                # EXTRACT NAMA
                if "Nama" in word: self.extract_name(word)

                # EXTRACT
                if any(keyword in word for keyword in self.patterns_to_found["tempat"]): self.extract_tempat_tanggal_lahir(word)

                if 'Darah' in word: self.extract_golongan_darah(word)

                if 'Alamat' in word: self.extract_alamat(word)

                if 'NO.' in word:
                    self.result.alamat = self.result.alamat + ' '+word

                if "Kecamatan" in word:
                    try: self.result.kecamatan = word.split(':')[1].strip()
                    except: self.result.kecamatan = None

                if "Desa" in word: self.extract_kelurahan_atau_desa(word)

                if 'Kewarganegaraan' in word:
                    try: self.result.kewarganegaraan = word.split(':')[1].strip()
                    except: self.result.kewarganegaraan = None

                if 'Pekerjaan' in word: self.extract_pekerjaan(word)

                if 'Agama' in word: self.extract_agama(word)

                if 'Status Perkawinan' in word:
                    try:
                        self.result.status_perkawinan = re.search("BELUM KAWIN|KAWIN|CERAI HIDUP|CERAI MATI",word)[0]
                    except:
                        self.result.status_perkawinan = None

                if any(keyword in word for keyword in self.patterns_to_found["rtrw"]): self.extract_rt_rw(word)

                if 'Berlaku' in word:
                    word = word.split(":")
                    try:
                        berlaku_hingga = word[1]
                        self.result.berlaku_hingga = berlaku_hingga.strip()
                    except: self.result.berlaku_hingga = None

                if 'SEUMUR' in word: self.result.berlaku_hingga = "SEUMUR HIDUP"

        elif information == "nik":
            # st.warning(extracted_result)
            for word in extracted_result.split("\n"):
                if "NIK" in word:
                    word = word.split(':')
                    self.result.nik = self.clean_nik(word[-1].replace(" ", ""))
                    break

    def extract_kelurahan_atau_desa(self, word):
        if ":" in word: word = word.split(':')
        elif "-" in word: word = word.split('-')
        else: word = word.split()
        try:
            kelurahan_atau_desa = "".join(word[1:]).strip()
            self.result.kelurahan_atau_desa = kelurahan_atau_desa
        except: self.result.kelurahan_atau_desa = kelurahan_atau_desa


    def extract_pekerjaan(self, word):
        word = word.split()
        pekerjaan = []
        for wr in word:
            if not '-' in wr: pekerjaan.append(wr)
        pekerjaan = ' '.join(pekerjaan).replace('Pekerjaan', '').strip()
        pekerjaan = self.clean_semicolons_and_stripes(pekerjaan)
        self.result.pekerjaan = pekerjaan

    def extract_rt_rw(self,word: str):
        pattern = re.compile('|'.join(map(re.escape, self.patterns_to_found['rtrw'])))
        word = pattern.sub(" ", word).strip()
        digits = re.sub(r'\D', '', word)
        rt = digits[:len(digits)//2]
        rw = digits[len(digits)//2:]
        try:
            self.result.rt = rt
            self.result.rw = rw
        except:
            self.result.rt = None
            self.result.rw = None

    def extract_golongan_darah(self, word: str):
        self.result.jenis_kelamin = re.search("(LAKI-LAKI|LAKI|LELAKI|PEREMPUAN)", word)[0]
        word = word.split(':')
        try: self.result.golongan_darah = re.search("(O|A|B|AB)", word[-1])[0]
        except: self.result.golongan_darah = '-'

    def extract_alamat(self, word:str):
        alamat = self.word_to_number_converter(word).replace("Alamat","")
        alamat = self.clean_semicolons_and_stripes(alamat)
        self.result.alamat = alamat.strip()

    def extract_agama(self, word: str):
        word = word.replace("Agama","")
        agama = self.clean_semicolons_and_stripes(word).strip()
        try: self.result.agama = re.search(self.regex_patterns["agama"],agama)[0]
        except: self.result.agama = None

    def extract_provinsi(self, word: str):
        word = word.split(" ")
        provinsi = " ".join(word[1:])
        try: self.result.provinsi = re.search(self.regex_patterns["provinsi"],provinsi)[0].strip()
        except: self.result.provinsi = None

    def extract_kota_kabupaten(self, word: str, is_jakarta: bool):
        if is_jakarta:
            try: self.result.kota_atau_kabupaten = re.search(self.regex_patterns["jakarta"],word)[0].strip()
            except: self.result.kota_atau_kabupaten = None
        else:
            word = word.split(" ")
            if "KOTA" in word: index = word.index("KOTA")
            elif "KABUPATEN" in word: index = word.index("KABUPATEN")

            try: word = word[index:]
            except: pass

            kota_kabupaten = " ".join(word[1:])
            try: self.result.kota_atau_kabupaten = word[0] + " "+ re.search(self.regex_patterns["kota_kabupaten"],kota_kabupaten)[0].strip()
            except: self.result.kota_atau_kabupaten = None

    def extract_name(self, word: str):
        word = word.replace("Nama","")
        name = self.clean_semicolons_and_stripes(word).strip()
        try: self.result.nama = name
        except: self.result.nama = None

    def extract_tempat_tanggal_lahir(self, word: str):
        try:
            if ":" in word: word = word.split(':')
            elif "-" in word: word = word.split('-')
            word = word[1].split(" ")[1:]
            tempat_lahir, tanggal_lahir = word[0], word[1]
            # tanggal_lahir = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", "".join(word[1:]))[0].strip()
            self.result.tanggal_lahir = self.remove_dots_from_string(tanggal_lahir)
            self.result.tempat_lahir = self.remove_dots_from_string(tempat_lahir)

        except: self.result.tempat_lahir = None

    def master_process(self, information: str):
        raw_text = self.process(information)
        self.extract(raw_text, information)

    def to_json(self): return json.dumps(self.result.__dict__, indent=4)

    def json_to_pandas_dataframe(self, json_file: json):
        df = pd.DataFrame.from_dict(json_file, orient="index")
        df.columns = ["Value"]
        df.rename_axis("Information",inplace=True)
        return df

    def get_test_df(self): return self.tester.run()

    def make_dataframe(self):
        json_object = json.loads(self.to_json())
        df = self.json_to_pandas_dataframe(json_object)
        test_column = self.get_test_df()
        df['Should Return'] = test_column
        df['Check'] = np.where(df['Value'] == df['Should Return'], '✅', '❌')
        return df

    def verify_ocr(self, df: pd.DataFrame):
        threshold = int(self.ocr_threshold*18)
        # Count the number of ✅ and ❌
        check_counts = df['Check'].value_counts()
        num_correct = check_counts.get('✅', 0)
        if num_correct >= threshold: return True,num_correct,threshold
        else: return False,num_correct,threshold

    def showSelectBox(self):
        all_files = os.listdir(self.base_path)
        choice = st.selectbox("Select KTP File", all_files)
        return choice

    def run(self):
        if self.choice:
            self.file_path = self.base_path + self.choice
            self.image = cv2.imread(self.file_path)
            self.original_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image_name = self.get_ktp_image_name(False)
            self.image_name_for_test = self.get_ktp_image_name(True)
            self.tester = Test(self.image_name_for_test)

        st.header(self.image_name)
        st.image(self.original_image, caption=self.image_name,use_column_width=True)

        for information in self.informations: self.master_process(information)

        df = self.make_dataframe()
        is_verify,num_correct,threshold = self.verify_ocr(df)
        st.dataframe(df,use_container_width=True,height=700)

        show_ratio = f"{num_correct}/18"
        show_threshold = f" Threshold: {threshold}"

        if is_verify:
            st.success("VERIFIED!")
            st.balloons()
        else: st.error("NOT VERIFIED!")

        st.info(show_ratio+show_threshold)