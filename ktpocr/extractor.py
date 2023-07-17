import cv2
import json
import re
import pytesseract
import streamlit as st
import os
import pandas as pd
import numpy as np
from time import sleep
from form import KTPInformation
from test import Test
from regex_maker import RegexMaker
from annotated_text import annotated_text
from streamlit_modal import Modal
from jaro import jaro_winkler_metric
from difflib import get_close_matches
from cachetools import cached, TTLCache
from timer import timeit

cache = TTLCache(maxsize=100, ttl=86400)
class KTPOCR:
    def __init__(self):
        self.page_title = "KTP INDONESIA OCR"
        self.page_icon = "./icon_image.png"
        self.set_page_title_and_icon()
        self.hide_side_menu = False
        self.hide_footer = True
        self.hide_styles(self.hide_side_menu,self.hide_footer)
        self.pytesseract_path = r"C:\Users\chris\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        self.base_path = "./dataset/"
        self.file_path = None
        self.image = None
        self.original_image = None
        self.gray = None
        self.threshold_values = {
            'nik':100,
            'default': 127
        }
        self.informations = ["nik","default"]
        self.result = KTPInformation()
        self.image_name = None
        self.image_name_for_test = None
        self.tester = None
        self.choice = self.showSelectBox()
        self.verified = None
        self.threshold_num = None
        self.ocr_threshold = st.slider(label='OCR Threshold', min_value=0.0, max_value=1.0, step=0.05, value=0.75)
        self.data_kode_wilayah_df = pd.read_excel("./data_kode_wilayah.xlsx")
        self.regex_maker = None
        self.regex_patterns = None
        self.jaro_winkler_threshold = 0.8
        self.preprocessed = False
        self.special_characters = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    def set_page_title_and_icon(self): st.set_page_config(page_title=self.page_title,page_icon=self.page_icon,layout="wide")

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

    def remove_semicolons(self, text: str):
        text = text.replace(":","")
        return text

    def replace_letter_from_word_dict(self, word, d: dict):
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

    def extract(self, extracted_result:str, information:str):
        if information == "default":
            st.success(extracted_result)
            for word in extracted_result.split("\n"):
                info = word.split(" ")[0].strip()
                if info in self.special_characters: info = word.split(" ")[1].strip()
                st.info(info)

                # EXTRACT PROVINSI
                if self.find_string_similarity(info,"PROVINSI") >= self.jaro_winkler_threshold: self.extract_provinsi(word)

                # EXTRACT KOTA KABUPATEN NON JAKARTA
                if any(self.find_string_similarity(info, keyword) >= self.jaro_winkler_threshold for keyword in ["KOTA", "KABUPATEN"]): self.extract_kota_kabupaten(word, False)

                # EXTRACT KOTA KABUPATEN JAKARTA
                if self.find_string_similarity(info,"JAKARTA") >= self.jaro_winkler_threshold: self.extract_kota_kabupaten(word,True)

                # EXTRACT NAMA
                if self.find_string_similarity(info,"Nama") >= self.jaro_winkler_threshold: self.extract_name(word)

                # EXTRACT TEMPAT TANGGAL LAHIR
                if self.find_string_similarity(info,"Tempat/TglLahir") >= self.jaro_winkler_threshold: self.extract_tempat_tanggal_lahir(word)

                # EXTRACT DARAH AND JENIS KELAMIN
                if self.find_string_similarity(info,"Jenis") >= self.jaro_winkler_threshold: self.extract_golongan_darah_and_jenis_kelamin(word)

                # EXTRACT ALAMAT
                if self.find_string_similarity(info,"Alamat") >= self.jaro_winkler_threshold: self.extract_alamat(word)

                # EXTRACT KECAMATAN
                if self.find_string_similarity(info,"Kecamatan") >= self.jaro_winkler_threshold: self.extract_kecamatan(word)

                # EXTRACT KELURAHAN ATAU DESA
                if self.find_string_similarity(info,"Kel/Desa") >= self.jaro_winkler_threshold: self.extract_kelurahan_atau_desa(word)

                # EXTRACT KEWARGANEGARAAN
                if self.find_string_similarity(info,"Kewarganegaraan") >= self.jaro_winkler_threshold: self.extract_kewarganegaraan(word)

                # EXTRACT PEKERJAAN
                if self.find_string_similarity(info,"Pekerjaan") >= self.jaro_winkler_threshold: self.extract_pekerjaan(word)

                # EXTRACT AGAMA
                if self.find_string_similarity(info,"Agama") >= self.jaro_winkler_threshold: self.extract_agama(word)

                # EXTRACT STATUS PERKAWINAN
                if self.find_string_similarity(info,"Status") >= self.jaro_winkler_threshold: self.extract_perkawinan(word)

                # EXTRACT RT/RW
                if self.find_string_similarity(info,"RT/RW") >= self.jaro_winkler_threshold: self.extract_rt_rw(word)

                # EXTRACT BERLAKU HINGGA SEUMUR HIDUP
                if self.find_string_similarity(info,"SEUMUR") >= self.jaro_winkler_threshold: self.result.BerlakuHingga = "SEUMUR HIDUP"

                # EXTRACT BERLAKU HINGGA
                if self.find_string_similarity(info,"Berlaku") >= self.jaro_winkler_threshold: self.extract_berlaku_hingga(word)

        elif information == "nik":
            for word in extracted_result.split("\n"):
                if "NIK" in word:
                    self.extract_nik(word)
                    break

    def extract_nik(self, word):
        st.error(word)
        if ":" in word: word = word.split(':')
        try:
            nik = word[-1].replace(" ", "")
            missed_extracted_letters = {
                'b' : "6",
                'e' : "2",
                '?':'7',
            }
            nik = self.replace_letter_from_word_dict(nik, missed_extracted_letters)
            nik = re.sub(r"[^\d]", "", nik)
            self.result.NIK = nik.strip()
        except: self.result.NIK = None

    def extract_alamat(self, word: str):
        try:
            alamat = self.word_to_number_converter(word).replace("Alamat", "")
            alamat = self.remove_semicolons(alamat).strip()

            alamat_parts = alamat.split("Agama", 1)
            alamat = alamat_parts[0].strip()

            if self.result.Alamat: self.result.Alamat += alamat
            else: self.result.Alamat = alamat

        except: self.result.Alamat = None

    def extract_berlaku_hingga(self, word):
        if ':' in word: word = word.split(":")
        try:
            berlaku_hingga = word[1]
            self.result.BerlakuHingga = berlaku_hingga.strip()
        except: self.result.BerlakuHingga = None

    def extract_kewarganegaraan(self, word):
        if ":" in word: word = word.split(":")
        try: self.result.Kewarganegaraan = word[1].strip()
        except: self.result.Kewarganegaraan = None

    def extract_kecamatan(self, word):
        if ":" in word: word = word.split(":")
        elif "." in word: word = word.split('.')
        try: self.result.Kecamatan = word[1].strip()
        except: self.result.Kecamatan = None

    def extract_perkawinan(self,word):
        try: self.result.StatusPerkawinan = re.search(self.regex_patterns["status_perkawinan"],word)[0]
        except:self.result.StatusPerkawinan = None

    def extract_kelurahan_atau_desa(self, word):
        if ":" in word: word = word.split(':')
        elif "-" in word: word = word.split('-')
        else: word = word.split()
        try:
            kelurahan_atau_desa = "".join(word[1:]).strip()
            kelurahan_atau_desa = self.clean_semicolons_and_stripes(kelurahan_atau_desa)
            self.result.KelurahanAtauDesa = kelurahan_atau_desa
        except: self.result.KelurahanAtauDesa = kelurahan_atau_desa

    def extract_pekerjaan(self, word):
        try:
            word = word.split()
            pekerjaan = []
            for wr in word:
                if not '-' in wr: pekerjaan.append(wr)
            pekerjaan = ' '.join(pekerjaan).replace('Pekerjaan', '').strip()
            pekerjaan = self.clean_semicolons_and_stripes(pekerjaan)
            self.result.Pekerjaan = pekerjaan
        except: self.result.Pekerjaan = None

    def remove_all_letters(self, word: str): return ''.join(filter(str.isdigit, word))

    def extract_rt_rw(self,word):
        pattern = re.compile('|'.join(map(re.escape, ["RTRW", "RT/RW"])))
        word = pattern.sub(" ", word).strip()
        digits = re.sub(r'\D', '', word)
        if len(digits) == 6:
            rt = digits[:len(digits)//2]
            rw = digits[len(digits)//2:]
        else:
            rt = digits[:3]
            rw = digits[-3:]
        try:
            self.result.RT = rt
            self.result.RW = rw
        except:
            self.result.RT = None
            self.result.RW = None

    def extract_golongan_darah_and_jenis_kelamin(self, word):
        try:
            self.result.JenisKelamin = re.search(self.regex_patterns['gender'], word)[0]
            word = word.split(':')
            self.result.GolonganDarah = re.search(self.regex_patterns['goldar'], word[-1])[0]
        except: self.result.GolonganDarah = None

    def extract_agama(self, word):
        word = word.replace("Agama","")
        agama = self.clean_semicolons_and_stripes(word).strip()
        try: self.result.Agama = re.search(self.regex_patterns["agama"],agama)[0]
        except: self.result.Agama = None

    def find_best_match_from_regex_pattern(self, pattern_list: list, word: str):
        best_match = None
        best_similarity = 0
        for pattern in pattern_list:
            similarity = self.find_string_similarity(pattern, word)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern
        return best_match, best_similarity

    def extract_provinsi(self, word):
        word = word.split(" ")
        provinsi = " ".join(word[1:])
        provinsi = self.clean_semicolons_and_stripes(provinsi)
        pattern_list = self.regex_patterns["provinsi"].split("|")
        best_match, best_similarity = self.find_best_match_from_regex_pattern(pattern_list, provinsi)

        if best_match and best_similarity >= self.jaro_winkler_threshold: self.result.Provinsi = best_match.strip()
        else: self.result.Provinsi = None

    def extract_kota_kabupaten(self, word, is_jakarta: bool):
        if is_jakarta:
            try: self.result.KotaAtauKabupaten = re.search(self.regex_patterns["jakarta"],word)[0].strip()
            except: self.result.KotaAtauKabupaten = None
        else:
            word = word.split(" ")
            if "KOTA" in word: index = word.index("KOTA")
            elif "KABUPATEN" in word: index = word.index("KABUPATEN")

            try: word = word[index:]
            except: pass

            kota_kabupaten = " ".join(word[1:])
            try: self.result.KotaAtauKabupaten = word[0] + " "+ re.search(self.regex_patterns["kota_kabupaten"],kota_kabupaten)[0].strip()
            except: self.result.KotaAtauKabupaten = None

    def extract_name(self, word):
        try:
            word = word.replace("Nama","")
            name = self.clean_semicolons_and_stripes(word).strip()
            self.result.Nama = name
        except: self.result.Nama = None

    def extract_tempat_tanggal_lahir(self, word):
        try:
            word = word.split(" ")
            tempat_lahir, tanggal_lahir = word[-2],word[-1]

            self.result.TanggalLahir = tanggal_lahir
            self.result.TempatLahir = tempat_lahir

        except: self.result.TempatLahir = None

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
        df['Value'] = df['Value'].astype(str)
        df['Should Return'] = df['Should Return'].astype(str)
        df['Check'] = np.where(df.apply(lambda row: self.find_string_similarity(row['Value'], row['Should Return']) >= self.jaro_winkler_threshold, axis=1), '✅', '❌')
        df['Similarity'] = df.apply(lambda row: f"{self.find_string_similarity(row['Value'], row['Should Return']) * 100} %", axis=1)
        return df

    def verify_ocr(self, df: pd.DataFrame):
        threshold = int(self.ocr_threshold*18)
        check_counts = df['Check'].value_counts()
        num_correct = check_counts.get('✅', 0)
        if num_correct >= threshold: return True,num_correct,threshold
        else: return False,num_correct,threshold

    def showSelectBox(self):
        all_files = os.listdir(self.base_path)
        choice = st.selectbox("Select KTP File", all_files)
        return choice

    @timeit
    def preprocess_data_kode_wilayah_df(self):
        if self.preprocessed: return
        self.data_kode_wilayah_df = self.data_kode_wilayah_df.loc[:, ~self.data_kode_wilayah_df.columns.str.contains('^Unnamed')]
        self.data_kode_wilayah_df['Kecamatan'] = self.data_kode_wilayah_df['Kecamatan'].apply(lambda x: x.upper())
        self.data_kode_wilayah_df['Provinsi'] = self.data_kode_wilayah_df['Provinsi'].apply(lambda x: x.upper())
        self.data_kode_wilayah_df['DaerahTingkatDua'] = self.data_kode_wilayah_df['DaerahTingkatDua'].apply(lambda x: x.upper())
        self.data_kode_wilayah_df['DaerahTingkatDua'] = self.data_kode_wilayah_df['DaerahTingkatDua'].apply(lambda x: " ".join(x.split()[1:]))
        self.data_kode_wilayah_df["Provinsi"] = self.data_kode_wilayah_df["Provinsi"].replace({
            'ACEH (NAD)': 'ACEH',
            'NUSA TENGGARA TIMUR (NTT)': 'NUSA TENGGARA TIMUR',
            'NUSA TENGGARA BARAT (NTB)': 'NUSA TENGGARA BARAT',
            'DI YOGYAKARTA': 'DAERAH ISTIMEWA YOGYAKARTA'
        })
        self.preprocessed = True

    def make_annotated_text(self, main_text:str,secondary_text:str,color:str): return annotated_text((main_text,secondary_text,color))

    def hide_styles(self, hide_main_menu: bool, hide_footer: bool):
        style_string = ""

        if hide_main_menu:
            style_string += """
            #MainMenu {visibility: hidden}
            """

        if hide_footer:
            style_string += """
            footer {visibility: hidden}
            """

        if style_string:
            style_html = f"<style>{style_string}</style>"
            st.markdown(style_html, unsafe_allow_html=True)

    # def show_modal(self, is_verified: bool, key: str, button):
    #     modal = Modal("",key,max_width=250,padding=0)
    #     if button:
    #         modal.open()

    #     if modal.is_open():
    #         if is_verified:
    #             with modal.container():
    #                 st.image("./verified.gif")
    #                 sleep(2.5)
    #                 modal.close()

    def find_string_similarity(self, string1: str, string2: str): return round(jaro_winkler_metric(string1,string2),2)

    @cached(cache)
    def make_regex_dict(self): return self.regex_maker.make_regex_dict()

    def run(self):
        # verify_button = st.button("Verify")

        if self.choice:
            self.file_path = self.base_path + self.choice
            self.image = cv2.imread(self.file_path)
            self.original_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image_name = self.get_ktp_image_name(False)
            self.image_name_for_test = self.get_ktp_image_name(True)
            self.tester = Test(self.image_name_for_test)

        if True:
            st.header(self.image_name)
            col1, col2 = st.columns(2)

            self.make_annotated_text("DATA","LABS","#4CB9A2")

            with col2: st.image(self.original_image, caption=self.image_name,use_column_width=True)

            self.preprocess_data_kode_wilayah_df()
            self.regex_maker = RegexMaker(self.data_kode_wilayah_df)
            self.regex_patterns = self.make_regex_dict()

            for information in self.informations: self.master_process(information)

            df = self.make_dataframe()
            self.verified,num_correct,self.threshold_num = self.verify_ocr(df)

            with col1:
                st.dataframe(df,use_container_width=True,height=700)

            show_ratio = f"{num_correct}/18"
            show_threshold = f" Threshold: {self.threshold_num}"

            if self.verified:
                st.success("VERIFIED!")

            else: st.error("NOT VERIFIED!")

            st.info(show_ratio+show_threshold)