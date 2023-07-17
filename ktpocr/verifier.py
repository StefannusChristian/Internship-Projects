import pandas as pd
from jobs import avail_jobs

class Verifier:
    def __init__(self, dataframe: pd.DataFrame):
        self.unique_pronvisi_column = dataframe["Provinsi"].unique()
        self.unique_kota_kabupaten_column = dataframe["DaerahTingkatDua"].unique()
        self.unique_kecamatan_column = dataframe["Kecamatan"].unique()
        self.verifier_dict = {}
        self.jakarta_pattern = ["JAKARTA BARAT","JAKARTA PUSAT","JAKARTA TIMUR","JAKARTA SELATAN","JAKARTA UTARA"]
        self.goldar_pattern = "O|A|B|AB|-"
        self.agama_pattern = ["ISLAM","KRISTEN","KATOLIK","BUDHA","HINDU","KONGHUCHU"]
        self.gender_pattern = "LAKI-LAKI|PEREMPUAN|MALE|FEMALE|LAKILAKI|LAKI"
        self.status_perkawinan_pattern = "BELUM KAWIN|KAWIN|CERAI HIDUP|CERAI MATI|MARRIED"
        self.patterns_to_found = {
            "rtrw": ["RTRW", "RT/RW"],
            "kota_kabupaten": ["KOTA","KABUPATEN"],
            "tempat": ["Tempat","TgiLahir","TglLahir"],
            "berlaku_hingga":["SEUMUR","HIDUP","SEUMUR HIDUP"],
            "alamat":["NO","NO.","Alamat"],
            "kawin":["Status","Status Perkawinan"],
            "pekerjaan":["Pekerjaan","Peker"],
            "kewarganegaraan":["Kewarganegaraan","warganegaraan"],
        }

    def make_verifier_dict(self):
        self.verifier_dict["provinsi"] = self.unique_pronvisi_column
        self.verifier_dict["kota_kabupaten"] = self.unique_kota_kabupaten_column
        self.verifier_dict["jakarta"] = self.jakarta_pattern
        self.verifier_dict["goldar"] = self.goldar_pattern
        self.verifier_dict["agama"] = self.agama_pattern
        self.verifier_dict["gender"] = self.gender_pattern
        self.verifier_dict["jobs"] = avail_jobs
        self.verifier_dict["status_perkawinan"] = self.status_perkawinan_pattern
        return self.verifier_dict
