import pandas as pd
from web_scraper import JobScraper
from cachetools import cached, TTLCache

class RegexMaker:
    def __init__(self, dataframe: pd.DataFrame):
        self.unique_pronvisi_column = dataframe["Provinsi"].unique()
        self.unique_kota_kabupaten_column = dataframe["DaerahTingkatDua"].unique()
        self.unique_kecamatan_column = dataframe["Kecamatan"].unique()
        self.regex_dict = {}
        self.jakarta_regex_pattern = "JAKARTA BARAT|JAKARTA PUSAT|JAKARTA TIMUR|JAKARTA SELATAN|JAKARTA UTARA"
        self.goldar_regex_pattern = "O|A|B|AB|-"
        self.agama_regex_pattern = "ISLAM|KRISTEN|KATOLIK|BUDHA|HINDU|KONGHUCHU"
        self.gender_regex_pattern = "LAKI-LAKI|PEREMPUAN|MALE|FEMALE|LAKILAKI|LAKI"
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
        self.web_scraper = JobScraper()

    def make_regex_pattern(self, unique_column_list: list): return "|".join(unique_column_list)

    def make_provinsi_regex(self): return self.make_regex_pattern(self.unique_pronvisi_column)

    def make_kota_kabupaten_regex(self): return self.make_regex_pattern(self.unique_kota_kabupaten_column)

    def make_job_regex(self):
        all_jobs = self.web_scraper.get_job_titles()
        all_jobs.append("PEKERJAAN LAINNYA")
        return self.make_regex_pattern(all_jobs)


    def make_regex_dict(self):
        self.regex_dict["provinsi"] = self.make_provinsi_regex()
        self.regex_dict["kota_kabupaten"] = self.make_kota_kabupaten_regex()
        self.regex_dict["jakarta"] = self.jakarta_regex_pattern
        self.regex_dict["goldar"] = self.goldar_regex_pattern
        self.regex_dict["agama"] = self.agama_regex_pattern
        self.regex_dict["gender"] = self.gender_regex_pattern
        self.regex_dict["jobs"] = self.make_job_regex()
        self.regex_dict["status_perkawinan"] = self.status_perkawinan_pattern
        return self.regex_dict
