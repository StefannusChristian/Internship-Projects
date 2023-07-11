class Test:
    def __init__(self, ktp_name: str):
        self.ktp_name = ktp_name
        self.test_dict = {
            "stefannus": {
                "provinsi": "DKI JAKARTA",
                "kota_atau_kabupaten": "JAKARTA BARAT",
                "nik": "3173040811010001",
                "nama": "STEFANNUS CHRISTIAN",
                "tempat_lahir": "JAKARTA",
                "tanggal_lahir": "08-11-2001",
                "jenis_kelamin": "LAKI-LAKI",
                "golongan_darah": "A",
                "alamat": "JL. JEMBATAN ITEM",
                "rt": "006",
                "rw": "007",
                "kelurahan_atau_desa": "PEKOJAN",
                "kecamatan": "TAMBORA",
                "agama": "KRISTEN",
                "status_perkawinan": "BELUM KAWIN",
                "pekerjaan": "PELAJAR/MAHASISWA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP"
                },
            "arief": {
                "provinsi":"JAWA BARAT",
                "kota_atau_kabupaten": "KOTA CIMAHI",
                "nik":"3217061804870007",
                "nama":"ARIEF WIJAYA PUTRA",
                "tempat_lahir": "BANDUNG",
                "tanggal_lahir":"18-04-1987",
                "jenis_kelamin": "LAKI-LAKI",
                "golongan_darah": "-",
                "alamat": "JL. AMIR MAHMUD GG. SIRNAGALIH NO. 62",
                "rt": "005",
                "rw": "006",
                "kelurahan_atau_desa": "CIBABAT",
                "kecamatan": "CIMAHI UTARA",
                "agama": "ISLAM",
                "status_perkawinan": "BELUM KAWIN",
                "pekerjaan": "PELAJAR/MAHASISWA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP"
            },
            "galang":{
                "provinsi":"BENGKULU",
                "kota_atau_kabupaten": "KOTA BENGKULU",
                "nik":"1771042401930002",
                "nama":"GALANG RAKA PRATAMA",
                "tempat_lahir": "BENGKULU",
                "tanggal_lahir":"24-01-1993",
                "jenis_kelamin": "LAKI-LAKI",
                "golongan_darah": "A",
                "alamat": "JLUNIB PERMA III NO 35",
                "rt": "015",
                "rw": "003",
                "kelurahan_atau_desa": "PEMATANG GUBERNUR",
                "kecamatan": "MUARA BANGKAHULU",
                "agama": "ISLAM",
                "status_perkawinan": "BELUM KAWIN",
                "pekerjaan": "PELAJAR/MAHASISWA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "24-01-2017"
            },
            "riyanto":{
                "provinsi":"DAERAH ISTIMEWA YOGYAKARTA",
                "kota_atau_kabupaten": "KABUPATEN SLEMAN",
                "nik":"3471140209790001",
                "nama":"RIYANTO. SE",
                "tempat_lahir": "GROBOGAN",
                "tanggal_lahir":"02-09-1979",
                "jenis_kelamin": "LAKI-LAKI",
                "golongan_darah": "O",
                "alamat": "PRM PURI DOMAS D-3, SEMPU",
                "rt": "001",
                "rw": "024",
                "kelurahan_atau_desa": "WEDOMARTANI",
                "kecamatan": "NGEMPLAK",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "PEDAGANG",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "02-09-2017"
            },
            "widiarso":{
                "provinsi":"JAWA BARAT",
                "kota_atau_kabupaten": "KABUPATEN BEKASI",
                "nik":"3216061812590006",
                "nama":"WIDIARSO",
                "tempat_lahir": "PEMALANG",
                "tanggal_lahir":"18-12-1959",
                "jenis_kelamin": "LAKI-LAKI",
                "golongan_darah": "O",
                "alamat": "SKU JL.SUMATRA BLOK B78/15",
                "rt": "003",
                "rw": "004",
                "kelurahan_atau_desa": "MEKARSARI",
                "kecamatan": "TAMBUN SELATAN",
                "agama": "ISLAM",
                "status_perkawinan": "KAWIN",
                "pekerjaan": "KARYAWAN SWASTA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "18-12-2018"
            },
            "ren":{
                "provinsi":"JAWA TENGAH",
                "kota_atau_kabupaten": "KOTA SEMARANG",
                "nik":"33740369040001",
                "nama":"RENATA VALENCIA",
                "tempat_lahir": "SEMARANG",
                "tanggal_lahir":"29-04-2002",
                "jenis_kelamin": "PEREMPUAN",
                "golongan_darah": "B",
                "alamat": "NGABLAK KIDUL",
                "rt": "007",
                "rw": "008",
                "kelurahan_atau_desa": "MUKTIHARJO KIDUL",
                "kecamatan": "PEDURUNGAN",
                "agama": "KRISTEN",
                "status_perkawinan": "BELUM KAWIN",
                "pekerjaan": "PEKERJAAN LAINNYA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga": "SEUMUR HIDUP"
            },
            "victor":{
                "provinsi":"SULAWESI SELATAN",
                "kota_atau_kabupaten": "KOTA MAKASSAR",
                "nik":"7371061804020001",
                "nama":"VICTOR CHENDRA",
                "tempat_lahir": "MAKASSAR",
                "tanggal_lahir":"18-04-2002",
                "jenis_kelamin": ":LAKI-LAKI",
                "golongan_darah": "-",
                "alamat": "JL. URIP SUMHARJONO.53",
                "rt": "002",
                "rw": "001",
                "kelurahan_atau_desa": "MALIMONGAN BARU",
                "kecamatan": "BONTOALA",
                "agama": "KRISTEN",
                "status_perkawinan": "BELUM KAWIN",
                "pekerjaan": "PELAJAR/MAHASISWA",
                "kewarganegaraan": "WNI",
                "berlaku_hingga":"SEUMUR HIDUP"
            },
        }

    def get_test_values(self, dictionary: dict): return dictionary.values()

    def run(self):
        if self.ktp_name == "arief": return self.get_test_values(self.test_dict['arief'])
        elif self.ktp_name == "galang": return self.get_test_values(self.test_dict['galang'])
        elif self.ktp_name == "riyanto": return self.get_test_values(self.test_dict['riyanto'])
        elif self.ktp_name == "stefannus": return self.get_test_values(self.test_dict['stefannus'])
        elif self.ktp_name == "widiarso": return self.get_test_values(self.test_dict['widiarso'])
        elif self.ktp_name == "ren": return self.get_test_values(self.test_dict['ren'])
        elif self.ktp_name == "victor": return self.get_test_values(self.test_dict['victor'])