class Test:
    def __init__(self, ktp_name: str):
        self.ktp_name = ktp_name
        self.test_dict = {
            'adinata': {
                'provinsi': 'PROVINSI KALIMANTAN BARAT',
                'kota_atau_kabupaten': 'KOTA PONTIANAK',
                'nik': '6171042906910005',
                'nama': 'ADINATA',
                'tempat_lahir': 'PONTIANAK',
                'tanggal_lahir': '29-06-1991',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': '-',
                'alamat': 'JL. SELAT PANJANG',
                'rt': '004',
                'rw': '020',
                'kelurahan_atau_desa': 'SIANTAN HULU',
                'kecamatan': 'PONTIANAK UTARA',
                'agama': 'ISLAM',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'PELAJAR/MAHASISWA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '29-06-2017'
                },
            'arief': {
                'provinsi': 'PROVINSI JAWA BARAT',
                'kota_atau_kabupaten': 'KOTA CIMAHI',
                'nik': '3217061804870007',
                'nama': 'ARIEF WIJAYA PUTRA',
                'tempat_lahir': 'BANDUNG',
                'tanggal_lahir': '18-04-1987',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': '-',
                'alamat': 'JL. AMIR MAHMUD GG. SIRNAGALIH NO. 62',
                'rt': '005',
                'rw': '006',
                'kelurahan_atau_desa': 'CIBABAT',
                'kecamatan': 'CIMAHI UTARA',
                'agama': 'ISLAM',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'PELAJAR/MAHASISWA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
                },
            'benny': {
                'provinsi': 'PROVINSI DKI JAKARTA',
                'kota_atau_kabupaten': 'JAKARTA SELATAN',
                'nik': '3174071602530001',
                'nama': 'BENNY DARMAWAN USMAN',
                'tempat_lahir': 'U.PANDANG',
                'tanggal_lahir': '16-02-1953',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'O',
                'alamat': 'JL.ANTENE IV/9',
                'rt': '016',
                'rw': '002',
                'kelurahan_atau_desa': 'GANDARIA UTARA',
                'kecamatan': 'KEBAYORAN BARU',
                'agama': 'KATHOLIK',
                'status_perkawinan': 'CERAI HIDUP',
                'pekerjaan': 'PENSIUNAN',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
                },
            'billy': {
                'provinsi': 'PROVINSI DKI JAKARTA',
                'kota_atau_kabupaten': 'JAKARTA TIMUR',
                'nik': '3175070101909999',
                'nama': 'BILLY BUMBLEBEE SIFULAN',
                'tempat_lahir': 'SURABAYA',
                'tanggal_lahir': '01-01-1990',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'AB',
                'alamat': 'JL DIMANA NO 100',
                'rt': '001',
                'rw': '001',
                'kelurahan_atau_desa': 'ANTAH BERANTAH',
                'kecamatan': 'DUREN SAWIT',
                'agama': 'ISLAM',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'KARYAWAN SWASTA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
                },
            'debby': {
                'provinsi': 'PROVINSI DKI JAKARTA',
                'kota_atau_kabupaten': 'JAKARTA SELATAN',
                'nik': '3174096112900001',
                'nama': 'DEBBY ANGGRAINI',
                'tempat_lahir': 'JAKARTA',
                'tanggal_lahir': '21-12-1990',
                'jenis_kelamin': 'PEREMPUAN',
                'golongan_darah': '-',
                'alamat': 'JL KECAPI V',
                'rt': '006',
                'rw': '005',
                'kelurahan_atau_desa': 'JAGAKARSA',
                'kecamatan': 'JAGAKARSA',
                'agama': 'ISLAM',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'KARYAWAN SWASTA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '21-12-2016'
                },
            'eriko': {
                'provinsi': 'PROVINSI KALIMANTAN BARAT',
                'kota_atau_kabupaten': 'KABUPATEN MEMPAWAH',
                'nik': '6102180503820001',
                'nama': 'ERIKO',
                'tempat_lahir': 'MEMPAWAH',
                'tanggal_lahir': '05-03-1982',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'B',
                'alamat': 'JL. DJOHANSYAH BAKRI',
                'rt': '016',
                'rw': '005',
                'kelurahan_atau_desa': 'ANTIBAR',
                'kecamatan': 'MEMPAWAH TIMUR',
                'agama': 'ISLAM',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'KARYAWAN BUMN',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
                },
            'febrina':{
                'provinsi': 'PROVINSI DKI JAKARTA',
                'kota_atau_kabupaten': 'JAKARTA TIMUR',
                'nik': '3175076508940009',
                'nama': 'FEBRINA RESHITA DEVI',
                'tempat_lahir': 'JAKARTA',
                'tanggal_lahir': '25-08-1994',
                'jenis_kelamin': 'PEREMPUAN',
                'golongan_darah': '-',
                'alamat': 'JL. KELURAHAN IV',
                'rt': '011',
                'rw': '011',
                'kelurahan_atau_desa': 'DUREN SAWIT',
                'kecamatan': 'DUREN SAWIT',
                'agama': 'ISLAM',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'KARYAWAN SWASTA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
                },
            'fitrokhah':{
                'provinsi': 'PROVINSI JAWA TENGAH',
                'kota_atau_kabupaten': 'KABUPATEN WONOSOBO',
                'nik': '3307114404920004',
                'nama': 'FITROKHAH KHASANAH',
                'tempat_lahir': 'WONOSOBO',
                'tanggal_lahir': '04-04-1992',
                'jenis_kelamin': 'PEREMPUAN',
                'golongan_darah': 'B',
                'alamat': 'KALIBEBER',
                'rt': '003',
                'rw': '001',
                'kelurahan_atau_desa': 'KALIBEBER',
                'kecamatan': 'MOJOTENGAH',
                'agama': 'ISLAM',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'WIRASWASTA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
                },
            'galang': {
                'provinsi': 'PROVINSI BENGKULU',
                'kota_atau_kabupaten': 'KOTA BENGKULU',
                'nik': '1771042401930002',
                'nama': 'GALANG RAKA PRATAMA',
                'tempat_lahir': 'BENGKULU',
                'tanggal_lahir': '24-01-1993',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'A',
                'alamat': 'JLUNIB PERMA III NO 35 PERUMNAS UNIB',
                'rt': '015',
                'rw': '003',
                'kelurahan_atau_desa': 'PEMATANG GUBERNUR',
                'kecamatan': 'MUARA BANGKAHULU',
                'agama': 'ISLAM',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'PELAJAR/MAHASISWA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '24-01-2017'
                },
            'guohui': {
                'provinsi': 'PROVINSI JAWA BARAT',
                'kota_atau_kabupaten': 'KABUPATEN CIANJUR',
                'nik': '3203012503770011',
                'nama': 'GUOHUI CHEN',
                'tempat_lahir': 'FUJIAN',
                'tanggal_lahir': '25-03-1977',
                'jenis_kelamin': 'MALE',
                'golongan_darah': '-',
                'alamat': 'JL SELAMET PERUMAHAN RANCABALI NO. 40',
                'rt': '002',
                'rw': '004',
                'kelurahan_atau_desa': 'MUKA',
                'kecamatan': 'CIANJUR',
                'agama': 'CHRISTIAN',
                'status_perkawinan': 'MARRIED',
                'pekerjaan': 'OTHERS',
                'kewarganegaraan': 'CHINA',
                'berlaku_hingga': '12-12-2023'
                },
            'handoko': {
                'provinsi': 'PROVINSI KEPULAUAN RIAU',
                'kota_atau_kabupaten': 'KOTA BATAM',
                'nik': '2171101212749021',
                'nama': 'HANDOKO',
                'tempat_lahir': 'BANJARMASIN',
                'tanggal_lahir': '12-12-1974',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': '-',
                'alamat': 'GOLDEN LAND BLOK. F NO. 39',
                'rt': '002',
                'rw': '013',
                'kelurahan_atau_desa': 'TAMAN BALOI',
                'kecamatan': 'BATAM KOTA',
                'agama': 'KRISTEN',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'WIRASWASTA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
                },
            'jokowi': {
                'provinsi': 'PROVINSI DKI JAKARTA',
                'kota_atau_kabupaten': 'JAKARTA PUSAT',
                'nik': '3372052106610006',
                'nama': 'IR. JOKO WIDODO',
                'tempat_lahir': 'SURAKARTA',
                'tanggal_lahir': '21-06-1961',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'A',
                'alamat': 'JL. TAMAN SUROPATI NO. 7',
                'rt': '005',
                'rw': '005',
                'kelurahan_atau_desa': 'MENTENG',
                'kecamatan': 'MENTENG',
                'agama': 'ISLAM',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'GUBERNUR',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '21-06-2017'
                },
            'juliana': {
                'provinsi': 'PROVINSI BANTEN',
                'kota_atau_kabupaten': 'KABUPATEN TANGERANG',
                'nik': '3603284407690007',
                'nama': 'JULIANA',
                'tempat_lahir': 'JAKARTA',
                'tanggal_lahir': '04-07-1969',
                'jenis_kelamin': 'PEREMPUAN',
                'golongan_darah': 'O-',
                'alamat': 'GADING SERPONG 7B DD.7 NO. 29',
                'rt': '004',
                'rw': '003',
                'kelurahan_atau_desa': 'CURUG SANGERENG',
                'kecamatan': 'KELAPA DUA',
                'agama': 'KRISTEN',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'WIRASWASTA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '04-07-2018'
                },
            'kirill': {
                'provinsi': 'PROVINSI DKI JAKARTA',
                'kota_atau_kabupaten': 'JAKARTA PUSAT',
                'nik': '3372052106610006',
                'nama': 'IR. JOKO WIDODO',
                'tempat_lahir': 'SURAKARTA',
                'tanggal_lahir': '21-06-1961',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'A',
                'alamat': 'JL. TAMAN SUROPATI NO. 7',
                'rt': '005',
                'rw': '005',
                'kelurahan_atau_desa': 'MENTENG',
                'kecamatan': 'MENTENG',
                'agama': 'ISLAM',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'GUBERNUR',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '21-06-2017'
                },
            'mcd': {
                'provinsi': 'PROVINSI DKI JAKARTA',
                'kota_atau_kabupaten': 'JAKARTA BARAT',
                'nik': '1271032612010001',
                'nama': 'MICHAEL DAVID HANITIO',
                'tempat_lahir': 'MEDAN',
                'tanggal_lahir': '26-12-2001',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'A',
                'alamat': 'GREEN GARDEN BLOK O 1/49',
                'rt': '009',
                'rw': '010',
                'kelurahan_atau_desa': 'KEDOYA UTARA',
                'kecamatan': 'KEBON JERUK',
                'agama': 'KRISTEN',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'PELAJAR/MAHASISWA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
                },
            'mira': {
                'provinsi': 'PROVINSI DKI JAKARTA',
                'kota_atau_kabupaten': 'JAKARTA BARAT',
                'nik': '3171234567890123',
                'nama': 'MIRA SETIAWAN',
                'tempat_lahir': 'JAKARTA',
                'tanggal_lahir': '18-02-1986',
                'jenis_kelamin': 'PEREMPUAN',
                'golongan_darah': 'B',
                'alamat': 'JL. PASTI CEPAT A7/66',
                'rt': '007',
                'rw': '008',
                'kelurahan_atau_desa': 'PEGADUNGAN',
                'kecamatan': 'KALIDERES',
                'agama': 'ISLAM',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'PEGAWAI SWASTA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '22-02-2017'
                },
            'mustofa': {
                'provinsi': 'PROVINSI JAWA TIMUR',
                'kota_atau_kabupaten': 'KABUPATEN MOJOKERTO',
                'nik': '3516120910870003',
                'nama': 'MUSTOFA MAHMUD ABUBAKAR',
                'tempat_lahir': 'JOMBANG',
                'tanggal_lahir': '09-10-1987',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'O',
                'alamat': 'JL. RADEN WIJAYA NO.93',
                'rt': '003',
                'rw': '003',
                'kelurahan_atau_desa': 'TAWANGSARI',
                'kecamatan': 'TROWULAN',
                'agama': 'ISLAM',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'PELAJAR/MAHASISWA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '09-10-2017'
                },
            'ren': {
                'provinsi': 'PROVINSI JAWA TENGAH',
                'kota_atau_kabupaten': 'KOTA SEMARANG',
                'nik': '3374036904020001',
                'nama': 'RENATA VALENCIA SOETARDJO',
                'tempat_lahir': 'SEMARANG',
                'tanggal_lahir': '29-04-2002',
                'jenis_kelamin': 'PEREMPUAN',
                'golongan_darah': 'B',
                'alamat': 'NGABLAK KIDUL',
                'rt': '007',
                'rw': '008',
                'kelurahan_atau_desa': 'MUKTIHARJO KIDUL',
                'kecamatan': 'PEDURUNGAN',
                'agama': 'KRISTEN',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'PEKERJAAN LAINNYA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
                },
            'ridha': {
                'provinsi': 'PROVINSI JAWA TENGAH',
                'kota_atau_kabupaten': 'KOTA SURAKARTA',
                'nik': '3372032503850004',
                'nama': 'RIDHA TAQOBALALLAH',
                'tempat_lahir': 'SURAKARTA',
                'tanggal_lahir': '25-03-1985',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'O',
                'alamat': 'GAJAHAN',
                'rt': '002',
                'rw': '002',
                'kelurahan_atau_desa': 'GAJAHAN',
                'kecamatan': 'PASAR KLIWON',
                'agama': 'ISLAM',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'WARTAWAN',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '25-03-2017'
                },
            'riyanto': {
                'provinsi': 'PROVINSI DAERAH ISTIMEWA YOGYAKARTA',
                'kota_atau_kabupaten': 'KABUPATEN SLEMAN',
                'nik': '3471140209790001',
                'nama': 'RIYANTO. SE',
                'tempat_lahir': 'GROBOGAN',
                'tanggal_lahir': '02-09-1979',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'O',
                'alamat': 'PRM PURI DOMAS D-3, SEMPU',
                'rt': '001',
                'rw': '024',
                'kelurahan_atau_desa': 'WEDOMARTANI',
                'kecamatan': 'NGEMPLAK',
                'agama': 'ISLAM',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'PEDAGANG',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '02-09-2017'
            },
            'sri': {
                'provinsi': 'PROVINSI JAWA BARAT',
                'kota_atau_kabupaten': 'KOTA BEKASI',
                'nik': '32755117005860006',
                'nama': 'SRI SURATMI',
                'tempat_lahir': 'BOYOLALI',
                'tanggal_lahir': '30-05-1986',
                'jenis_kelamin': 'PEREMPUAN',
                'golongan_darah': 'B',
                'alamat': 'DUKUH ZAMRUD BLOK M. 11 / 07',
                'rt': '002',
                'rw': '014',
                'kelurahan_atau_desa': 'PADURENAN',
                'kecamatan': 'MUSTIKA JAYA',
                'agama': 'ISLAM',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'KARYAWAN SWASTA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '30-05-2017'
            },
            'stefannus': {
                'provinsi': 'PROVINSI DKI JAKARTA',
                'kota_atau_kabupaten': 'JAKARTA BARAT',
                'nik': '3173040811010001',
                'nama': 'STEFANNUS CHRISTIAN',
                'tempat_lahir': 'JAKARTA',
                'tanggal_lahir': '08-11-2001',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'A',
                'alamat': 'JL. JEMBATAN ITEM',
                'rt': '006',
                'rw': '007',
                'kelurahan_atau_desa': 'PEKOJAN',
                'kecamatan': 'TAMBORA',
                'agama': 'KRISTEN',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'PELAJAR/MAHASISWA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
            },
            'sulistyono': {
                'provinsi': 'PROVINSI JAWA TIMUR',
                'kota_atau_kabupaten': 'KABUPATEN KEDIRI',
                'nik': '3506042602660001',
                'nama': 'SULISTYONO',
                'tempat_lahir': 'KEDIRI',
                'tanggal_lahir': '26-02-1966',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': '-',
                'alamat': 'JL.RAYA - DSN PURWORKERTO',
                'rt': '002',
                'rw': '003',
                'kelurahan_atau_desa': 'PURWOKERTO',
                'kecamatan': 'NGADILUWIH',
                'agama': 'ISLAM',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'GURU',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '26-02-2017'
            },
            'tata': {
                'provinsi': 'PROVINSI JAWA BARAT',
                'kota_atau_kabupaten': 'KOTA BANDUNG',
                'nik': '3204096103020001',
                'nama': 'RENATA TAMARA TEGUH KARYADI',
                'tempat_lahir': 'BANDUNG',
                'tanggal_lahir': '21-03-2002',
                'jenis_kelamin': 'PEREMPUAN',
                'golongan_darah': 'O',
                'alamat': 'JL.BATU MAS III BLOG G NO.9',
                'rt': '004',
                'rw': '008',
                'kelurahan_atau_desa': 'CISEUREUH',
                'kecamatan': 'REGOL',
                'agama': 'KRISTEN',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'PELAJAR/MAHASISWA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
            },
            'victor': {
                'provinsi': 'PROVINSI SULAWESI SELATAN',
                'kota_atau_kabupaten': 'KOTA MAKASSAR',
                'nik': '7371061804020001',
                'nama': 'VICTOR CHENDRA',
                'tempat_lahir': 'MAKASSAR',
                'tanggal_lahir': '18-04-2002',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': '-',
                'alamat': 'JL.URIP SUMOHARJO NO.53',
                'rt': '002',
                'rw': '001',
                'kelurahan_atau_desa': 'MALIMONGAN BARU',
                'kecamatan': 'BONTOALA',
                'agama': 'KRISTEN',
                'status_perkawinan': 'BELUM KAWIN',
                'pekerjaan': 'PELAJAR/MAHASISWA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': 'SEUMUR HIDUP'
            },
            'widiarso': {
                'provinsi': 'PROVINSI JAWA BARAT',
                'kota_atau_kabupaten': 'KABUPATEN BEKASI',
                'nik': '3216061812590006',
                'nama': 'WIDIARSO',
                'tempat_lahir': 'PEMALANG',
                'tanggal_lahir': '18-12-1959',
                'jenis_kelamin': 'LAKI-LAKI',
                'golongan_darah': 'O',
                'alamat': 'SKU JL.SUMATRA BLOK B78/15',
                'rt': '003',
                'rw': '004',
                'kelurahan_atau_desa': 'MEKARSARI',
                'kecamatan': 'TAMBUN SELATAN',
                'agama': 'ISLAM',
                'status_perkawinan': 'KAWIN',
                'pekerjaan': 'KARYAWAN SWASTA',
                'kewarganegaraan': 'WNI',
                'berlaku_hingga': '18-12-2018'
            }
    }

    def get_test_values(self, dictionary: dict): return dictionary.values()

    def run(self): return self.get_test_values(self.test_dict[self.ktp_name])