from requests import get, Session
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class JobScraper:
    def __init__(self):
        self.url = "https://nasional.tempo.co/read/1609887/88-jenis-pekerjaan-yang-bisa-dicantumkan-di-ktp-dan-cara-mengubahnya"

    def get_job_titles(self):
        session = Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount(self.url, adapter)
        html_text = session.get(self.url).text
        soup = BeautifulSoup(html_text, 'lxml')
        p_tags = soup.find_all("p")

        pekerjaan = [p.text.split(". ", 1)[1].upper().strip() for p in p_tags
                    if p.text.startswith(tuple(f"{i}." for i in range(1, 1000)))]

        pekerjaan = [item.replace('/ ', '/') for item in pekerjaan]
        return pekerjaan