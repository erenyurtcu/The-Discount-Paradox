
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PAGE = "https://www.chicagobooth.edu/research/kilts/research-data/dominicks"
OUT = Path("Dominicks_Dataset")
OUT.mkdir(exist_ok=True)

session = requests.Session()

session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
})

retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry)

session.mount("https://", adapter)
session.mount("http://", adapter)

html = Path("dominicks.html").read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")
files = {}

# CSV bağlantılarını bağlantı metnine göre bul
for a in soup.find_all("a", href=True):
    href = urljoin(PAGE, a["href"])
    label = a.get_text(" ", strip=True).lower()

    if re.fullmatch(r"(upc[a-z]{3}|w[a-z]{3})\.csv", label, re.I):
        files[label] = href

print("Bulunan dosyalar:")
for filename in sorted(files):
    print(filename)
# İndirilecek bağlantıları göster
print(f"{len(files)} CSV dosyası bulundu.")

def download(item):
    filename, url = item
    target = OUT / filename

    if target.exists() and target.stat().st_size > 0:
        return f"MEVCUT: {filename}"

    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return f"OK: {filename}"
    except Exception as e:
        target.unlink(missing_ok=True)
        return f"HATA: {filename}: {e}"

with ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(download, files.items()):
        print(result)

print("İndirme tamamlandı. Klasör:", OUT.resolve())
