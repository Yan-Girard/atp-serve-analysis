import urllib.request
import os

os.makedirs("data/raw", exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"

for year in range(2010, 2025):
    url = f"{BASE_URL}/{year}.csv"
    dest = f"data/raw/{year}.csv"
    urllib.request.urlretrieve(url, dest)
    print(f"baixado: {year}.csv")

urllib.request.urlretrieve(f"{BASE_URL}/ATP_Database.csv", "data/raw/ATP_Database.csv")
print("baixado: ATP_Database.csv")