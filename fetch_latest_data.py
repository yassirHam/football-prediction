import requests
import os

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

LEAGUES = {
    'E0': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv', # Premier League
    'D1': 'https://www.football-data.co.uk/mmz4281/2425/D1.csv', # Bundesliga
    'I1': 'https://www.football-data.co.uk/mmz4281/2425/I1.csv', # Serie A
    'SP1': 'https://www.football-data.co.uk/mmz4281/2425/SP1.csv', # La Liga
    'F1': 'https://www.football-data.co.uk/mmz4281/2425/F1.csv', # Ligue 1
}

def download_data():
    print("Downloading latest data from football-data.co.uk...")
    for league, url in LEAGUES.items():
        try:
            print(f"Fetching {league} from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            
            file_path = os.path.join(DATA_DIR, f"{league}_2425.csv")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {file_path}")
        except Exception as e:
            print(f"Failed to download {league}: {e}")

if __name__ == "__main__":
    download_data()
