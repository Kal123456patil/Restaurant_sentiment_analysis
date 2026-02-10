import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = pd.read_csv(self.file_path)
        return df


# 👇 THIS PART IS VERY IMPORTANT
if __name__ == "__main__":
    loader = DataLoader("data/restaurant_reviews.csv")
    data = loader.load_data()
    print("Data loaded successfully!")
    print(data.head())
