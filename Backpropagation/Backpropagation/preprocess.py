import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('./data/assignment2data.csv')
    qualities = df['quality']

    qualities.hist()