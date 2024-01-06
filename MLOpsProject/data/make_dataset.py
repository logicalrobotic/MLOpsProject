import pandas as pd
from os.path import dirname as up

# Convert data from string/boolean to integer:
def convert_data(df):
    df['map'] = df['map'].map({'de_dust2': 0, 'de_mirage': 1, 'de_nuke': 2, 'de_inferno': 3, 'de_overpass': 4, 'de_vertigo': 5, 'de_train': 6, 'de_cache': 7})
    df['round_winner'] = df['round_winner'].map({'CT': 0, 'T': 1})
    df['bomb_planted'] = df['bomb_planted'].astype(int)
    return df

if __name__ == '__main__':
    # Get the data and process it
    two_up = up(up(up(__file__)))
    # replace '\\' with '/' for Windows
    two_up = two_up.replace('\\', '/')

    # Join the paths to the csv files:
    filename = two_up + "/data/raw/csgo_round_snapshots.csv"
    output_filename = two_up + "/data/processed/csgo_converted.csv"

    print("Reading csv file from: ", filename)

    # Read the csv file:
    df = pd.read_csv(filename)

    # Convert data from string/boolean to integer:
    df = convert_data(df)

    # Save the new dataframe to a csv file:
    df.to_csv(output_filename)

    print("Saved converted csv file to: ", output_filename)