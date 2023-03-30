# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/dataStrcuture/05_ts_format.ipynb.

# %% auto 0
__all__ = ['FILE_METADATA', 'create_ts_file']

# %% ../../nbs/dataStrcuture/05_ts_format.ipynb 3
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from ..config.mongo import mongo_init
from soccer_prediction_data.datastructure.data_extractor import (
    STATS,
    COMPETITION_IDS,
    data_aggregator,
)

# %% ../../nbs/dataStrcuture/05_ts_format.ipynb 5
FILE_METADATA = {
    "source": "InStat",
    "creator": "Real-Analytics",
    "data_description": """More than 5 years of football data from leagues across the world that includes stats from both competing teams in each played game in each dimension.\n#The data describes a series of numbers achieved by each squad, which indicate their athletic performance.\n#This yields a database of almost 52k time series.""",
    "task": "The task is to  predict the outcomes of a set of soccer matches from leagues worldwide.",
    "problem_name": "soccer-preds",
    "time_stamps": "false",
    "missing": "true",
    "univariate": "false",
    "dimensions": 1,
    "equal_length": "true",
    "class_label": "true",
}

# %% ../../nbs/dataStrcuture/05_ts_format.ipynb 11
def create_ts_file(
    df: pd.DataFrame,  # Pandas Dataframe input.
    file_path: str = ".",  # Where should we save our file ??.
    file_name: str = "games",  # File name.
    file_metadata: Dict = FILE_METADATA,  # File metadata.
) -> None:
    "Create a ts file from a Pandas dataframe."

    # Check path.
    Path(file_path).mkdir(parents=True, exist_ok=True)

    # Create an empty ts file.
    with open(f"{file_path}/{file_name}.ts", "w") as f:
        # Add data length in file metadata.
        file_metadata["series_length"] = len(STATS)+1
        # Init header file information.
        header = "\n".join(
            (
                f'#Source: {file_metadata["source"]}',
                f'#Creator: {file_metadata["creator"]}',
                "#",
                "#Data Set Information:",
                "#",
                f'#{file_metadata["data_description"]}',
                "#",
                f'#{file_metadata["task"]}',
                f'@problemName {file_metadata["problem_name"]}',
                f'@timeStamps {file_metadata["time_stamps"]}',
                f'@missing {file_metadata["missing"]}',
                f'@univariate {file_metadata["univariate"]}',
                f'@dimensions {file_metadata["dimensions"]}',
                f'@equalLength {file_metadata["equal_length"]}',
                f'@seriesLength {file_metadata["series_length"]}',
                f'@classLabel {file_metadata["class_label"]}',
                f"@data",
            )
        )
        # Add header file information.
        f.write(header)
        # Init teams dict.
        team_last_game = {}
        # Loop over data to extract info.
        for _, row in df.iterrows():
            # Extract game information.
            # Home team features.
            home_team_id = row["homeTeamId"]
            home_team_feats = row.filter(like="homeTeam")[2:].tolist()
            # Away team features.
            away_team_id = row["awayTeamId"]
            away_team_feats = row.filter(like="awayTeam")[2:].tolist()
            
            # Add temporal feature.
            # Home.
            home_team_period = 0
            if home_team_id in team_last_game:
                home_team_period = (
                    row["gameDate"] - team_last_game[home_team_id]
                ).days + 1
            team_last_game[home_team_id] = row["gameDate"]

            # Away.
            away_team_period = 0
            if away_team_id in team_last_game:
                away_team_period = (
                    row["gameDate"] - team_last_game[away_team_id]
                ).days + 1
            team_last_game[away_team_id] = row["gameDate"]

            # Put on each row each team features.
            # Add target values(gameId, home and away team Id, scored goals by the given team)
            h_data_str = f"{','.join(str(val) for val in home_team_feats)},{home_team_period}:{row.gameId},{home_team_id},{away_team_id},{row.HS}"
            a_data_str = f"{','.join(str(val) for val in away_team_feats)},{away_team_period}:{row.gameId},{home_team_id},{away_team_id},{row.AS}"
            # Write stringq to the ts file.
            f.write("\n" + h_data_str + "\n" + a_data_str)
