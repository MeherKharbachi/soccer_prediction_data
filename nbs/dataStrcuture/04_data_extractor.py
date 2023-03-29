#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#| default_exp datastructure.data_extractor


# In[ ]:


#| hide

#from IPython.core.debugger import set_trace

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# # Data Extractor
# > Extract games and its features from multiple DB collections.

# In[ ]:


#| export

import datetime
import json
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple
from soccer_prediction_data.config.mongo import mongo_init
from soccer_prediction_data.datastructure.fixture import * 
from soccer_prediction_data.datastructure.team_stats import *


# In[ ]:


# |export

STATS = [
    "Ball possession in own half",
    "Ball possession in opp. half",
    "Challenges / won",
    "Challenges lost",
    "Challenge intensity index",
    "Passes accurate",
    "Accurate passes into the final third of the pitch",
    "Passes into the penalty box",
    "Diagonal passes",
    "Key passes",
    "Crosses accurate",
    "Crosses to the near post - % efficiency",
    "Crosses into the six-yard box - % efficiency",
    "Shots on target",
    "Shots on post / bar",
    "Shots from the box - accurate",
    "GK actions - Shots saved",
    "xG",
    "xG per shot",
    "Chance TOTAL",
    "Chances, % of conversion",
    "Attacking mentality index",
    "Dribbles successful",
    "Dribbles unsuccessful",
    "Successful tackles",
    "Unsuccessful tackles",
    "Interceptions / in opp. half",
    "Ball recoveries / in opp. half",
    "Fouls",
    "Lost balls / in own half",
    "Average distance to the goal at ball losses",
    "Average distance to the goal at ball recoveries",
]

COMPETITION_IDS = [
    1,    # Russia. Premier League
    9,    # Russia. FNL
    20,   # Spain. Primera Division
    24,   # Italy. Serie A
    28,   # Portugal. Liga NOS
    29,   # Netherlands. Eredivisie
    31,   # Germany. Bundesliga
    37,   # France. Ligue 1
    39,   # England. Premier League
    41,   # United States. MLS
    45,   # Belgium. Jupiler Pro League
    52,   # Sweden. Allsvenskan
    72,   # Germany. 2. Bundesliga
    76,   # England. League Two
    78,   # Switzerland. Super League
    80,   # Italy. Serie B
    86,   # Norway. Eliteserien
    93,   # Argentina. Primera Division
    95,   # Australia. A-League
    103,  # Scotland. Premier League
    105,  # England. Championship
    108,  # Mexico. Liga MX
    109,  # Spain. Segunda Division
    110,  # France. Ligue 2
    112,  # Morocco. GNF 1
    123,  # England. League One
    193,  # Algeria. Ligue 1
    213,  # Chile. Primera Division
    300,  # Scotland. Championship
    307,  # South Africa. PSL
    464,  # Germany. 3. Liga
    496,  # France. Championnat National
    792,  # Scotland. League One
    903,  # Scotland League Two
    936,  # England. National League North
]


# ## Aggregate Data
# 

# We provide a function that seeks to retrieve the list of games recorded in our `gameFeatures` MongoDb Collection and aggregate it with its additional features such as `Lineups` information (lineups Collection) and `1x2`, `Asian Handicap` and `Total` odds (Odds collection).

# In[ ]:


# | export


def data_aggregator(
    competition_ids=COMPETITION_IDS,  # Competitions to extract.
    limit: int = None,  # Number of rows to extract.
) -> pd.DataFrame:  # Mapped games.
    "Returns and aggregates games information from multiple Db collections."

    def _team_stats(
        game_id: int,  # Instat game identifier.
        team_id: int,  # Instat team identifier.
    ) -> pd.DataFrame:  # Team Stats
        "Returns stats of a given team in a given game."

        # Team features.
        team_feats = TeamStats.get_game_team_stats(
            game_id=game_id,
            team_id=team_id,
        )
        if team_feats is None:
            print("gameId:", game_id)
            print("teamId:", team_id)
            missings.append(game_id)
            return pd.DataFrame(index=[0])

        # Team stats.
        team_stats = {
            stat.action_name.strip()
            .replace("- ", "")
            .replace(", ", "")
            .replace("/ ", "")
            .title()
            .replace(" ", ""): stat.value
            for stat in team_feats.stats
            if stat.action_name in STATS
        }

        return pd.DataFrame(team_stats, index=[0])

    # Extract games.
    games = Fixture.get_games_by_competition(
        competition_ids=competition_ids, limit=limit
    )
    games = pd.DataFrame(games.as_pymongo())
    print(games.shape)

    def _laaa(x):
        print(x)
        if len(x)>0:
            print("-----------------------------------")
            return x[0]
        else:
            print("-----------------------------------")
            return -1
        

    # Map results {HS: Home goals scored, AS: Away goals scored}.
    games[["HS", "AS"]] = games["fullTimeScore"].apply(
    lambda lst: pd.Series(
        [
            (lst[0] if len(lst) > 0  else -1),
            (lst[1] if len(lst) > 0  else -1),
        ]
    )
    if isinstance(lst, list)
    else pd.Series([-1, -1])
)

    # Filter Data.
    games = games[
        [
            "gameId",
            "gameDate",
            "seasonName",
            "competitionName",
            "homeTeamId",
            "homeTeamName",
            "awayTeamId",
            "awayTeamName",
            "HS",
            "AS",
        ]
    ]

    # Filter df.
    future_games = games[games.HS == -1].copy()
    played_games = games[games.HS != -1].copy()

    if played_games.empty:
        return None, future_games

    # compute other features
    def _one_game(row):
        ht_stats = _team_stats(
            game_id=row["gameId"], team_id=row["homeTeamId"]
        ).add_prefix("homeTeam")

        at_stats = _team_stats(
            game_id=row["gameId"], team_id=row["awayTeamId"]
        ).add_prefix("awayTeam")

        res = pd.concat([ht_stats, at_stats], axis=1)
        res.loc[:, "gameId"] = row.gameId
        print("------------------------------------------------")

        return res

    played_games = played_games.merge(
        pd.concat(
            [
                _one_game(row)
                for _, row in tqdm(played_games.iterrows(), total=played_games.shape[0])
            ]
        ).reset_index(drop=True),
        on="gameId",
        how="left",
    )

    return played_games, future_games


# In[ ]:


missings=[]


# In[ ]:


mongo_init(db_host="prod_atlas")

x2,x3 = data_aggregator()


# In[ ]:


x2.to_csv("p_games.csv")
x3.to_csv("f_games.csv")


# In[ ]:


l = x2.filter(like="awayTeam").iloc[:,2:].columns
ll = x2.filter(like="homeTeam").iloc[:,2:].columns

print(len(l))
print(len(ll))


# In[ ]:


for p in ll:
    p = p.replace("homeTeam","awayTeam")
    if p not in l:
        print(p)



print("indexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print(missings)

# In[ ]:


#| hide

import nbdev

nbdev.nbdev_export()

