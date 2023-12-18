import pandas as pd
from datetime import datetime, timedelta

def findCont(df):
    scoresDict = {}
    df['Collection_Date'] = pd.to_datetime(df['Collection_Date'])
    for name in df['Pangolin']:
        name_df = df[df['Pangolin'] == name]

        startDate = name_df['Collection_Date'].min()
        endDate = startDate + timedelta(days=30)
        print("Start: ", startDate)
        print("End: ", endDate)

        uploads_count = name_df[(name_df['Collection_Date'] >= startDate) & (name_df['Collection_Date'] <= endDate)].shape[0]

        firstVac = "Dec 14 00:00:00 2020"
        secondVac = "Sep 20 00:00:00 2021"
        firstVac, secondVac = map(lambda ts: datetime.strptime(ts, "%b %d %H:%M:%S %Y"), [firstVac, secondVac])
        if endDate > secondVac:
            uploads_count *= 2
        elif endDate > firstVac:
            uploads_count *= 1.5
        scoresDict[name] = uploads_count

    return scoresDict

df = pd.read_csv("../data/processed_data.csv")
scoresDict = findCont(df)
print(scoresDict)

