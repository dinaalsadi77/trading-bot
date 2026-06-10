import pandas as pd
pairs = ["USDJPYm", "EURUSDm", "GBPUSDm", "AUDUSDm", "XAUUSDm", "XAGUSDm", "BTCUSDm"]
for pair in pairs:
    try:
        df = pd.read_csv("forex_labeled/" + pair + "_M15_labeled.csv")
        print(pair + ": " + str(len(df)) + " bars | " + str(df["datetime"].min()) + " to " + str(df["datetime"].max()))
    except FileNotFoundError:
        print(pair + ": FILE NOT FOUND")