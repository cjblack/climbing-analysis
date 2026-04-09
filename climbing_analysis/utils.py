import json
import pandas as pd

def saveas_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def saveas_dataframe(file_path, data):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data