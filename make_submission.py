import pandas as pd
from baseline import make_pedictions

TEST_DATASET = "./data/private_info/test_df.parquet"
SUBMISSION_PATH = "./data/submission.csv"

if __name__ == "__main__":
    predictions = pd.Series(make_pedictions(TEST_DATASET))
    predictions.to_csv(SUBMISSION_PATH, index=False, header=True)
