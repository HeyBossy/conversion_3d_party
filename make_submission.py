from baseline import make_pedictions

TEST_DATASET = "./data/private_info/test_df.parquet"
SUBMISSION_PATH = "./data/submission.csv"

if __name__ == "__main__":
    predictions = make_pedictions(TEST_DATASET)
    with open(SUBMISSION_PATH, "w") as f:
        for line in predictions:
            f.write(f"{line}\n")
