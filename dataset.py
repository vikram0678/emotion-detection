import pandas as pd

isear_url = "https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/master/isear.csv"

isear_df = pd.read_csv(
    isear_url,
    sep="|",
    engine="python",       # more tolerant parser
    on_bad_lines="skip"    # skip malformed rows instead of crashing
)

isear_clean = isear_df[["SIT", "Field1"]].rename(columns={"SIT": "text", "Field1": "emotion"})
isear_clean.to_csv("data/isear_train.csv", index=False)

print("✅ ISEAR saved:", isear_clean.shape)
print(isear_clean.head())