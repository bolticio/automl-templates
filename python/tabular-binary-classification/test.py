import pickle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

test = pd.read_csv("./data/test.csv", index_col=0)

features_to_drop = ["ind_var2_0", "ind_var2", "ind_var27_0", "ind_var28_0", "ind_var28",
                    "ind_var27", "ind_var41", "ind_var46_0", "ind_var46", "num_var27_0",
                    "num_var28_0", "num_var28", "num_var27", "num_var41", "num_var46_0",
                    "num_var46", "saldo_var28", "saldo_var27", "saldo_var41", "saldo_var46",
                    "imp_amort_var18_hace3", "imp_amort_var34_hace3",
                    "imp_reemb_var13_hace3", "imp_reemb_var33_hace3",
                    "imp_trasp_var17_out_hace3", "imp_trasp_var33_out_hace3",
                    "num_var2_0_ult1", "num_var2_ult1", "num_reemb_var13_hace3",
                    "num_reemb_var33_hace3", "num_trasp_var17_out_hace3",
                    "num_trasp_var33_out_hace3", "saldo_var2_ult1",
                    "saldo_medio_var13_medio_hace3"]

# Drop these columns from both the test datasets
test = test.drop(features_to_drop, axis=1)

scaler = MinMaxScaler()
X_test = scaler.fit_transform(test)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X_test)

binary_predictions = (predictions >= 0.5).astype(int)

print(binary_predictions)
