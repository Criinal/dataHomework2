import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from pandas import read_csv
import numpy as np

input_data = []
def ReadInputCSV(input_csv):
    f = open(input_csv, encoding="UTF-8")
    names = [
        "id",
        "country",
        "description",
        "designation",
        "points",
        "price",
        "province",
        "region_1",
        "region_2",
        "variety",
        "winery",
    ]
    data = read_csv(f, names=names)
    print(data)
    data2 = data.fillna("")
    # data3 = data2[["province", "price", "points"]]
    # print(data2[["price", "points"]])
    return data2
