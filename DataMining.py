from DataProcessing import ReadInputCSV
from dataMining import mining

input_data = []

# 使用预处理模块读入数据
input_filename = "./DataSet/winemag-data_first150k.csv"
data = ReadInputCSV(input_filename)

# 定义 price >= 50 的数据为高价格
highPrice = []
for price in data["price"]:
    if price == "" or price < 50:
        highPrice.append("LowPrice")
    else:
        highPrice.append("HighPrice")
data["highPrice"] = highPrice

# 定义 points >= 90 的数据为高分数
highPoints = []
for points in data["points"]:
    if points == "" or points < 90:
        highPoints.append("LowPoints")
    else:
        highPoints.append("HighPoints")
data["highPoints"] = highPoints


data1 = data[["province", "highPrice", "highPoints"]]

print("input end")

# 支持度
min_supp = 0.1
# 置信度
min_conf = 0.8
# lift值
min_lift = 2.0

mining(data1, min_supp=0.1, min_conf=0.8, min_lift=2.0)
