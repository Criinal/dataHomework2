import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def mining(data,
           min_supp=0.1,
           min_conf=0.8,
           min_lift=2.0):
    # item_df = pd.DataFrame(input_data)
    item_df = data
    # print(item_df)
    te = TransactionEncoder()
    input_data = data.values.tolist()
    print("input data")
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            input_data[i][j] = str(input_data[i][j])
    df_tf = te.fit_transform(input_data)

    df = pd.DataFrame(df_tf, columns=te.columns_)
    print("encode end")

    print("frequent pattern begin")
    # use_colnames=True表示使用元素名字，默认的False使用列名代表元素, 设置最小支持度min_support
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
    print("frequent pattern end")
    frequent_itemsets.sort_values(by="support", ascending=False, inplace=True)

    # 选择2频繁项集
    print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) == 2])
    print("frequent pattern end")
    frequent_itemsets.to_csv("frequentItemsets.csv")


    # metric可以有很多的度量选项，返回的表列名都可以作为参数
    association_rule = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    association_rule.to_csv("associationRules.csv")

    # 关联规则可以提升度排序
    association_rule.sort_values(by="lift", ascending=False, inplace=True)
    print(association_rule)
    # 规则是：antecedents->consequents
    print("rule end")
