import pandas as pd
import numpy as np

data = pd.read_csv("D:/python/bank/train.csv")
test = pd.read_csv("D:/python/bank/test.csv")
# 特征工程
# 客户号CUST_ID,开户机构OPEN_ORG_NUM,证件类型IDF_TYP_CD,性别GENDER,是否有欧元账户CUST_EUP_ACCT_FLAG,是否有澳元账户CUST_AU_ACCT_FLAG,CUST_DOLLER_FLAG	是否美元卡，CUST_INTERNATIONAL_GOLD_FLAG	是否国际金卡，CUST_INTERNATIONAL_COMMON_FLAG	是否国际普卡，CUST_INTERNATIONAL_SIL_FLAG	是否国际银卡，CUST_INTERNATIONAL_DIAMOND_FLAG	是否国际钻石卡，CUST_GOLD_COMMON_FLAG	是否金普卡，CUST_STAD_PLATINUM_FLAG	是否标准白金卡，CUST_LUXURY_PLATINUM_FLAG	是否豪华白金卡，CUST_PLATINUM_FINANCIAL_FLAG	是否白金理财卡，CUST_DIAMOND_FLAG	是否钻石卡，CUST_INFINIT_FLAG	是否无限卡 ，CUST_BUSINESS_FLAG	是否商务卡
# 明显无意义的列
MeaninglessColumns = [
    "OPEN_ORG_NUM",
    "IDF_TYP_CD",
    "GENDER",
    "CUST_EUP_ACCT_FLAG",
    "CUST_AU_ACCT_FLAG",
    "CUST_DOLLER_FLAG",
    "CUST_INTERNATIONAL_GOLD_FLAG",
    "CUST_INTERNATIONAL_COMMON_FLAG",
    "CUST_INTERNATIONAL_SIL_FLAG",
    "CUST_INTERNATIONAL_DIAMOND_FLAG",
    "CUST_GOLD_COMMON_FLAG",
    "CUST_STAD_PLATINUM_FLAG",
    "CUST_LUXURY_PLATINUM_FLAG",
    "CUST_PLATINUM_FINANCIAL_FLAG",
    "CUST_DIAMOND_FLAG",
    "CUST_INFINIT_FLAG",
    "CUST_BUSINESS_FLAG",
]
# 将训练集中取值唯一、取值各不相同的特征删除函数
def del_features(data):
    columns = []
    for name in data.columns:
        unique = data[name].unique().shape[0]
        full = data[name].shape[0]
        if (unique == 1) | (unique == full):
            columns.append(name)
    return columns


# 保存
for i in del_features(data):
    MeaninglessColumns.append(i)

print(MeaninglessColumns)
data.drop(MeaninglessColumns, axis=1, inplace=True)
MeaninglessColumns.remove("CUST_ID")
test.drop(MeaninglessColumns, axis=1, inplace=True)
print(data.head())
# 数据预处理
# 显示全部
pd.set_option("display.max_rows", None)
# 是否有缺失值
print(data.isnull().any())
print(test.isnull().any())
# 训练集中是否有重复行
print(data.duplicated().value_counts())
# 删除训练集中的重复行
data = data.drop_duplicates(keep="first")
print(data.duplicated().value_counts())
# 分类
# 找出数字列
data_num = data.select_dtypes(include=[np.number])
data_non_num = data.select_dtypes(exclude=[np.number])
# one-hot编码
data_non_num_num = pd.get_dummies(data_non_num)
# one-hot编码后数据并合并
data = pd.concat([data_num, data_non_num_num], axis=1)
# 找出数字列
test_num = test.select_dtypes(include=[np.number])
test_non_num = test.select_dtypes(exclude=[np.number])
# one-hot编码
test_non_num_num = pd.get_dummies(test_non_num)
# one-hot编码后数据并合并
test = pd.concat([test_num, test_non_num_num], axis=1)
print(data.info())
print(test.info())
# 提取标签
data_target = data["bad_good"]
data.drop(["bad_good"], axis=1, inplace=True)
# 划分训练集
from sklearn.model_selection import train_test_split

# 划分数据集 三七分数据集，这里将训练集再分为测试集和训练集是为了便于本地测试模型效果
x_train_local, x_test_local, y_train_local, y_test_local = train_test_split(
    data, data_target, test_size=0.3
)
from sklearn.model_selection import cross_val_score
import time

# 决策树
from sklearn.tree import DecisionTreeClassifier as DTC

# 随机森林
from sklearn.ensemble import RandomForestClassifier as RFC

# xgboost
from xgboost import XGBRegressor

# Adaboost
from sklearn.ensemble import AdaBoostRegressor as ABR

times = []
scores = []
models = [DTC(), RFC(n_estimators=100), XGBRegressor(), ABR()]
names = ["决策树", "随机森林", "XGBoost", "Adaboost"]
for i in range(4):
    time1 = time.time()
    #     model = models[i].fit(x_train_local,y_train_local)
    # 10次交叉验证
    score = cross_val_score(models[i], x_train_local, y_train_local, cv=10).mean()
    time2 = time.time()
    #     score=model.score(x_test_local,y_test_local)
    scores.append(float(score))
    endTime = time2 - time1
    times.append(endTime)
    print(names[i], score, "耗时:" + str(endTime))
# 画图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
x = np.arange(4)
width = 0.25
ax = plt.subplot(1, 1, 1)
ax.bar(x, times, width, color="r", label="耗时")
ax.bar(x + width, scores, width, color="b", label="分数")
ax.set_xticks(x + width)
ax.set_xticklabels(names)
ax.legend()
plt.show()
# 选择模型并输出结果
x_test = test.drop(["CUST_ID"], axis=1)
print("选择的模型是:" + names[np.argmax(scores)])
pred = models[np.argmax(scores)].fit(x_train_local, y_train_local).predict(x_test)
pred = pd.DataFrame(pred)
pred["bad_good"] = pred
pred.drop(0, axis=1, inplace=True)
sub = pd.concat([test["CUST_ID"], pred], axis=1)
sub.to_csv("D:/submission.csv", index=0)
