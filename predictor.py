# 导入Stremalit库，用于构建web应用
import streamlit as st

# 导入pickle库，用于加载和保存机器学习模型
import pickle

# 导入Numpy库，用于数值计算
import numpy as np

# 导入Pandas库，用于数据处理和操作
import pandas as pd

# 导入Matplotlib库，用于数据可视化
import matplotlib.pyplot as plt

#
from sklearn.preprocessing import StandardScaler

with open("RSF.pkl", "rb") as f:
    model = pickle.load(f)

# 从x_test.csv文件加载测试数据
# X_test = pd.read_csv("X_test.csv")

feature_names = [
    "xmqc",  # 胸闷气喘
    "stage",  # 分期
    "surgery",  # 是否手术
    "LDH",  # 乳酸脱氢酶
    "D-dimer"  # D二聚体
]

# Streamlit 用户界面
st.title("肺癌合并间质性肺炎生存预测器")  # 设置网页标题

# 胸闷气喘：分类选择框
xmqc = st.selectbox("胸闷气喘", options=[0, 1],
                         format_func=lambda x: "无" if x == 1 else "有")

# 分期：分类选择框
stage = st.selectbox("分期", options=[1, 2, 3, 4])

# 手术：分类选择框
surgery = st.selectbox("手术治疗", options=[0, 1],
                       format_func=lambda x: "否" if x == 1 else "是")

# 乳酸脱氢酶
LDH = st.number_input("乳酸脱氢酶（LDH: U/L）", min_value=0, max_value=1000, value=0)

# 乳酸脱氢酶
Ddimer = st.number_input("D二聚体（D-dimer: mg/L）", min_value=0, max_value=1000, value=0)

# 处理输入数据并进行预测
features = np.array([xmqc, stage, surgery, LDH, Ddimer]).reshape(1, -1) # 将用户输入得特征值存入列表


# 加载 scaler
scaler = StandardScaler()
train_data_scaled = pd.read_csv("train_data_notscaled.csv",index_col=0)
continuous_vars = [ 'LDH', 'Ddimer']
continuous_indices = [3,4]
train_data_scaled[continuous_vars] = scaler.fit(train_data_scaled[continuous_vars])

# 对输入特征中的连续变量进行标准化
features_cont = features[:, continuous_indices]
features_cont_scaled = scaler.transform(features_cont)

# 替换原来的连续变量值
features[:, continuous_indices] = features_cont_scaled

# 当用户点击“Predict”按钮时执行以下代码
if st.button("Predict"):
    # 预测生存函数
    func = model.predict_survival_function(features)
    func = func[0]  # 取第一个样本的生存函数

    # 计算中位生存时间
    thresholds = np.where(func.y <= 0.5)[0]
    if len(thresholds) > 0:
        median_time = func.x[thresholds[0]]
        st.write(f"该患者的中位生存时间: {median_time:.2f} 天")
    else:
        st.write("无法计算中位生存时间（生存概率未降至 0.5 以下）")

    # 绘制生存曲线
    st.subheader("该患者的生存曲线如下：")
    fig, ax = plt.subplots()
    ax.step(func.x, func.y, where="post", label="Survival Function")
    ax.set_xlabel("Time(Days)")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Survival Curve")
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # 在 Streamlit 中显示图表
    st.pyplot(fig)