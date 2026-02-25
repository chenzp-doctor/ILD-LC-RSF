# 导入Stremalit库，用于构建web应用
import streamlit as st

# 导入joblib库，用于加载和保存机器学习模型
import joblib

# 导入Numpy库，用于数值计算
import numpy as np

# 导入Pandas库，用于数据处理和操作
import pandas as pd

# 导入SHAP库，用于解释机器学习模型的预测
import shap

# 导入Matplotlib库，用于数据可视化
import matplotlib.pyplot as plt

model = joblib.load("RSF.pkl")

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
xmqc = st.selectbox("胸闷气喘（0：无；1：有）", options=[0, 1],
                         format_func=lambda x: "无" if x == 1 else "有")

# 分期：分类选择框
stage = st.selectbox("分期", options=[1, 2, 3, 4])

# 手术：分类选择框
surgery = st.selectbox("手术", options=[0, 1],
                       format_func=lambda x: "无" if x == 1 else "有")

# 乳酸脱氢酶
LDH = st.number_input("乳酸脱氢酶（LDH）", min_value=0, max_value=1000, value=0)

# 乳酸脱氢酶
Ddimer = st.number_input("D二聚体（D-dimer）", min_value=0, max_value=1000, value=0)

# 处理输入数据并进行预测
feature_values = ["xmqc", "stage", "surgery",
                  "LDH", "Ddimer"]  # 将用户输入得特征值存入列表
features = np.array([feature_values])

# 加载 scaler
import joblib
scaler = joblib.load('scaler.pkl')
features = scaler.transform(features)
# 当用户点击“Predict”按钮时执行以下代码

if st.button("Predict") :
    # 预测生存函数（返回每个时间点的生存概率）
    func = survgb_model.predict_survival_function(features)
    try:
        median_time = func.x[np.argmax(func.y <= 0.5)]
        st.write(f"该患者的中位生存时间: {median_time:.2f} 天")
    except:
        st.write("无法计算中位生存时间（生存概率未降至 0.5 以下）")
    # 绘制生存曲线
    st.subheader("患者的生存曲线")
    fig, ax = plt.subplots()
    ax.step(func.x, func.y, where="post")
    ax.set_xlabel("时间 (天)")
    ax.set_ylabel("生存概率")
    ax.set_title("生存函数曲线 (Survival Curve)")
    ax.grid(True)

    # 在 Streamlit 中显示图表
    st.pyplot(fig)