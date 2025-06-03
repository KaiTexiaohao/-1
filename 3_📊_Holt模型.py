import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Holt模型", page_icon="��", layout="wide")

# 确保样式被加载
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

import pandas as pd
import numpy as np
from utils import (
    load_sample_data,
    load_custom_css,
    plot_time_series,
    calculate_metrics,
    display_metrics,
    display_latex_formula,
    create_model_card,
    plot_residuals,
    apply_custom_style_to_section
)
from statsmodels.tsa.holtwinters import Holt

# 加载自定义样式
load_custom_css()

# 创建模型信息卡片
create_model_card(
    "Holt模型 (双参数指数平滑)",
    """
    Holt模型是简单指数平滑的扩展，增加了对趋势的处理。
    通过同时平滑水平和趋势两个分量，可以更好地处理具有趋势的时间序列数据。
    """,
    [
        ("支持趋势预测", True),
        ("支持季节性", False),
        ("需要参数数量", True),
        ("计算复杂度", True),
        ("适合实时处理", True)
    ]
)

# 显示数学原理
st.markdown("### 📐 数学原理")

# LaTeX公式
st.markdown("#### 水平分量")
st.latex(r"l_t = \alpha y_t + (1-\alpha)(l_{t-1} + b_{t-1})")

st.markdown("#### 趋势分量")
st.latex(r"b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}")

st.markdown("#### 预测公式")
st.latex(r"y_{t+h} = l_t + hb_t")

st.markdown("""
其中：
- $l_t$ 是t时刻的水平分量
- $b_t$ 是t时刻的趋势分量
- $y_t$ 是t时刻的实际观测值
- $\alpha$ 是水平平滑系数 (0 < α ≤ 1)
- $\beta$ 是趋势平滑系数 (0 < β ≤ 1)
- $h$ 是预测步长
""")

# 原理解释
st.markdown("""
#### 💡 工作原理
1. 水平分量($l_t$)捕捉时间序列的基准值
2. 趋势分量($b_t$)捕捉时间序列的变化率
3. α控制对新数据的响应速度
4. β控制趋势变化的灵敏度
5. 预测值是当前水平加上趋势的延伸
""")

# 加载数据
st.markdown("### 📈 数据与预测")
data, dataset_name = load_sample_data()

# 参数设置
st.sidebar.markdown("### ⚙️ 参数设置")
col1, col2 = st.sidebar.columns(2)

with col1:
    alpha = st.slider(
        "水平平滑系数 (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="控制水平分量的平滑程度"
    )

with col2:
    beta = st.slider(
        "趋势平滑系数 (β)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="控制趋势分量的平滑程度"
    )

# 预测步长设置
forecast_steps = st.sidebar.slider(
    "预测步长",
    min_value=1,
    max_value=30,
    value=10,
    help="向未来预测的时间步数"
)

# 计算Holt模型预测
model = Holt(data)
fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=beta)
holt_fitted = fitted_model.fittedvalues

# 进行预测
future_dates = pd.date_range(
    start=data.index[-1],
    periods=forecast_steps + 1,
    freq=data.index.freq
)[1:]
forecast = fitted_model.forecast(forecast_steps)

# 合并拟合值和预测值
full_prediction = pd.concat([holt_fitted, forecast])

# 显示预测结果
st.plotly_chart(
    plot_time_series(
        data,
        full_prediction,
        f"{dataset_name} - Holt模型预测结果 (α: {alpha}, β: {beta})"
    ),
    use_container_width=True
)

# 显示误差指标
st.markdown("### 📊 预测效果评估")
metrics = calculate_metrics(data, holt_fitted)
display_metrics(metrics)

# 显示残差分析
st.markdown("### 🔍 残差分析")
st.plotly_chart(plot_residuals(data, holt_fitted), use_container_width=True)

# 分解图
st.markdown("### 📊 时间序列分解")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 水平分量")
    level = fitted_model.level
    fig_level = go.Figure()
    fig_level.add_trace(go.Scatter(x=level.index, y=level, name="水平分量"))
    fig_level.update_layout(title="水平分量变化", height=300)
    st.plotly_chart(fig_level, use_container_width=True)

with col2:
    st.markdown("#### 趋势分量")
    trend = fitted_model.trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=trend.index, y=trend, name="趋势分量"))
    fig_trend.update_layout(title="趋势分量变化", height=300)
    st.plotly_chart(fig_trend, use_container_width=True)

# 方法优缺点说明
st.markdown("### 💡 方法优缺点")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 优点")
    st.markdown("""
    - 可以处理具有趋势的数据
    - 支持多步预测
    - 模型解释性强
    - 计算效率较高
    """)

with col2:
    st.markdown("#### 缺点")
    st.markdown("""
    - 不能处理季节性变化
    - 需要调整两个参数
    - 对异常值敏感
    - 长期预测可能不准确
    """)

# 适用场景
st.markdown("### 🎯 适用场景")
st.markdown("""
- 具有明显趋势的时间序列
- 销售额预测
- 股票价格趋势分析
- 经济指标预测
""")

# 添加交互式数据表格
if st.checkbox("显示详细数据"):
    st.markdown("### 📑 详细数据")
    df_display = pd.DataFrame({
        '原始数据': data,
        'Holt模型拟合': holt_fitted,
        '预测值': pd.Series(forecast, index=future_dates)
    })
    st.dataframe(df_display) 