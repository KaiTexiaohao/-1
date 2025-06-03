import streamlit as st

st.set_page_config(page_title="Holt-Winters模型", page_icon="��", layout="wide")

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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

# 加载自定义样式
load_custom_css()

# 创建模型信息卡片
create_model_card(
    "Holt-Winters模型 (三参数指数平滑)",
    """
    Holt-Winters模型是最复杂的指数平滑方法，可以同时处理趋势和季节性。
    通过分别平滑水平、趋势和季节性三个分量，实现对复杂时间序列的预测。
    """,
    [
        ("支持趋势预测", True),
        ("支持季节性", True),
        ("需要参数数量", True),
        ("计算复杂度", True),
        ("适合实时处理", False)
    ]
)

# 显示数学原理
st.markdown("### 📐 数学原理")

# LaTeX公式
st.markdown("#### 加法模型的数学公式")

st.markdown("水平分量：")
st.latex(r"l_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(l_{t-1} + b_{t-1})")

st.markdown("趋势分量：")
st.latex(r"b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}")

st.markdown("季节性分量：")
st.latex(r"s_t = \gamma(y_t - l_t) + (1-\gamma)s_{t-m}")

st.markdown("预测公式：")
st.latex(r"y_{t+h} = l_t + hb_t + s_{t-m+h_m}")

st.markdown("""
其中：
- $l_t$ 是t时刻的水平分量
- $b_t$ 是t时刻的趋势分量
- $s_t$ 是t时刻的季节性分量
- $y_t$ 是t时刻的实际观测值
- $\alpha$ 是水平平滑系数
- $\beta$ 是趋势平滑系数
- $\gamma$ 是季节性平滑系数
- $m$ 是季节周期长度
- $h_m$ 是h对m的模运算
""")

# 原理解释
st.markdown("""
#### 💡 工作原理
1. 水平分量捕捉基准值
2. 趋势分量捕捉长期变化
3. 季节性分量捕捉周期性模式
4. 三个平滑参数分别控制各个分量的更新速度
5. 预测值是三个分量的组合
""")

# 加载数据
st.markdown("### 📈 数据与预测")
data, dataset_name = load_sample_data()

# 参数设置
st.sidebar.markdown("### ⚙️ 参数设置")

# 基本参数
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
    beta = st.slider(
        "趋势平滑系数 (β)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="控制趋势分量的平滑程度"
    )

with col2:
    gamma = st.slider(
        "季节平滑系数 (γ)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="控制季节性分量的平滑程度"
    )
    seasonal_periods = st.slider(
        "季节周期",
        min_value=2,
        max_value=24,
        value=12,
        help="一个完整季节的长度"
    )

# 预测设置
forecast_steps = st.sidebar.slider(
    "预测步长",
    min_value=1,
    max_value=30,
    value=10,
    help="向未来预测的时间步数"
)

# 模型类型选择
model_type = st.sidebar.selectbox(
    "模型类型",
    ["加法模型", "乘法模型"],
    help="加法模型适用于季节性波动幅度恒定的数据，乘法模型适用于季节性波动幅度随水平变化的数据"
)

# 计算Holt-Winters模型预测
model = ExponentialSmoothing(
    data,
    seasonal_periods=seasonal_periods,
    trend='add',
    seasonal='add' if model_type == "加法模型" else 'mul'
)

fitted_model = model.fit(
    smoothing_level=alpha,
    smoothing_trend=beta,
    smoothing_seasonal=gamma
)

hw_fitted = fitted_model.fittedvalues

# 进行预测
future_dates = pd.date_range(
    start=data.index[-1],
    periods=forecast_steps + 1,
    freq=data.index.freq
)[1:]
forecast = fitted_model.forecast(forecast_steps)

# 合并拟合值和预测值
full_prediction = pd.concat([hw_fitted, forecast])

# 显示预测结果
st.plotly_chart(
    plot_time_series(
        data,
        full_prediction,
        f"{dataset_name} - Holt-Winters模型预测结果\n(α:{alpha}, β:{beta}, γ:{gamma})"
    ),
    use_container_width=True
)

# 显示误差指标
st.markdown("### 📊 预测效果评估")
metrics = calculate_metrics(data, hw_fitted)
display_metrics(metrics)

# 显示残差分析
st.markdown("### 🔍 残差分析")
st.plotly_chart(plot_residuals(data, hw_fitted), use_container_width=True)

# 分解图
st.markdown("### 📊 时间序列分解")

# 创建三列布局
col1, col2, col3 = st.columns(3)

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

with col3:
    st.markdown("#### 季节性分量")
    season = fitted_model.season
    fig_season = go.Figure()
    fig_season.add_trace(go.Scatter(x=season.index, y=season, name="季节性分量"))
    fig_season.update_layout(title="季节性分量变化", height=300)
    st.plotly_chart(fig_season, use_container_width=True)

# 方法优缺点说明
st.markdown("### 💡 方法优缺点")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 优点")
    st.markdown("""
    - 可以同时处理趋势和季节性
    - 支持加法和乘法季节性
    - 预测准确度高
    - 可解释性强
    """)

with col2:
    st.markdown("#### 缺点")
    st.markdown("""
    - 参数较多，需要仔细调整
    - 计算复杂度较高
    - 需要较长的历史数据
    - 对异常值敏感
    """)

# 适用场景
st.markdown("### 🎯 适用场景")
st.markdown("""
- 具有明显季节性的数据
- 销售数据预测
- 旅游人数预测
- 能源消耗预测
- 零售销量预测
""")

# 添加交互式数据表格
if st.checkbox("显示详细数据"):
    st.markdown("### 📑 详细数据")
    df_display = pd.DataFrame({
        '原始数据': data,
        'Holt-Winters拟合': hw_fitted,
        '预测值': pd.Series(forecast, index=future_dates)
    })
    st.dataframe(df_display) 
