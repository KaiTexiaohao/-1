import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    load_custom_css,
    plot_time_series,
    calculate_metrics,
    display_metrics,
    display_latex_formula,
    create_model_card,
    plot_residuals
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
import plotly.express as px

st.set_page_config(page_title="时间序列预测方法可视化教学", page_icon="📈", layout="wide")
load_custom_css()

st.markdown("<h1 style='text-align: center; color: #2c3e50;'>时间序列预测方法可视化教学</h1>", unsafe_allow_html=True)

# 侧边栏模型选择
model = st.sidebar.radio(
    "选择时间序列预测模型：",
    ["移动平均法", "简单指数平滑法", "Holt模型", "Holt-Winters模型"]
)

# 数据选择
st.sidebar.markdown("---")
st.sidebar.markdown("#### 数据集选择")
def load_sample_data():
    np.random.seed(42)
    dataset = st.sidebar.selectbox("选择数据集", ["冰淇淋销量", "旅游人数", "股票价格"])
    if dataset == "冰淇淋销量":
        dates = pd.date_range(start='2023-01-01', periods=48, freq='M')
        trend = np.linspace(0, 10, 48)
        seasonal = 15 * np.sin(np.linspace(0, 4*np.pi, 48))
        noise = np.random.normal(0, 2, 48)
        values = 100 + trend + seasonal + noise
    elif dataset == "旅游人数":
        dates = pd.date_range(start='2023-01-01', periods=60, freq='M')
        trend = np.linspace(0, 30, 60)
        seasonal = 30 * np.sin(np.linspace(0, 4*np.pi, 60))
        noise = np.random.normal(0, 5, 60)
        values = 200 + trend + seasonal + noise
    else:
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        trend = np.linspace(0, 5, 100)
        seasonal = 0
        noise = np.random.normal(0, 1, 100)
        values = 50 + trend + noise
    return pd.Series(values, index=dates), dataset

data, dataset_name = load_sample_data()

# ================== 移动平均法 ==================
if model == "移动平均法":
    create_model_card(
        "移动平均法 (Moving Average)",
        """
        移动平均法是一种最基本的时间序列分析方法，通过计算过去固定窗口期内数据的平均值来平滑时间序列数据。
        适合去除短期波动，突出长期趋势。
        """,
        [
            ("支持趋势预测", False),
            ("支持季节性", False),
            ("需要参数数量", True),
            ("计算复杂度", False),
            ("适合实时处理", True)
        ]
    )
    display_latex_formula("移动平均计算公式", r"MA_t = \frac{1}{w} \sum_{i=0}^{w-1} Y_{t-i}")
    st.markdown("""
    其中：$MA_t$ 是t时刻的移动平均值，$w$ 是窗口大小，$Y_{t-i}$ 是t-i时刻的实际值。
    """)
    window_size = st.sidebar.slider("窗口大小", 2, 30, 7)
    ma_result = data.rolling(window=window_size, center=False).mean()
    st.markdown(f"#### 📉 移动平均预测结果（窗口={window_size}）")
    st.plotly_chart(plot_time_series(data, ma_result, f"{dataset_name} - 移动平均法预测结果"), use_container_width=True)
    metrics = calculate_metrics(data, ma_result)
    display_metrics(metrics)
    st.plotly_chart(plot_residuals(data, ma_result), use_container_width=True)
    if st.checkbox("显示详细数据"):
        st.dataframe(pd.DataFrame({'原始数据': data, '移动平均结果': ma_result}))

# ================== 简单指数平滑法 ==================
elif model == "简单指数平滑法":
    create_model_card(
        "简单指数平滑法 (Simple Exponential Smoothing)",
        """
        简单指数平滑法是一种赋予近期数据更大权重的时间序列预测方法。
        通过单一参数α来控制最新观测值的权重，实现对时间序列的平滑和短期预测。
        """,
        [
            ("支持趋势预测", False),
            ("支持季节性", False),
            ("需要参数数量", True),
            ("计算复杂度", False),
            ("适合实时处理", True)
        ]
    )
    display_latex_formula("预测公式", r"S_t = \alpha Y_t + (1-\alpha)S_{t-1}")
    st.markdown("""
    其中：$S_t$ 是t时刻的平滑值，$Y_t$ 是t时刻的实际观测值，$\alpha$ 是平滑系数 (0 < α ≤ 1)，$S_{t-1}$ 是上一期的平滑值。
    """)
    alpha = st.sidebar.slider("平滑系数 (α)", 0.0, 1.0, 0.2, 0.01)
    def simple_exponential_smoothing(data, alpha):
        result = [data.iloc[0]]
        for n in range(1, len(data)):
            result.append(alpha * data.iloc[n] + (1 - alpha) * result[n-1])
        return pd.Series(result, index=data.index)
    ses_result = simple_exponential_smoothing(data, alpha)
    st.markdown(f"#### 📉 简单指数平滑预测结果（α={alpha}）")
    st.plotly_chart(plot_time_series(data, ses_result, f"{dataset_name} - 简单指数平滑预测结果"), use_container_width=True)
    metrics = calculate_metrics(data, ses_result)
    display_metrics(metrics)
    st.plotly_chart(plot_residuals(data, ses_result), use_container_width=True)
    if st.checkbox("显示详细数据"):
        st.dataframe(pd.DataFrame({'原始数据': data, '指数平滑结果': ses_result}))

# ================== Holt模型 ==================
elif model == "Holt模型":
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
    st.markdown("#### Holt模型数学原理")
    st.latex(r"l_t = \alpha y_t + (1-\alpha)(l_{t-1} + b_{t-1})")
    st.latex(r"b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}")
    st.latex(r"y_{t+h} = l_t + hb_t")
    st.markdown("""
    其中：$l_t$ 是水平分量，$b_t$ 是趋势分量，$\alpha$ 是水平平滑系数，$\beta$ 是趋势平滑系数。
    """)
    alpha = st.sidebar.slider("水平平滑系数 (α)", 0.0, 1.0, 0.2, 0.01)
    beta = st.sidebar.slider("趋势平滑系数 (β)", 0.0, 1.0, 0.1, 0.01)
    forecast_steps = st.sidebar.slider("预测步长", 1, 30, 10)
    model_holt = Holt(data)
    fitted_model = model_holt.fit(smoothing_level=alpha, smoothing_trend=beta)
    holt_fitted = fitted_model.fittedvalues
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq=data.index.freq)[1:]
    forecast = fitted_model.forecast(forecast_steps)
    full_prediction = pd.concat([holt_fitted, forecast])
    st.markdown(f"#### 📉 Holt模型预测结果（α={alpha}, β={beta}）")
    st.plotly_chart(plot_time_series(data, full_prediction, f"{dataset_name} - Holt模型预测结果"), use_container_width=True)
    metrics = calculate_metrics(data, holt_fitted)
    display_metrics(metrics)
    st.plotly_chart(plot_residuals(data, holt_fitted), use_container_width=True)
    if st.checkbox("显示详细数据"):
        df_display = pd.DataFrame({'原始数据': data, 'Holt模型拟合': holt_fitted, '预测值': pd.Series(forecast, index=future_dates)})
        st.dataframe(df_display)

# ================== Holt-Winters模型 ==================
else:
    create_model_card(
        "Holt-Winters模型 (三参数指数平滑)",
        """
        Holt-Winters模型可以同时处理趋势和季节性。
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
    st.markdown("#### Holt-Winters模型数学原理")
    st.latex(r"l_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(l_{t-1} + b_{t-1})")
    st.latex(r"b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}")
    st.latex(r"s_t = \gamma(y_t - l_t) + (1-\gamma)s_{t-m}")
    st.latex(r"y_{t+h} = l_t + hb_t + s_{t-m+h_m}")
    st.markdown("""
    其中：$l_t$ 是水平分量，$b_t$ 是趋势分量，$s_t$ 是季节性分量，$\alpha$、$\beta$、$\gamma$ 分别为平滑系数，$m$ 为季节周期。
    """)
    alpha = st.sidebar.slider("水平平滑系数 (α)", 0.0, 1.0, 0.2, 0.01)
    beta = st.sidebar.slider("趋势平滑系数 (β)", 0.0, 1.0, 0.1, 0.01)
    gamma = st.sidebar.slider("季节平滑系数 (γ)", 0.0, 1.0, 0.1, 0.01)
    seasonal_periods = st.sidebar.slider("季节周期", 2, 24, 12)
    forecast_steps = st.sidebar.slider("预测步长", 1, 30, 10)
    model_type = st.sidebar.selectbox("模型类型", ["加法模型", "乘法模型"])
    model_hw = ExponentialSmoothing(
        data,
        seasonal_periods=seasonal_periods,
        trend='add',
        seasonal='add' if model_type == "加法模型" else 'mul'
    )
    fitted_model = model_hw.fit(
        smoothing_level=alpha,
        smoothing_trend=beta,
        smoothing_seasonal=gamma
    )
    hw_fitted = fitted_model.fittedvalues
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq=data.index.freq)[1:]
    forecast = fitted_model.forecast(forecast_steps)
    full_prediction = pd.concat([hw_fitted, forecast])
    st.markdown(f"#### 📉 Holt-Winters模型预测结果（α={alpha}, β={beta}, γ={gamma}）")
    st.plotly_chart(plot_time_series(data, full_prediction, f"{dataset_name} - Holt-Winters模型预测结果"), use_container_width=True)
    metrics = calculate_metrics(data, hw_fitted)
    display_metrics(metrics)
    st.plotly_chart(plot_residuals(data, hw_fitted), use_container_width=True)
    if st.checkbox("显示详细数据"):
        df_display = pd.DataFrame({'原始数据': data, 'Holt-Winters拟合': hw_fitted, '预测值': pd.Series(forecast, index=future_dates)})
        st.dataframe(df_display) 
