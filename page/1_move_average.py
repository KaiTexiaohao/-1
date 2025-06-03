import streamlit as st

st.set_page_config(page_title="移动平均法", page_icon="📊", layout="wide")

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
    load_custom_css,
    load_sample_data,
    plot_time_series,
    calculate_metrics,
    display_metrics,
    display_latex_formula,
    create_model_card,
    plot_residuals,
    apply_custom_style_to_section
)

# 加载自定义样式
load_custom_css()

# 创建主容器
with st.container():
    # 创建模型信息卡片
    with st.container():
        create_model_card(
            "移动平均法 (Moving Average)",
            """
            移动平均法是一种最基本的时间序列分析方法，通过计算过去固定窗口期内数据的平均值来平滑时间序列数据。
            这种方法特别适合去除短期波动，突出长期趋势。
            """,
            [
                ("支持趋势预测", False),
                ("支持季节性", False),
                ("需要参数数量", True),
                ("计算复杂度", False),
                ("适合实时处理", True)
            ]
        )

    # 数学原理部分
    with st.container():
        st.markdown("### 📐 数学原理")
        with st.container():
            display_latex_formula(
                "移动平均计算公式",
                r"MA_t = \frac{1}{w} \sum_{i=0}^{w-1} Y_{t-i}")

            st.markdown("""
            其中：
            - $MA_t$ 是t时刻的移动平均值
            - $w$ 是窗口大小
            - $Y_{t-i}$ 是t-i时刻的实际值
            """)

    # 数据与预测部分
    with st.container():
        st.markdown("### 📈 数据与预测")
        data, dataset_name = load_sample_data()

        # 参数设置
        st.sidebar.markdown("### ⚙️ 参数设置")
        window_size = st.sidebar.slider(
            "窗口大小",
            min_value=2,
            max_value=30,
            value=7,
            help="移动平均计算使用的历史数据点数量"
        )

        # 计算移动平均
        ma_result = data.rolling(window=window_size, center=False).mean()

        # 显示预测结果
        st.plotly_chart(
            plot_time_series(
                data,
                ma_result,
                f"{dataset_name} - 移动平均法预测结果 (窗口大小: {window_size})"
            ),
            use_container_width=True
        )

    # 预测效果评估
    with st.container():
        st.markdown("### 📊 预测效果评估")
        metrics = calculate_metrics(data, ma_result)
        display_metrics(metrics)

    # 残差分析
    with st.container():
        st.markdown("### 🔍 残差分析")
        st.plotly_chart(plot_residuals(data, ma_result), use_container_width=True)

    # 方法优缺点说明
    with st.container():
        st.markdown("### 💡 方法优缺点")
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.markdown("#### 优点")
                st.markdown("""
                - 简单直观，易于理解和实现
                - 计算速度快，适合实时处理
                - 有效去除随机波动
                - 不需要复杂的参数估计
                """)

        with col2:
            with st.container():
                st.markdown("#### 缺点")
                st.markdown("""
                - 无法预测未来值，只能平滑历史数据
                - 对异常值敏感
                - 存在滞后性，可能错过重要转折点
                - 首尾各损失(w-1)/2个数据点
                """)

    # 适用场景
    with st.container():
        st.markdown("### 🎯 适用场景")
        st.markdown("""
        - 需要平滑短期波动的时间序列数据
        - 实时数据处理和监控
        - 趋势识别和可视化
        - 噪声较大的数据预处理
        """)

    # 详细数据
    if st.checkbox("显示详细数据"):
        with st.container():
            st.markdown("### 📑 详细数据")
            df_display = pd.DataFrame({
                '原始数据': data,
                '移动平均结果': ma_result
            })
            st.dataframe(df_display) 
