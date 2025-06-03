import streamlit as st

st.set_page_config(page_title="简单指数平滑法", page_icon="📈", layout="wide")

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

# 加载自定义样式
load_custom_css()

# 创建主容器
with st.container():
    # 创建模型信息卡片
    with st.container():
        create_model_card(
            "简单指数平滑法 (Simple Exponential Smoothing)",
            """
            简单指数平滑法是一种赋予近期数据更大权重的时间序列预测方法。
            通过单一参数α来控制最新观测值的权重，实现对时间序列的平滑和短期预测。
            """,
            [
                ("支持趋势预测", True),
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
                "预测公式",
                r"S_t = \alpha Y_t + (1-\alpha)S_{t-1}")

            st.markdown("""
            其中：
            - $S_t$ 是t时刻的平滑值
            - $Y_t$ 是t时刻的实际观测值
            - $\alpha$ 是平滑系数 (0 < α ≤ 1)
            - $S_{t-1}$ 是上一期的平滑值
            """)

            st.markdown("""
            #### 💡 工作原理
            1. α越大，模型对新数据越敏感
            2. α越小，平滑效果越明显
            3. 预测值是历史数据的加权平均
            4. 权重随时间呈指数衰减
            """)

    # 数据与预测部分
    with st.container():
        st.markdown("### 📈 数据与预测")
        data, dataset_name = load_sample_data()

        # 参数设置
        st.sidebar.markdown("### ⚙️ 参数设置")
        alpha = st.sidebar.slider(
            "平滑系数 (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="控制新数据的权重"
        )

        # 计算简单指数平滑
        def simple_exponential_smoothing(data, alpha):
            result = [data.iloc[0]]  # 初始值
            for n in range(1, len(data)):
                result.append(alpha * data.iloc[n] + (1 - alpha) * result[n-1])
            return pd.Series(result, index=data.index)

        # 计算预测结果
        ses_result = simple_exponential_smoothing(data, alpha)

        # 显示预测结果
        st.plotly_chart(
            plot_time_series(
                data,
                ses_result,
                f"{dataset_name} - 简单指数平滑预测结果 (α: {alpha})"
            ),
            use_container_width=True
        )

    # 预测效果评估
    with st.container():
        st.markdown("### 📊 预测效果评估")
        metrics = calculate_metrics(data, ses_result)
        display_metrics(metrics)

    # 残差分析
    with st.container():
        st.markdown("### 🔍 残差分析")
        st.plotly_chart(plot_residuals(data, ses_result), use_container_width=True)

    # 权重衰减可视化
    with st.container():
        st.markdown("### 📊 权重衰减可视化")
        periods = 10
        weights = [(1-alpha)**i * alpha for i in range(periods)]
        weights_df = pd.DataFrame({
            '时间延迟': range(periods),
            '权重': weights
        })

        import plotly.express as px
        fig = px.bar(
            weights_df,
            x='时间延迟',
            y='权重',
            title=f'权重随时间的衰减 (α={alpha})'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # 方法优缺点说明
    with st.container():
        st.markdown("### 💡 方法优缺点")
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.markdown("#### 优点")
                st.markdown("""
                - 计算简单，易于实现
                - 能够快速响应数据变化
                - 存储需求小，适合实时处理
                - 可以进行短期预测
                """)

        with col2:
            with st.container():
                st.markdown("#### 缺点")
                st.markdown("""
                - 不适合有明显趋势的数据
                - 不能处理季节性变化
                - 预测值总是滞后于实际变化
                - 对异常值较敏感
                """)

    # 适用场景
    with st.container():
        st.markdown("### 🎯 适用场景")
        st.markdown("""
        - 短期预测和实时监控
        - 平稳的时间序列数据
        - 需要快速响应数据变化的场景
        - 计算资源有限的环境
        """)

    # 详细数据
    if st.checkbox("显示详细数据"):
        with st.container():
            st.markdown("### 📑 详细数据")
            df_display = pd.DataFrame({
                '原始数据': data,
                '指数平滑结果': ses_result
            })
            st.dataframe(df_display) 
