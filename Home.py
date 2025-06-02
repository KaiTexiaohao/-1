import streamlit as st

st.set_page_config(
    page_title="时间序列预测方法展示",
    page_icon="📈",
    layout="wide"
)

import pandas as pd
from utils import load_custom_css, apply_custom_style_to_title, apply_custom_style_to_section, create_styled_container

# 确保样式被加载
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 加载自定义样式
load_custom_css()

# 添加标题和介绍
apply_custom_style_to_title("📈 时间序列预测方法可视化展示")

# 添加简介
def intro_content():
    st.markdown("""
    ## 欢迎使用时间序列预测方法展示工具！

    本应用旨在帮助您理解和比较不同的时间序列预测方法。通过交互式的可视化和详细的解释，
    您可以深入了解每种方法的特点、适用场景和实现原理。

    ### 🎯 主要功能

    - 📊 交互式可视化展示
    - 🎛️ 实时参数调整
    - 📝 详细的原理解释
    - 🔍 模型对比分析
    - 📥 支持自定义数据
    """)

create_styled_container(intro_content)

# 创建模型对比表格
apply_custom_style_to_section("📋 模型特点对比")

def comparison_table():
    comparison_data = {
        "方法名": ["移动平均法", "简单指数平滑", "Holt线性趋势模型", "Holt-Winters模型"],
        "是否支持趋势": ["❌", "✅(弱)", "✅", "✅"],
        "是否支持季节性": ["❌", "❌", "❌", "✅"],
        "是否支持未来预测": ["⛔(仅平滑)", "✅", "✅", "✅"],
        "典型应用场景": [
            "短期平滑需求曲线",
            "稳定销售预测",
            "销售、股价、线性趋势业务",
            "月度气温、消费、电商流量等"
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.markdown('''
    <style>
    .comparison-table {
        font-family: Arial, sans-serif;
    }
    .comparison-table th {
        background-color: #3498db !important;
        color: white !important;
        font-weight: bold !important;
    }
    .comparison-table td {
        background-color: #f8f9fa !important;
    }
    .comparison-table tr:nth-child(even) td {
        background-color: #ffffff !important;
    }
    </style>
    ''', unsafe_allow_html=True)
    
    st.table(df_comparison.style.set_table_attributes('class="comparison-table"'))

create_styled_container(comparison_table)

# 添加导航说明
apply_custom_style_to_section("🧭 导航指南")

def navigation_guide():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 模型演示页面
        - 每个预测方法都有独立的演示页面
        - 包含交互式参数调整
        - 实时查看预测效果

        #### 🎓 原理教学
        - 通俗易懂的算法解释
        - 数学公式推导
        - 动态图解演示
        """)
    
    with col2:
        st.markdown("""
        #### 📈 模型对比
        - 多模型效果对比
        - 误差指标分析
        - 自动推荐最佳模型

        #### 🛠️ 实用工具
        - 数据导入导出
        - 预测结果下载
        - 参数优化建议
        """)

create_styled_container(navigation_guide)

# 添加使用提示
with st.sidebar:
    st.markdown('''
    <div style="background-color: #f0f4f8; padding: 1rem; border-radius: 8px; border-left: 4px solid #3498db;">
        <h3 style="color: #2c3e50;">💡 使用提示</h3>
        <ol style="color: #34495e;">
            <li>使用左侧导航栏切换不同页面</li>
            <li>在各个模型页面调整参数</li>
            <li>观察预测效果变化</li>
            <li>比较不同模型的表现</li>
        </ol>
    </div>
    ''', unsafe_allow_html=True)

# 添加页脚
st.markdown("---")
apply_custom_style_to_section("📚 参考资料")

def references():
    st.markdown('''
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px;">
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin: 0.5rem 0;">
                <a href="https://otexts.com/fpp2/" style="color: #3498db; text-decoration: none;">
                    📖 时间序列分析基础
                </a>
            </li>
            <li style="margin: 0.5rem 0;">
                <a href="https://www.statsmodels.org/stable/index.html" style="color: #3498db; text-decoration: none;">
                    📊 Statsmodels文档
                </a>
            </li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

create_styled_container(references) 