import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from typing import Tuple, List
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots

# 添加自定义样式
def load_custom_css():
    """Load custom CSS styles"""
    return st.markdown("""
        <style>
        /* 全局字体大小调整 */
        .stMarkdown, .stText, .stTable {
            font-size: 22px !important;
            line-height: 1.6 !important;
        }
        
        /* 标题样式 */
        h1 {
            font-size: 38px !important;
            font-weight: 600 !important;
            margin-bottom: 1em !important;
            color: #2c3e50 !important;
        }
        
        h2 {
            font-size: 34px !important;
            font-weight: 600 !important;
            margin-bottom: 0.8em !important;
            color: #34495e !important;
        }
        
        h3 {
            font-size: 30px !important;
            font-weight: 600 !important;
            margin-bottom: 0.6em !important;
            color: #2c3e50 !important;
        }
        
        h4 {
            font-size: 26px !important;
            font-weight: 600 !important;
            margin-bottom: 0.4em !important;
            color: #34495e !important;
        }

        /* 卡片样式 */
        .stCard {
            background: white !important;
            border-radius: 10px !important;
            padding: 25px !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1) !important;
            margin-bottom: 25px !important;
        }

        /* 数据指标样式 */
        .metric-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin: 15px 0;
        }

        /* 公式容器样式 */
        .formula-container {
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin: 20px 0;
        }

        /* 图表容器样式 */
        .plot-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin: 20px 0;
        }

        /* 按钮样式 */
        .stButton > button {
            font-size: 20px !important;
            padding: 12px 24px !important;
            border-radius: 6px !important;
            transition: all 0.3s ease !important;
        }

        /* 数据表格样式 */
        .dataframe {
            font-size: 20px !important;
            width: 100% !important;
        }

        .dataframe th {
            background-color: #f8f9fa !important;
            padding: 12px !important;
        }

        .dataframe td {
            padding: 12px !important;
        }

        /* 指标值样式 */
        .css-1wivap2 {
            font-size: 24px !important;
            font-weight: 600 !important;
        }

        /* 指标标签样式 */
        .css-1wivap2 label {
            font-size: 22px !important;
        }

        /* LaTeX公式样式 */
        .katex { 
            font-size: 1.3em !important; 
        }
        
        /* 列表样式 */
        .element-container ol, .element-container ul {
            font-size: 22px !important;
            margin-left: 20px !important;
            margin-bottom: 1em !important;
            line-height: 1.8 !important;
        }

        /* 段落样式 */
        p {
            margin-bottom: 1em !important;
            line-height: 1.8 !important;
        }

        /* 代码块样式 */
        .stCode {
            font-size: 20px !important;
            padding: 15px !important;
            border-radius: 6px !important;
            background-color: #f8f9fa !important;
        }

        /* 提示文本样式 */
        .caption {
            font-size: 18px !important;
            color: #666 !important;
            font-style: italic !important;
            margin-top: 8px !important;
        }
        
        </style>
    """, unsafe_allow_html=True)

def apply_custom_style_to_title(title_text):
    """为标题应用自定义样式"""
    st.markdown(f'<h1>{title_text}</h1>', unsafe_allow_html=True)

def apply_custom_style_to_section(section_text):
    """为章节标题应用自定义样式"""
    st.markdown(f'<h3>{section_text}</h3>', unsafe_allow_html=True)

def create_styled_container(content_function):
    """创建带样式的容器并执行内容函数"""
    st.markdown('<div class="styled-container">', unsafe_allow_html=True)
    content_function()
    st.markdown('</div>', unsafe_allow_html=True)

def create_model_card(title, description, features):
    """创建模型信息卡片"""
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(description)
    
    st.markdown("### 模型特点")
    for feature, value in features:
        icon = "✅" if value else "❌"
        st.markdown(f"{icon} {feature}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_latex_formula(title, formula):
    """显示带样式的LaTeX公式"""
    st.markdown('<div class="formula-container">', unsafe_allow_html=True)
    st.markdown(f"#### {title}")
    st.latex(formula)
    st.markdown('</div>', unsafe_allow_html=True)

def load_sample_data():
    """加载示例数据"""
    # 生成示例时间序列数据
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
    trend = np.linspace(0, 10, 100)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 12)
    noise = np.random.normal(0, 1, 100)
    values = trend + seasonal + noise
    
    data = pd.Series(values, index=dates)
    return data, "示例销售数据"

def plot_time_series(actual, predicted, title):
    """绘制时间序列图表"""
    fig = go.Figure()
    
    # 添加实际值
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual,
        name="实际值",
        line=dict(color="#3498db", width=2)
    ))
    
    # 添加预测值
    fig.add_trace(go.Scatter(
        x=predicted.index,
        y=predicted,
        name="预测值",
        line=dict(color="#e74c3c", width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="值",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def calculate_metrics(actual, predicted):
    """计算预测指标"""
    # 去除NaN值
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        "均方误差 (MSE)": f"{mse:.4f}",
        "均方根误差 (RMSE)": f"{rmse:.4f}",
        "平均绝对误差 (MAE)": f"{mae:.4f}",
        "平均绝对百分比误差 (MAPE)": f"{mape:.2f}%"
    }

def display_metrics(metrics):
    """显示带样式的指标"""
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    cols = st.columns(len(metrics))
    for col, (metric_name, value) in zip(cols, metrics.items()):
        with col:
            st.metric(metric_name, value)
    st.markdown('</div>', unsafe_allow_html=True)

def plot_residuals(actual, predicted):
    """绘制残差分析图"""
    residuals = actual - predicted
    residuals = residuals.dropna()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "残差时间序列",
            "残差直方图",
            "残差散点图",
            "残差QQ图"
        )
    )
    
    # 残差时间序列
    fig.add_trace(
        go.Scatter(x=residuals.index, y=residuals, mode='lines',
                  name="残差", line=dict(color="#3498db")),
        row=1, col=1
    )
    
    # 残差直方图
    fig.add_trace(
        go.Histogram(x=residuals, name="频率分布",
                    marker=dict(color="#3498db")),
        row=1, col=2
    )
    
    # 残差散点图
    fig.add_trace(
        go.Scatter(x=predicted[residuals.index], y=residuals,
                  mode='markers', name="残差散点",
                  marker=dict(color="#3498db")),
        row=2, col=1
    )
    
    # 残差QQ图
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.random.normal(
        loc=np.mean(residuals),
        scale=np.std(residuals),
        size=len(residuals)
    )
    theoretical_quantiles = np.sort(theoretical_quantiles)
    
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                  mode='markers', name="QQ图",
                  marker=dict(color="#3498db")),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig 