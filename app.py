"""
YOLO Vision Studio — Main Application
======================================
Image Object Detection, Segmentation & Pose Estimation
powered by YOLO26, YOLO World v2 and Streamlit.
"""

import streamlit as st
import config
import image_service


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(config.APP_TITLE)
    st.caption(f"版本 {config.APP_VERSION}")
    st.markdown("---")


    # 2️⃣  Task selection ─────────────────────────────────────────────────────
    task = st.radio("处理任务", config.TASKS_LIST, key="task")

    st.markdown("---")

    # 3️⃣  Model selector ─────────────────────────────────────────────────────
    catalog = config.get_model_catalog(task)
    model_label = st.selectbox(
        "模型",
        options=list(catalog.keys()),
        index=0,
        help="微型模型计算量较小，适合先试用。文本检测模型支持按输入的类别进行识别。",
        key="model_select",
    )
    selected_model = catalog[model_label]

    st.markdown("---")

    # 4️⃣  Confidence slider ──────────────────────────────────────────────────
    confidence = (
        st.slider(
            "置信度阈值（%）",
            min_value=config.MIN_CONFIDENCE,
            max_value=config.MAX_CONFIDENCE,
            value=int(config.DEFAULT_CONFIDENCE * 100),
        )
        / 100.0
    )

# ── Main content ──────────────────────────────────────────────────────────────
st.title(config.APP_TITLE)
st.markdown(f"*{config.APP_DESCRIPTION}*")

image_service.render(task, confidence, selected_model)
