import streamlit as st
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw


# ===============================
# 页面配置
# ===============================
st.set_page_config(
    page_title="Umami_SFST",
    page_icon="🧬",
    layout="centered"
)


# ===============================
# UI样式
# ===============================
st.markdown("""
<style>

/* 页面整体字体 */
html, body, [class*="css"] {
    font-size:22px !important;
}

/* 输入框label */
label{
font-size:28px !important;
font-weight:600;
}

/* 输入框 */
.stTextInput input{
font-size:26px !important;
padding:14px !important;
border-radius:10px !important;
}

/* 按钮 */
.stButton button{
font-size:26px !important;
padding:14px 40px !important;
border-radius:12px !important;
}

/* 结果框 */
.result-box{
background-color:#eafaf1;
padding:30px;
border-radius:14px;
text-align:center;
font-size:36px;
margin-top:40px;
font-weight:700;
}

</style>
""", unsafe_allow_html=True)


# ===============================
# 标题（用HTML强制放大）
# ===============================
st.markdown("""
<h1 style='
text-align:center;
font-size:72px;
font-weight:800;
color:#1f2d3d;
margin-bottom:5px;
letter-spacing:2px;
'>
Umami_SFST
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style='
text-align:center;
font-size:28px;
color:#5d6d7e;
margin-bottom:35px;
font-weight:500;
'>
AI Umami Molecule Prediction System
</h2>
""", unsafe_allow_html=True)


# ===============================
# 读取模型
# ===============================
model = joblib.load("model.pkl")


# ===============================
# RDKit描述符初始化
# ===============================
descriptor_names = [desc[0] for desc in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


# ===============================
# 特征计算函数
# ===============================
def smiles_to_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    fp_array = np.array(fp)

    # RDKit descriptors
    descriptors = calc.CalcDescriptors(mol)

    features = np.concatenate([fp_array, descriptors])

    fp_cols = [f"FP_{i}" for i in range(2048)]
    desc_cols = descriptor_names

    all_cols = fp_cols + desc_cols

    df = pd.DataFrame([features], columns=all_cols)

    return df, mol


# ===============================
# 模型特征
# ===============================
selected_features = [
'NumSaturatedRings','FP_989','FP_1102','fr_Ndealkylation2',
'FP_1697','FP_255','FP_828','FP_1290','FP_724','FP_486',
'FP_1287','FP_1272','FP_841','FP_911','FP_117','FP_739','FP_1017'
]


# ===============================
# 输入区域
# ===============================
smiles = st.text_input(
"Enter SMILES",
"CC1CCCCC1NC(=O)c1cccc(OC(F)(F)F)c1"
)


# ===============================
# 预测
# ===============================
if st.button("Predict Umami Probability"):

    feat_df, mol = smiles_to_features(smiles)

    if feat_df is None:
        st.error("Invalid SMILES")
        st.stop()

    # 显示分子结构
    img = Draw.MolToImage(mol, size=(350,350))
    st.image(img, caption="Molecule Structure")

    # 选择特征
    X = feat_df[selected_features].fillna(0)

    # 预测概率
    prob = model.predict_proba(X)[0,1]

    # 输出概率
    st.markdown(
        f'<div class="result-box">Umami probability: {prob:.4f}</div>',
        unsafe_allow_html=True
    )

    # 概率进度条

    st.progress(float(prob))

