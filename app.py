import streamlit as st
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# ===============================
# 页面配置
# ===============================
st.set_page_config(
    page_title="Umami_SFST",
    page_icon="🧬",
    layout="centered"
)

st.title("Umami_SFST Batch Prediction")

# ===============================
# 读取模型
# ===============================
model = joblib.load("model.pkl")

# ===============================
# RDKit描述符
# ===============================
descriptor_names = [desc[0] for desc in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# ===============================
# 特征计算函数
# ===============================
def smiles_to_features(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    fp_array = np.array(fp)
    descriptors = calc.CalcDescriptors(mol)

    features = np.concatenate([fp_array, descriptors])

    fp_cols = [f"FP_{i}" for i in range(2048)]
    desc_cols = descriptor_names
    all_cols = fp_cols + desc_cols

    return pd.DataFrame([features], columns=all_cols)


# ===============================
# 特征列
# ===============================
selected_features = [
'NumSaturatedRings','FP_989','FP_1102','fr_Ndealkylation2',
'FP_1697','FP_255','FP_828','FP_1290','FP_724','FP_486',
'FP_1287','FP_1272','FP_841','FP_911','FP_117','FP_739','FP_1017'
]

best_threshold = 0.374


# ===============================
# 文件上传
# ===============================
uploaded_file = st.file_uploader("Upload CSV file (must contain SMILES column)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "SMILES" not in df.columns:
        st.error("CSV must contain a column named 'SMILES'")
        st.stop()

    results = []

    # ===============================
    # 批量预测
    # ===============================
    for smi in df["SMILES"]:

        feat_df = smiles_to_features(smi)

        if feat_df is None:
            results.append([smi, None, "Invalid"])
            continue

        X = feat_df[selected_features].fillna(0)

        prob = model.predict_proba(X)[0,1]

        label = "Umami" if prob >= best_threshold else "Non-Umami"

        results.append([smi, prob, label])

    # ===============================
    # 结果表
    # ===============================
    result_df = pd.DataFrame(results, columns=[
        "SMILES", "Probability", "Prediction"
    ])

    st.success("Prediction Completed")

    st.dataframe(result_df)

    # ===============================
    # 下载按钮
    # ===============================
    csv = result_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Results",
        data=csv,
        file_name="umami_prediction.csv",
        mime='text/csv'
    )
