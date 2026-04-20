import streamlit as st
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import SaltRemover

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

# 去盐（非常重要）
remover = SaltRemover.SaltRemover()

# ===============================
# 特征计算函数
# ===============================
def smiles_to_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # 去盐（处理Na+, Mg+等）
    mol = remover.StripMol(mol)

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
# 特征列（必须和训练一致）
# ===============================
selected_features = [
    'NumSaturatedRings','FP_989','FP_1102','fr_Ndealkylation2',
    'FP_1697','FP_255','FP_828','FP_1290','FP_724','FP_486',
    'FP_1287','FP_1272','FP_841','FP_911','FP_117','FP_739','FP_1017'
]

best_threshold = 0.374

# ===============================
# 文件上传（支持xlsx）
# ===============================
uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    # ===============================
    # 列名清洗（防止空格坑）
    # ===============================
    df.columns = df.columns.str.strip()

    # ===============================
    # 必要列检查
    # ===============================
    if "canonical SMILES" not in df.columns:
        st.error("Excel must contain 'canonical SMILES' column")
        st.stop()

    if "Name" not in df.columns:
        df["Name"] = "Unknown"

    results = []

    progress = st.progress(0)

    # ===============================
    # 批量预测
    # ===============================
    for i, row in df.iterrows():

        name = row["Name"]
        smi = row["canonical SMILES"]

        feat_df = smiles_to_features(smi)

        if feat_df is None:
            results.append([name, smi, None, "Invalid"])
            continue

        X = feat_df[selected_features].fillna(0)

        prob = model.predict_proba(X)[0, 1]

        label = "Umami" if prob >= best_threshold else "Non-Umami"

        results.append([name, smi, prob, label])

        progress.progress((i + 1) / len(df))

    # ===============================
    # 结果输出
    # ===============================
    result_df = pd.DataFrame(results, columns=[
        "Name", "SMILES", "Probability", "Prediction"
    ])

    st.success("Prediction Completed")

    st.dataframe(result_df, use_container_width=True)

    # ===============================
    # Top 10（加分项）
    # ===============================
    st.subheader("Top 10 Umami Candidates")
    top_df = result_df.sort_values("Probability", ascending=False).head(10)
    st.dataframe(top_df)

    # ===============================
    # 统计（加分项）
    # ===============================
    st.subheader("Prediction Summary")
    st.write(result_df["Prediction"].value_counts())

    # ===============================
    # 下载Excel
    # ===============================
    from io import BytesIO

    buffer = BytesIO()
    result_df.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        label="Download Results (Excel)",
        data=buffer,
        file_name="umami_prediction.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
