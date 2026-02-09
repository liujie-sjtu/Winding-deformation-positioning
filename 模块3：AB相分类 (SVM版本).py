"""
模块3：A/B相分类 (SVM增强版)
功能：特征提取 + PCA降维 + SVM分类 + 中文混淆矩阵可视化
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# 设置中文字体 (解决Matplotlib中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
OUTPUT_PATH = "output"
MODEL_DIR = "models"
FIGURE_DIR = "figures"  # 图片保存目录

# 剔除通道 (同前次讨论: 高压原边电压 + 绕组振动)
DROP_CHANNELS = [0, 4, 8, 12, 13, 14, 15, 16, 17]

# [可调参数]
PCA_VARIANCE = 0.99  # PCA保留方差比例 (0.95 ~ 0.99)
SVM_C = 10  # SVM惩罚系数 (0.1, 1, 10, 100)
SVM_GAMMA = 'scale'  # 核函数系数 ('scale', 'auto')


def create_directories():
    for d in [MODEL_DIR, FIGURE_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)


# ... (load_data 和 extract_features 函数与之前完全相同，此处省略以节省篇幅，请保留之前的实现) ...
# 请确保这里包含了 load_data 和 extract_features 的完整定义

def plot_confusion_matrix_custom(y_true, y_pred, title, filename):
    """绘制美观的混淆矩阵 (百分比基准：总样本数)"""
    cm = confusion_matrix(y_true, y_pred)

    # [修改点] 改为除以总样本数，这样四个格子的百分比之和为 100%
    total_samples = cm.sum()
    cm_percent = cm.astype('float') / total_samples * 100

    labels = ['A相', 'B相']  # 如果是高低压模块，请改为 ['高压', '低压']

    plt.figure(figsize=(8, 6))

    # 自定义注释文本: 数量 + (占总数百分比)
    annot_labels = [f"{val}\n({pct:.1f}%)" for val, pct in zip(cm.flatten(), cm_percent.flatten())]
    annot_labels = np.asarray(annot_labels).reshape(2, 2)

    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 14}, cbar=True)

    plt.xlabel('预测结果', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()

    save_path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"可视化图表已保存: {save_path}")
    plt.close()


def main():
    create_directories()

    # 1. 加载数据
    train_csv = os.path.join(OUTPUT_PATH, 'train_dataset.csv')
    test_csv = os.path.join(OUTPUT_PATH, 'test_dataset.csv')

    # 注意: 这里的 load_data 函数需使用之前定义的完整版本
    X_train, y_train, _ = load_data(train_csv)
    X_test, y_test, df_test = load_data(test_csv)

    # 2. 模型流水线 (参数可调)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=PCA_VARIANCE)),
        ('svm', SVC(kernel='rbf', C=SVM_C, gamma=SVM_GAMMA, probability=True, random_state=42))
    ])

    # 3. 训练
    print(f"\n开始训练 (C={SVM_C}, PCA={PCA_VARIANCE})...")
    pipeline.fit(X_train, y_train)
    print(f"PCA保留特征数: {pipeline.named_steps['pca'].n_components_}")

    # 4. 预测
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 5. 评估与可视化
    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=['A相', 'B相']))

    # 绘制混淆矩阵
    plot_confusion_matrix_custom(
        y_test, y_pred,
        title=f"A/B相分类结果 - 混淆矩阵 (SVM, 准确率={acc:.1%})",
        filename="AB_confusion_matrix.png"
    )

    # 6. 保存
    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'svm_AB_phase.pkl'))

    results_df = pd.DataFrame({
        'true_label': y_test,
        'pred_label': y_pred,
        'pred_prob_B': y_prob
    })
    results_df.to_csv(os.path.join(OUTPUT_PATH, 'AB_classification_results.csv'), index=False)


# 补全缺失的函数定义以确保运行 (复制粘贴即可)
def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    X_list, y_list = [], []
    for idx, row in df.iterrows():
        if os.path.exists(row['file_path']):
            try:
                data = np.load(row['file_path'])
                X_list.append(extract_features(data))
                y_list.append(row['phase_binary'])
            except:
                pass
    return np.array(X_list), np.array(y_list), df


def extract_features(data):
    all_indices = np.arange(data.shape[1])
    keep_indices = np.setdiff1d(all_indices, DROP_CHANNELS)
    filtered_data = data[:, keep_indices]
    feature_vector = []
    for col in range(filtered_data.shape[1]):
        signal = filtered_data[:, col]
        f_std = np.std(signal)
        if f_std > 1e-6:
            f_skew = stats.skew(signal)
            f_kurt = stats.kurtosis(signal)
        else:
            f_skew = 0
            f_kurt = 0
        feature_vector.extend([
            np.mean(signal), f_std, np.sqrt(np.mean(signal ** 2)),
            np.ptp(signal), f_skew, f_kurt
        ])
    return np.array(feature_vector)


if __name__ == "__main__":
    main()