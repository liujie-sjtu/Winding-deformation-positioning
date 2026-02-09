"""
模块2：数据集划分（适配版）
功能：
1. 读取模块1生成的索引文件 labeled_data.csv
2. 创建4类组合标签 (A高/A低/B高/B低) 用于分层抽样
3. 验证 .npy 数据文件的有效性
4. 划分训练集和测试集并保存为CSV
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 输入路径需与模块1输出保持一致
OUTPUT_PATH = "output"
INPUT_CSV = os.path.join(OUTPUT_PATH, 'labeled_data.csv')

# 划分参数
TEST_SIZE = 0.3  # 测试集比例 30%
RANDOM_STATE = 42  # 随机种子，保证可复现

# 4类标签映射 (仅用于打印信息)
LABEL_MAPPING = {
    0: 'A相高压 (PH=0, VOL=0)',
    1: 'A相低压 (PH=0, VOL=1)',
    2: 'B相高压 (PH=1, VOL=0)',
    3: 'B相低压 (PH=1, VOL=1)'
}


def create_output_directory():
    """创建输出目录（如果不存在）"""
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)


def load_and_validate_data(input_file):
    """加载索引文件并验证数据有效性"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到输入文件: {input_file}，请先运行模块1。")

    df = pd.read_csv(input_file)
    print(f"原始索引包含 {len(df)} 条记录")

    # 1. 检查必需列
    required_cols = ['file_path', 'phase_binary', 'voltage_binary']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV文件缺少必需列: {required_cols}")

    # 2. 移除标签缺失的行 (理论上Mod1已处理，这里双重保险)
    df_clean = df.dropna(subset=['phase_binary', 'voltage_binary']).copy()

    # 3. 验证 .npy 文件是否存在
    # 这步很重要，防止训练时报 FileNotFoundError
    valid_indices = []
    for idx, row in df_clean.iterrows():
        if os.path.exists(row['file_path']):
            valid_indices.append(idx)
        else:
            print(f"警告: 文件丢失，已剔除: {row['file_path']}")

    df_final = df_clean.loc[valid_indices].copy()

    # 转换标签为整数
    df_final['phase_binary'] = df_final['phase_binary'].astype(int)
    df_final['voltage_binary'] = df_final['voltage_binary'].astype(int)

    # 4. 生成组合标签用于分层抽样
    # 0: A高, 1: A低, 2: B高, 3: B低
    df_final['stratify_label'] = df_final['phase_binary'] * 2 + df_final['voltage_binary']

    print(f"验证后有效样本数: {len(df_final)}")
    return df_final


def analyze_distribution(df, label_col='stratify_label'):
    """分析类别分布"""
    counts = Counter(df[label_col])
    print("\n---------- ���别分布分析 ----------")
    for label in sorted(counts.keys()):
        count = counts[label]
        pct = count / len(df) * 100
        desc = LABEL_MAPPING.get(label, f"未知标签 {label}")
        print(f"  类别 {label} [{desc}]: {count} 样本 ({pct:.1f}%)")

    # 简单检查平衡性
    if counts:
        min_c = min(counts.values())
        max_c = max(counts.values())
        ratio = max_c / min_c if min_c > 0 else float('inf')
        print(f"  不平衡比例 (Max/Min): {ratio:.2f}")
        if ratio > 1.5:
            print("  提示: 类别存在一定不平衡，分层抽样非常必要。")


def main():
    print("==================== 模块2: 数据集划分 ====================")
    create_output_directory()

    try:
        # 1. 加载和清洗
        df = load_and_validate_data(INPUT_CSV)

        # 2. 分析分布
        analyze_distribution(df)

        # 3. 分层抽样划分
        # stratify=df['stratify_label'] 确保训练集/测试集中各类别的比例一致
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            stratify=df['stratify_label'],
            random_state=RANDOM_STATE
        )

        print(f"\n数据集划分完成 (测试集比例 {TEST_SIZE}):")
        print(f"  训练集: {len(train_df)} 样本")
        print(f"  测试集: {len(test_df)} 样本")

        # 4. 保存结果
        train_path = os.path.join(OUTPUT_PATH, 'train_dataset.csv')
        test_path = os.path.join(OUTPUT_PATH, 'test_dataset.csv')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\n文件已保存:")
        print(f"  - 训练集索引: {train_path}")
        print(f"  - 测试集索引: {test_path}")
        print("==================== 模块2 执行完毕 ====================")

    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()