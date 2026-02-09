"""
模块1：数据整理、融合与标签生成（修正版）
解决问题：
1. 适配4层目录结构：根目录 -> 故障类型 -> 传感器类型 -> 数据文件
2. 解��文件名不对应导致的文件缺失问题（改为通过变形百分比匹配）
3. 自动忽略"无变形"文件夹和<5%的样本
"""

import os
import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
DATA_ROOT_PATH = r"C:\\Users\\Alice\\Desktop\\数据处理"  # 根据你的实际路径修改
OUTPUT_PATH = "output"
PROCESSED_DATA_DIR = os.path.join(OUTPUT_PATH, "processed_data")

# 数据参数
START_TIME = 0.02
TIME_STEP = 5e-4
START_INDEX = int(START_TIME / TIME_STEP)  # 40
MIN_DEFORMATION = 5  # 仅处理 >= 5%


def create_directories():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)


def extract_labels_from_folder_name(folder_name):
    """从文件夹名提取标签，如果不是故障文件夹（如'正常'），则返回None"""
    if 'A相' in folder_name:
        phase = 0
    elif 'B相' in folder_name:
        phase = 1
    else:
        return None, None  # 跳过无相别信息的文件夹（如正常样本）

    if '高压' in folder_name:
        voltage = 0
    elif '低压' in folder_name:
        voltage = 1
    else:
        return None, None

    return phase, voltage


def parse_percentage(filename):
    """从文件名解析变形百分比，支持 '...-5%.xlsx' 或 '..._5%.xls' 等格式"""
    # 匹配 -数字% 或 _数字% 或 (数字)%
    match = re.search(r'[-_（\(]?(\d+)%[）\)]?', filename)
    if match:
        return int(match.group(1))
    return None


def find_file_by_percentage(folder_path, target_percent):
    """在指定文件夹中寻找匹配特定百分比的Excel文件"""
    if not os.path.exists(folder_path):
        return None

    for f in os.listdir(folder_path):
        if not (f.endswith('.xlsx') or f.endswith('.xls')):
            continue

        p = parse_percentage(f)
        if p == target_percent:
            return os.path.join(folder_path, f)
    return None


def process_single_sample(curr_path, wind_path, tank_path):
    """读取并融合三个文件"""
    try:
        # 读取数据 (根据你的表头结构调整列索引)
        # 假设所有文件都有时间列在A，数据从B开始
        df_curr = pd.read_excel(curr_path).iloc[START_INDEX:, 1:13].values  # 12列
        df_wind = pd.read_excel(wind_path).iloc[START_INDEX:, 1:7].values  # 6列
        df_tank = pd.read_excel(tank_path).iloc[START_INDEX:, 1:10].values  # 9列

        # 截取最小长度以对齐
        min_len = min(len(df_curr), len(df_wind), len(df_tank))

        # 水平拼接: 12 + 6 + 9 = 27特征
        merged_data = np.hstack([
            df_curr[:min_len],
            df_wind[:min_len],
            df_tank[:min_len]
        ])
        return merged_data
    except Exception as e:
        print(f"    ! 读取错误: {e}")
        return None


def main():
    create_directories()

    # 获取根目录下的一级文件夹（15个分类）
    if not os.path.exists(DATA_ROOT_PATH):
        print(f"路径不存在: {DATA_ROOT_PATH}")
        return

    main_folders = [f for f in os.listdir(DATA_ROOT_PATH) if os.path.isdir(os.path.join(DATA_ROOT_PATH, f))]
    print(f"扫描到 {len(main_folders)} 个一级目录")

    dataset_records = []

    for folder_name in sorted(main_folders):
        # 1. 提取标签，如果提取失败（比如是"正常样本"文件夹），则直接跳过
        phase_label, voltage_label = extract_labels_from_folder_name(folder_name)
        if phase_label is None:
            # print(f"跳过非故障文件夹: {folder_name}")
            continue

        print(f"\n正在处理故障类型: {folder_name}")
        folder_path = os.path.join(DATA_ROOT_PATH, folder_name)

        # 2. 定位内部的3个子文件夹
        sub_dirs = os.listdir(folder_path)
        dir_curr_name = next((d for d in sub_dirs if '电流' in d), None)
        dir_wind_name = next((d for d in sub_dirs if '绕组' in d), None)
        dir_tank_name = next((d for d in sub_dirs if '油箱' in d), None)

        if not (dir_curr_name and dir_wind_name and dir_tank_name):
            print(f"  警告: 缺少子文件夹结构，跳过。")
            continue

        path_curr_root = os.path.join(folder_path, dir_curr_name)
        path_wind_root = os.path.join(folder_path, dir_wind_name)
        path_tank_root = os.path.join(folder_path, dir_tank_name)

        # 3. 遍历电流文件夹中的文件
        curr_files = [f for f in os.listdir(path_curr_root) if f.endswith('.xlsx') or f.endswith('.xls')]

        count = 0
        for f_curr in curr_files:
            # 解析变形程度
            percent = parse_percentage(f_curr)

            # 过滤条件: 解析失败 或 程度小于5%
            if percent is None or percent < MIN_DEFORMATION:
                continue

            # 4. 关键修改：去另外两个文件夹里找“包含相同百分比”的文件
            # 不再依赖文件名完全一致，只依赖百分比一致
            path_curr = os.path.join(path_curr_root, f_curr)
            path_wind = find_file_by_percentage(path_wind_root, percent)
            path_tank = find_file_by_percentage(path_tank_root, percent)

            if not (path_wind and path_tank):
                print(f"  警告: 缺失对应文件 (变形度 {percent}%)")
                continue

            # 5. 处理数据
            data = process_single_sample(path_curr, path_wind, path_tank)
            if data is not None:
                save_name = f"{folder_name}_{percent}%.npy"
                save_path = os.path.join(PROCESSED_DATA_DIR, save_name)
                np.save(save_path, data)

                dataset_records.append({
                    'file_path': save_path,
                    'original_folder': folder_name,
                    'deformation_percent': percent,
                    'phase_binary': phase_label,
                    'voltage_binary': voltage_label
                })
                count += 1

        print(f"  -> 成功处理 {count} 个样本")

    # 保存索引
    if dataset_records:
        df = pd.DataFrame(dataset_records)
        df.to_csv(os.path.join(OUTPUT_PATH, 'labeled_data.csv'), index=False)
        print(f"\n全部完成！共生成 {len(df)} 个样本索引。")
    else:
        print("未生成任何数据，请检查路径。")


if __name__ == '__main__':
    main()