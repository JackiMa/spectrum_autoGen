#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import glob
import time
import h5py
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse
import math
import json
import logging
from typing import List, Dict, Tuple, Any, Optional

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(process)d] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- 常量 ---
# 目标：每个HDF5文件 < 1.8 GB
# 每行数据大小: 20 (列) * 8 (bytes/double) = 160 bytes
# 1.8 GB = 1.8 * 1024 * 1024 * 1024 bytes ≈ 1,932,735,283 bytes
# 每文件最大行数 ≈ 1,932,735,283 / 160 ≈ 12,079,595
# 为了安全和方便取整，使用 10,000,000 行
DEFAULT_MAX_ROWS_PER_HDF5_PART: int = 10_000_000
# 期望的 HDF5 内部块形状
DEFAULT_CHUNK_SHAPE: Tuple[int, int] = (10000, 20) # (chunk_rows, num_layers)
# HDF5 数据集名称
HDF5_DATASET_NAME: str = 'detector_response'
# HDF5 索引文件名
INDEX_JSON_FILENAME: str = "hdf5_index.json"

def get_actual_chunk_shape(max_rows_in_file: int, num_layers: int) -> Tuple[int, int]:
    """根据文件最大行数调整块大小，防止块比文件还大"""
    chunk_rows = min(DEFAULT_CHUNK_SHAPE[0], max_rows_in_file)
    return (chunk_rows, num_layers)

def convert_energy_level_to_hdf5(task_info: Tuple[str, List[str], str, int, int]) -> Optional[Dict[str, Any]]:
    """
    处理单个能量级别的所有CSV文件，将其转换为HDF5格式（可能分多个部分文件）。
    在一个单独的进程中运行。

    Args:
        task_info: Tuple 包含:
            energy_level_name (str): 能量级别名称 (e.g., "0.1MeV")
            csv_files (List[str]): 属于该能量级别的CSV文件路径列表
            output_base_dir (str): 输出 HDF5 文件的根目录
            num_layers (int): 探测器层数 (列数)
            max_rows_per_part (int): 每个 HDF5 部分文件的最大行数

    Returns:
        Optional[Dict[str, Any]]: 包含该能量级别处理结果的字典，如果处理失败则返回 None。
            {
                'energy_level': str,
                'total_rows': int,
                'parts': List[Dict[str, Any]] # 每个部分文件的信息
                    [{'path': str, 'rows': int, 'part_num': int}]
            }
    """
    energy_level_name, csv_files, output_base_dir, num_layers, max_rows_per_part = task_info
    pid = os.getpid()
    logger = logging.getLogger() # 获取根logger

    logger.info(f"开始处理能量级别: {energy_level_name} ({len(csv_files)} 个 CSV 文件)")
    energy_start_time = time.time()

    h5_part_info_list: List[Dict[str, Any]] = [] # 存储生成的部分文件信息
    current_h5_file_path: Optional[str] = None
    current_h5_file: Optional[h5py.File] = None
    current_dataset: Optional[h5py.Dataset] = None
    rows_in_current_part: int = 0
    part_number: int = 0 # 从0开始编号
    total_rows_processed_for_level: int = 0

    # 确保输出子目录存在 (e.g., output_dir/0.1MeV/)
    energy_output_dir = os.path.join(output_base_dir, energy_level_name)
    try:
        os.makedirs(energy_output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"无法创建输出目录 {energy_output_dir}: {e}")
        return None

    # --- 文件处理主循环 ---
    try:
        for csv_file_index, csv_file in enumerate(csv_files):
            logger.info(f"处理 CSV: {os.path.basename(csv_file)} ({csv_file_index+1}/{len(csv_files)}) for {energy_level_name}")
            csv_start_time = time.time()
            rows_processed_in_csv = 0
            csv_skipped = False

            try:
                # 使用分块读取CSV，指定类型以提高效率，处理坏行
                chunk_size = DEFAULT_CHUNK_SHAPE[0] # 读取CSV的块大小
                for chunk_index, chunk in enumerate(pd.read_csv(
                        csv_file,
                        skiprows=1,          # 跳过表头
                        header=None,         # 无表头
                        chunksize=chunk_size,
                        dtype=np.float64,    # 指定数据类型
                        on_bad_lines='warn', # 对坏行发出警告
                        low_memory=False     # 尝试优化内存使用
                )):

                    # 验证列数
                    if chunk.shape[1] != num_layers:
                        logger.error(f"文件 {csv_file} 的块 {chunk_index} 有 {chunk.shape[1]} 列，期望 {num_layers} 列。跳过此文件的剩余部分。")
                        csv_skipped = True
                        break # 跳出内部 chunk 循环，处理下一个 CSV

                    rows_in_chunk = chunk.shape[0]
                    if rows_in_chunk == 0: continue # 跳过空块

                    chunk_data = chunk.values
                    processed_in_chunk = 0

                    # --- 将当前CSV块写入一个或多个HDF5文件 ---
                    while processed_in_chunk < rows_in_chunk:
                        # 检查是否需要新的 HDF5 部分文件
                        if current_h5_file is None or rows_in_current_part >= max_rows_per_part:
                            if current_h5_file is not None:
                                # 关闭旧文件，记录信息
                                logger.info(f"HDF5 文件 {os.path.basename(current_h5_file_path)} 已满 ({rows_in_current_part} 行)。正在关闭...")
                                # 在关闭前存储元数据
                                current_dataset.attrs['rows_in_this_part'] = rows_in_current_part
                                current_h5_file.close()
                                h5_part_info_list.append({
                                    'path': current_h5_file_path,
                                    'rows': rows_in_current_part,
                                    'part_num': part_number
                                })
                                part_number += 1

                            # 创建新的 HDF5 文件名和文件对象
                            base_name = f"e_{float(energy_level_name.replace('MeV', '')):.2f}MeV"
                            current_h5_file_path = os.path.join(energy_output_dir, f"{base_name}_part_{part_number:04d}.hdf5") # 使用4位补零
                            logger.info(f"打开新的 HDF5 文件: {os.path.basename(current_h5_file_path)}")

                            current_h5_file = h5py.File(current_h5_file_path, 'w')
                            actual_chunk_shape = get_actual_chunk_shape(max_rows_per_part, num_layers)
                            # 创建可调整大小的数据集
                            current_dataset = current_h5_file.create_dataset(
                                HDF5_DATASET_NAME,
                                shape=(0, num_layers),           # 初始大小为0
                                maxshape=(None, num_layers),     # 行数无限
                                dtype='float64',                 # 使用 float64
                                chunks=actual_chunk_shape,       # 使用计算出的块大小
                                compression="gzip",              # 启用压缩
                                compression_opts=8               # Gzip 压缩级别 (0-9)
                            )
                            # 存储元数据 (属性)
                            current_dataset.attrs['energy_level'] = energy_level_name
                            current_dataset.attrs['part_number'] = part_number
                            current_dataset.attrs['columns'] = np.arange(1, num_layers + 1).tolist() # 探测器层数/列名
                            # 其他元数据可以按需添加，例如来源CSV列表等
                            # current_dataset.attrs['source_csvs'] = csv_files # 如果列表不长的话

                            rows_in_current_part = 0 # 重置计数器

                        # 计算可以写入当前 HDF5 文件的数据量
                        rows_can_write_to_current = max_rows_per_part - rows_in_current_part
                        rows_left_in_chunk = rows_in_chunk - processed_in_chunk
                        rows_to_write = min(rows_left_in_chunk, rows_can_write_to_current)

                        if rows_to_write <= 0: break # 不应发生，但作为保护

                        # 获取要写入的数据切片
                        data_to_write = chunk_data[processed_in_chunk : processed_in_chunk + rows_to_write]

                        # 调整数据集大小并写入数据
                        new_size = rows_in_current_part + rows_to_write
                        current_dataset.resize(new_size, axis=0)
                        current_dataset[rows_in_current_part : new_size] = data_to_write

                        # 更新计数器
                        rows_in_current_part += rows_to_write
                        processed_in_chunk += rows_to_write
                        total_rows_processed_for_level += rows_to_write
                        rows_processed_in_csv += rows_to_write

                    if csv_skipped: break # 如果当前CSV文件因错误被跳过

            except FileNotFoundError:
                logger.error(f"CSV 文件未找到: {csv_file}。跳过此文件。")
                continue # 处理下一个 CSV 文件
            except pd.errors.EmptyDataError:
                logger.warning(f"CSV 文件为空或只有表头: {csv_file}。跳过此文件。")
                continue
            except Exception as e:
                logger.exception(f"处理 CSV 文件 {csv_file} 时发生未知错误: {e}。跳过此文件。")
                continue # 尝试继续处理下一个CSV

            if not csv_skipped:
                csv_time = time.time() - csv_start_time
                logger.info(f"  完成 CSV: {os.path.basename(csv_file)} ({rows_processed_in_csv:,} 行写入 HDF5), 耗时: {csv_time:.2f}s")

        # --- 清理 ---
        # 处理完一个能量级别的所有 CSV 后，关闭最后一个打开的 HDF5 文件
        if current_h5_file is not None:
            logger.info(f"关闭最后一个 HDF5 文件 {os.path.basename(current_h5_file_path)} ({rows_in_current_part} 行)。")
            # 更新最后一个文件的行数属性
            current_dataset.attrs['rows_in_this_part'] = rows_in_current_part
            current_h5_file.close()
            h5_part_info_list.append({
                'path': current_h5_file_path,
                'rows': rows_in_current_part,
                'part_num': part_number
            })

    except Exception as e:
        logger.exception(f"处理能量级别 {energy_level_name} 时发生严重意外错误: {e}")
        # 尝试关闭可能打开的文件句柄
        if current_h5_file is not None and not current_h5_file.id.__bool__(): # Check if file is open
             try:
                 current_h5_file.close()
                 logger.warning(f"发生错误后，文件 {current_h5_file_path} 可能未完全写入或已关闭。")
             except Exception as ce:
                 logger.error(f"关闭出错的文件 {current_h5_file_path} 时再次发生错误: {ce}")
        return None # 返回 None 表示失败

    energy_time = time.time() - energy_start_time
    logger.info(f"完成能量级别: {energy_level_name} ({total_rows_processed_for_level:,} 总行写入 HDF5), "
                f"生成 {len(h5_part_info_list)} 个 HDF5 文件, 耗时: {energy_time:.2f}s")

    # 返回包含详细信息的字典
    return {
        'energy_level': energy_level_name,
        'total_rows': total_rows_processed_for_level,
        'parts': h5_part_info_list
    }

def main():
    """主函数，负责扫描、分发任务、收集结果并生成索引文件"""
    parser = argparse.ArgumentParser(description="将能谱CSV数据并行转换为分块HDF5格式（按能量级别），并按大小拆分文件。")
    parser.add_argument("--input_dir", required=True, help="包含 *MeV 子文件夹的输入数据目录")
    parser.add_argument("--output_dir", required=True, help="输出 HDF5 文件和索引文件的根目录")
    parser.add_argument("--num_layers", type=int, default=20, help="探测器层数 (CSV文件的列数)，默认为 20")
    parser.add_argument("--max_rows_per_part", type=int, default=DEFAULT_MAX_ROWS_PER_HDF5_PART,
                        help=f"每个 HDF5 部分文件的最大行数，默认为 {DEFAULT_MAX_ROWS_PER_HDF5_PART:,} (约 1.6GB)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2), # 默认使用一半CPU核心
                        help="用于并行处理能量级别的最大工作进程数")

    args = parser.parse_args()
    logger = logging.getLogger()

    # --- 1. 发现并分组 CSV 文件 ---
    logger.info(f"正在扫描输入目录 {args.input_dir} 以查找能量级别文件夹和CSV文件...")
    all_csv_files_by_energy: Dict[str, List[str]] = {}
    # 匹配像 0.1MeV, 1.5MeV, 10MeV 这样的文件夹
    energy_folders = glob.glob(os.path.join(args.input_dir, "[0-9.]*MeV"))

    if not energy_folders:
        logger.error(f"在 {args.input_dir} 中找不到任何符合 '*MeV' 格式的能量级别子文件夹。请检查路径和文件夹命名。")
        sys.exit(1)

    total_csv_count = 0
    for folder in sorted(energy_folders): # 按能量排序文件夹
        if not os.path.isdir(folder):
            continue
        energy_name = os.path.basename(folder)
        # 匹配 e_开头，后面是数字和点，然后是MeV，可选数字后缀，最后是.csv
        # 例如 e_0.10MeV.csv, e_1.5MeV1.csv, e_10MeV_extra.csv (注意：这个glob可能需要根据实际情况调整)
        # 更简单的匹配：所有 .csv 文件
        csv_pattern = os.path.join(folder, "*.csv")
        csv_files = sorted(glob.glob(csv_pattern)) # 在文件夹内也排序，确保处理顺序一致

        if csv_files:
            all_csv_files_by_energy[energy_name] = csv_files
            total_csv_count += len(csv_files)
            logger.info(f"  找到能量级别 '{energy_name}' 包含 {len(csv_files)} 个 CSV 文件。")
        else:
            logger.warning(f"能量级别文件夹 {folder} 中没有找到 .csv 文件。")

    if not all_csv_files_by_energy:
        logger.error(f"在 {args.input_dir} 的任何 *MeV 文件夹中都没有找到有效的 .csv 文件。")
        sys.exit(1)

    num_energy_levels = len(all_csv_files_by_energy)
    logger.info(f"共找到 {num_energy_levels} 个能量级别，合计 {total_csv_count} 个 CSV 文件。")

    # --- 2. 准备并行任务 ---
    tasks: List[Tuple[str, List[str], str, int, int]] = []
    for energy_name, csv_list in all_csv_files_by_energy.items():
        tasks.append((energy_name, csv_list, args.output_dir, args.num_layers, args.max_rows_per_part))

    # --- 3. 执行并行转换 ---
    num_workers = min(num_energy_levels, args.workers, multiprocessing.cpu_count())
    logger.info(f"\n开始使用最多 {num_workers} 个进程并行转换 {num_energy_levels} 个能量级别...")
    total_start_time = time.time()
    results_from_workers: List[Optional[Dict[str, Any]]] = []

    try:
        # 使用 ProcessPoolExecutor 来管理进程池
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # map 会保持任务提交的顺序，结果也会按顺序返回
            results_iterator = executor.map(convert_energy_level_to_hdf5, tasks)
            # 收集所有结果（包括可能失败的 None）
            results_from_workers = list(results_iterator)

    except Exception as e:
        logger.exception(f"\n并行处理过程中发生严重错误: {e}")
        logger.error("转换可能未完全完成或出错。请检查之前的日志。")
        # 可以在这里添加清理逻辑，例如删除部分生成的文件

    # --- 4. 收集结果并生成 index.json ---
    logger.info("所有工作进程已完成，正在整理结果并生成索引文件...")
    final_index_data: Dict[str, Dict[str, Any]] = {}
    total_hdf5_files_generated = 0
    successful_levels = 0

    for result in results_from_workers:
        if result is not None and result['parts']: # 检查任务是否成功且生成了文件
            energy_level = result['energy_level']
            # 将文件路径转换为相对于 output_dir 的相对路径，以便索引文件更具可移植性
            relative_parts = []
            for part_info in result['parts']:
                try:
                    relative_path = os.path.relpath(part_info['path'], args.output_dir)
                    relative_parts.append({
                        'path': relative_path,
                        'rows': part_info['rows'],
                        'part_num': part_info['part_num']
                    })
                except ValueError: # 如果路径不在同一个驱动器上，无法生成相对路径
                     logger.warning(f"无法为 {part_info['path']} 生成相对于 {args.output_dir} 的相对路径，将使用绝对路径。")
                     relative_parts.append(part_info.copy()) # 使用原始绝对路径

            final_index_data[energy_level] = {
                'total_rows_in_bin': result['total_rows'],
                'rows_per_part_nominal': args.max_rows_per_part, # 记录配置的最大行数
                'chunk_shape': list(get_actual_chunk_shape(args.max_rows_per_part, args.num_layers)), # 记录使用的块形状
                'dataset_name': HDF5_DATASET_NAME, # 记录数据集名称
                'parts': relative_parts
            }
            total_hdf5_files_generated += len(result['parts'])
            successful_levels += 1
        elif result is None:
            # 可以在这里记录哪个能量级别处理失败
            logger.error("检测到一个或多个能量级别处理失败（返回值为 None）。")
        else: # result is not None but result['parts'] is empty
             logger.warning(f"能量级别 {result.get('energy_level', '未知')} 处理完成，但没有生成任何 HDF5 文件（可能所有CSV都为空或出错）。")


    # 将索引数据写入 JSON 文件
    index_file_path = os.path.join(args.output_dir, INDEX_JSON_FILENAME)
    if final_index_data:
        try:
            with open(index_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_index_data, f, indent=4, ensure_ascii=False)
            logger.info(f"索引文件已生成: {index_file_path}")
        except IOError as e:
            logger.error(f"无法写入索引文件 {index_file_path}: {e}")
        except TypeError as e:
             logger.error(f"无法将索引数据序列化为 JSON: {e}")
    else:
        logger.warning("没有成功的转换结果，未生成索引文件。")


    # --- 5. 结束 ---
    total_time = time.time() - total_start_time
    logger.info(f"\n--- 转换统计 ---")
    logger.info(f"成功处理了 {successful_levels} / {num_energy_levels} 个能量级别。")
    logger.info(f"总共生成了 {total_hdf5_files_generated} 个 HDF5 文件。")
    logger.info(f"转换后的数据存储在: {os.path.abspath(args.output_dir)}")
    if final_index_data:
        logger.info(f"详细文件列表和元数据见: {index_file_path}")
    logger.info(f"总耗时: {total_time:.2f} 秒")

    if successful_levels < num_energy_levels:
         logger.warning("部分能量级别处理失败或未生成文件，请仔细检查上面的日志输出。")
         # sys.exit(1) # 可以选择在有失败时退出并返回错误码

if __name__ == "__main__":
    # 可以在这里添加检查 h5py 和 pandas 是否安装
    try:
        import h5py
        import pandas as pd
    except ImportError as e:
        print(f"错误: 缺少必要的库: {e}. 请安装 h5py 和 pandas.", file=sys.stderr)
        sys.exit(1)
    main()