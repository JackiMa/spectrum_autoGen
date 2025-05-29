#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ========================================
# 配置常量 (在脚本顶部集中配置)
# ========================================

# 数据路径配置
INPUT_DATA_DIR = r"sim_data_hd5_V2"  # 输入数据根目录
TARGET_ENERGY_LIST = ["0.1MeV", "0.5MeV", "1.0MeV"]  # 目标能量点列表
OUTPUT_BASE_DIR = "数据集/JiTai_Data"  # 输出根目录

# 数据处理配置
NUM_ROWS_PER_DATASET = 10000000  # 每个数据集使用的行数

# 运行配置
SIMULATION_RUN_NAME = "JiTai_Run"  # 模拟运行名称

# HDF5文件配置
HDF5_DATASET_NAME = 'detector_response'  # HDF5数据集名称
DEFAULT_CHUNK_SIZE = 10000  # 读取块大小

# 固定常量
NUM_DETECTOR_LAYERS = 20  # 探测器层数（固定为20）

# 输出格式配置 - 完整的能量点范围
ALL_ENERGY_POINTS = [
    "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0",
    "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0"
]  # 完整的能量点列表（用于输出格式）

# ========================================
# 导入库
# ========================================
import os
import sys
import time
import h5py
import numpy as np
import logging
import glob
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# 配置日志 - 只显示WARNING及以上级别，避免混乱
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ========================================
# 全局变量用于进度条状态
# ========================================
progress_lock = threading.Lock()
current_file_status: Dict[str, Dict] = {}

# ========================================
# 主要函数
# ========================================

def find_hdf5_files(energy_dir: str) -> List[str]:
    """查找指定能量点目录下的所有HDF5文件"""
    hdf5_pattern = os.path.join(energy_dir, "*.h5")
    files = glob.glob(hdf5_pattern)
    if not files:
        hdf5_pattern = os.path.join(energy_dir, "*.hdf5")
        files = glob.glob(hdf5_pattern)
    
    files.sort()  # 确保顺序一致
    return files

def get_hdf5_info(file_path: str) -> Tuple[int, int]:
    """获取HDF5文件的行数和列数信息"""
    try:
        with h5py.File(file_path, 'r') as f:
            if HDF5_DATASET_NAME not in f:
                return 0, 0
            
            dataset = f[HDF5_DATASET_NAME]
            rows, cols = dataset.shape
            return rows, cols
    except Exception as e:
        return 0, 0

def update_file_progress(filename: str, current: int, total: int, status: str = "processing"):
    """更新文件处理进度"""
    with progress_lock:
        current_file_status[filename] = {
            'current': current,
            'total': total,
            'status': status,
            'progress': current / total if total > 0 else 0
        }

def process_single_hdf5_file(file_path: str, target_rows: int) -> Optional[np.ndarray]:
    """处理单个HDF5文件，读取指定行数的数据"""
    filename = os.path.basename(file_path)
    
    try:
        with h5py.File(file_path, 'r') as f:
            if HDF5_DATASET_NAME not in f:
                update_file_progress(filename, 0, 1, "error")
                return None
            
            dataset = f[HDF5_DATASET_NAME]
            total_rows, cols = dataset.shape
            
            if total_rows == 0:
                update_file_progress(filename, 0, 1, "empty")
                return None
            
            # 确定实际读取的行数
            if target_rows > total_rows:
                actual_rows = total_rows
            else:
                actual_rows = target_rows
            
            # 分块读取数据以节省内存
            collected_chunks = []
            rows_read = 0
            
            update_file_progress(filename, 0, actual_rows, "processing")
            
            while rows_read < actual_rows:
                chunk_size = min(DEFAULT_CHUNK_SIZE, actual_rows - rows_read)
                chunk_data = dataset[rows_read:rows_read + chunk_size, :]
                collected_chunks.append(chunk_data)
                rows_read += chunk_size
                
                # 更新进度
                update_file_progress(filename, rows_read, actual_rows, "processing")
            
            # 合并数据块
            if collected_chunks:
                result = np.vstack(collected_chunks)
                update_file_progress(filename, actual_rows, actual_rows, "completed")
                return result
            else:
                update_file_progress(filename, 0, 1, "error")
                return None
                
    except Exception as e:
        update_file_progress(filename, 0, 1, "error")
        return None

def collect_data_parallel(hdf5_files: List[str], rows_per_file: int) -> Optional[np.ndarray]:
    """并行处理多个HDF5文件并收集数据"""
    if not hdf5_files:
        print("错误: 没有可处理的HDF5文件")
        return None
    
    # 确定线程数
    max_workers = min(os.cpu_count() - 2, len(hdf5_files))
    max_workers = max(1, max_workers)  # 至少使用1个线程
    
    print(f"使用 {max_workers} 个线程并行处理 {len(hdf5_files)} 个文件")
    print(f"每个文件目标读取行数: {rows_per_file:,}")
    
    # 初始化文件状态
    global current_file_status
    current_file_status = {}
    for file_path in hdf5_files:
        filename = os.path.basename(file_path)
        current_file_status[filename] = {
            'current': 0,
            'total': rows_per_file,
            'status': 'waiting',
            'progress': 0
        }
    
    collected_data = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_file = {
            executor.submit(process_single_hdf5_file, file_path, rows_per_file): file_path 
            for file_path in hdf5_files
        }
        
        # 创建主进度条
        with tqdm(total=len(hdf5_files), desc="处理进度", unit="file", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as main_pbar:
            
            completed = 0
            
            # 启动进度显示更新线程
            def update_progress_display():
                while completed < len(hdf5_files):
                    time.sleep(0.5)  # 每0.5秒更新一次
                    
                    with progress_lock:
                        # 找到当前正在处理的文件
                        active_files = []
                        for filename, status in current_file_status.items():
                            if status['status'] == 'processing':
                                progress_pct = status['progress'] * 100
                                active_files.append(f"{filename}({progress_pct:.0f}%)")
                        
                        if active_files:
                            # 更新进度条描述，显示正在处理的文件
                            desc = f"处理中: {', '.join(active_files[:3])}"  # 最多显示3个文件
                            if len(active_files) > 3:
                                desc += f" +{len(active_files)-3}个"
                            main_pbar.set_description(desc[:80])  # 限制长度
            
            # 启动进度更新线程
            progress_thread = threading.Thread(target=update_progress_display, daemon=True)
            progress_thread.start()
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                filename = os.path.basename(file_path)
                
                try:
                    result = future.result()
                    if result is not None:
                        collected_data.append(result)
                    
                    completed += 1
                    main_pbar.update(1)
                    
                except Exception as e:
                    completed += 1
                    main_pbar.update(1)
            
            main_pbar.set_description("处理完成")
    
    if not collected_data:
        print("错误: 没有成功处理任何文件")
        return None
    
    # 合并所有文件的数据
    print(f"合并来自 {len(collected_data)} 个文件的数据...")
    with tqdm(desc="合并数据", unit="步骤") as merge_pbar:
        final_data = np.vstack(collected_data)
        merge_pbar.update(1)
    
    total_rows = final_data.shape[0]
    total_cols = final_data.shape[1]
    print(f"成功合并数据: {total_rows:,} 行 × {total_cols} 列")
    
    return final_data

def save_output_file(output_dir: str, detector_response: np.ndarray, 
                    particles_count: int, energy_value: str) -> None:
    """保存输出文件，格式与原脚本保持一致"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = time.strftime('%Y%m%d%H%M%S')
        filename = f"{SIMULATION_RUN_NAME}_{timestamp}_event_000.txt"
        filepath = os.path.join(output_dir, filename)
        
        # 准备各行内容
        # 第一行：完整的能量点列表
        line1 = ", ".join(ALL_ENERGY_POINTS)
        
        # 第二行：对应的粒子数（只有当前能量点有数据，其他为0）
        current_energy_numeric = energy_value.replace('MeV', '')
        particle_counts = []
        for energy_point in ALL_ENERGY_POINTS:
            if energy_point == current_energy_numeric:
                particle_counts.append(str(particles_count))
            else:
                particle_counts.append("0")
        line2 = ", ".join(particle_counts)
        
        # 第三行：空行
        line3 = ""
        
        # 第四行：空行
        line4 = ""
        
        # 第五行：探测器层编号
        line5 = ", ".join(map(str, range(1, NUM_DETECTOR_LAYERS + 1)))
        
        # 第六行：探测器响应数据
        line6 = np.array2string(
            detector_response, 
            separator=', ', 
            formatter={'float_kind': lambda x: f"{x:.8e}"},
            threshold=np.inf,
            max_line_width=np.inf
        ).replace('[', '').replace(']', '')
        
        # 写入文件
        content = f"{line1}\n{line2}\n{line3}\n{line4}\n{line5}\n{line6}"
        
        with tqdm(desc="保存文件", unit="文件") as save_pbar:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            save_pbar.update(1)
        
        print(f"✓ 已保存输出文件: {filename}")
        
    except Exception as e:
        print(f"错误: 保存输出文件时出错: {e}")

def process_single_energy(target_energy: str) -> bool:
    """处理单个能量点"""
    print(f"\n{'='*60}")
    print(f"处理能量点: {target_energy}")
    print(f"{'='*60}")
    
    try:
        # 1. 构建输入和输出路径
        energy_input_dir = os.path.join(INPUT_DATA_DIR, target_energy)
        output_dir = os.path.join(OUTPUT_BASE_DIR, target_energy)
        
        print(f"输入目录: {energy_input_dir}")
        print(f"输出目录: {output_dir}")
        
        # 2. 检查输入目录
        if not os.path.exists(energy_input_dir):
            print(f"错误: 输入目录不存在: {energy_input_dir}")
            return False
        
        # 3. 查找HDF5文件
        hdf5_files = find_hdf5_files(energy_input_dir)
        if not hdf5_files:
            print(f"错误: 在 {energy_input_dir} 中未找到HDF5文件")
            return False
        
        print(f"找到 {len(hdf5_files)} 个HDF5文件")
        
        # 4. 计算每个文件需要读取的行数
        num_files = len(hdf5_files)
        rows_per_file = NUM_ROWS_PER_DATASET // num_files
        
        if rows_per_file == 0:
            rows_per_file = 1
            print(f"警告: 目标行数太少，每个文件将只读取1行")
        
        # 5. 并行收集数据
        all_data = collect_data_parallel(hdf5_files, rows_per_file)
        if all_data is None:
            print("错误: 数据收集失败")
            return False
        
        # 6. 检查数据维度
        if all_data.shape[1] != NUM_DETECTOR_LAYERS:
            print(f"警告: 数据列数 ({all_data.shape[1]}) 与探测器层数 ({NUM_DETECTOR_LAYERS}) 不匹配")
            # 调整配置或截断数据
            if all_data.shape[1] < NUM_DETECTOR_LAYERS:
                print("错误: 数据列数少于探测器层数，无法继续处理")
                return False
            else:
                print(f"截断数据到前 {NUM_DETECTOR_LAYERS} 列")
                all_data = all_data[:, :NUM_DETECTOR_LAYERS]
        
        # 7. 计算探测器响应
        print("计算探测器响应...")
        actual_rows = all_data.shape[0]
        
        with tqdm(desc="计算响应", unit="步骤") as calc_pbar:
            detector_response = np.sum(all_data, axis=0, dtype=np.float64)
            calc_pbar.update(1)
        
        # 8. 保存输出文件
        save_output_file(output_dir, detector_response, actual_rows, target_energy)
        
        print(f"✓ {target_energy} 处理完成! 实际使用了 {actual_rows:,} 行数据")
        return True
        
    except Exception as e:
        print(f"错误: 处理 {target_energy} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("JiTai数据生成器 - 简化版本（并行处理）")
    print("=" * 60)
    print(f"目标能量点: {TARGET_ENERGY_LIST}")
    print(f"每个数据集行数: {NUM_ROWS_PER_DATASET:,}")
    print(f"探测器层数: {NUM_DETECTOR_LAYERS}")
    print(f"可用CPU核心数: {os.cpu_count()}")
    
    total_start_time = time.time()
    successful_count = 0
    failed_energies = []
    
    try:
        for i, target_energy in enumerate(TARGET_ENERGY_LIST):
            print(f"\n进度: [{i+1}/{len(TARGET_ENERGY_LIST)}]")
            
            energy_start_time = time.time()
            success = process_single_energy(target_energy)
            energy_time = time.time() - energy_start_time
            
            if success:
                successful_count += 1
                print(f"✓ {target_energy} 完成，耗时: {energy_time:.2f} 秒")
            else:
                failed_energies.append(target_energy)
                print(f"✗ {target_energy} 处理失败")
        
        # 9. 总结
        total_time = time.time() - total_start_time
        print("\n" + "=" * 60)
        print("所有处理完成!")
        print(f"成功处理: {successful_count}/{len(TARGET_ENERGY_LIST)} 个能量点")
        if failed_energies:
            print(f"失败的能量点: {failed_energies}")
        print(f"总耗时: {total_time:.2f} 秒")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n用户中断了程序")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 检查必要的库
    try:
        import h5py
        import numpy
        from tqdm import tqdm
    except ImportError as e:
        print(f"错误: 缺少必要的库: {e}. 请确保 h5py, numpy, tqdm 已安装。", file=sys.stderr)
        print("安装命令: pip install h5py numpy tqdm", file=sys.stderr)
        sys.exit(1)
    
    main() 