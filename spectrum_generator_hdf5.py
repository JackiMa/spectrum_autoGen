#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import h5py
import numpy as np
import argparse
import logging
import pickle
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional, Union
from numpy.random import Generator, PCG64, SeedSequence
from create_sampling_plan import *
import concurrent.futures  # 添加导入
# --- 常量 ---
CHECKPOINT_FILENAME: str = "sampling_checkpoint.pkl"
HDF5_DATASET_NAME: str = 'detector_response'
DEFAULT_CHUNK_SHAPE: Tuple[int, int] = (10000, 20)
MIN_SAMPLING_RATIO = 0.3  # 块最低抽样效率阈值 (1/10)，值越大，效率越高，但随机性下降
# --- 配置日志 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(process)d] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def save_checkpoint(filepath: str, state: Dict[str, Any]):
    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, 'wb') as f: pickle.dump(state, f)
        os.replace(temp_filepath, filepath)
        logger.info(f"检查点已保存: {filepath}")
    except Exception as e:
        logger.exception(f"保存检查点失败 ({filepath})")
        if os.path.exists(temp_filepath):
            try: os.remove(temp_filepath)
            except OSError: pass

def load_checkpoint(filepath: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(filepath):
        logger.info(f"发现检查点文件，尝试加载: {filepath}")
        try:
            with open(filepath, 'rb') as f: state = pickle.load(f)
            required_keys = ['event_sums', 'event_rows_collected', 'last_completed_bin_index', 'original_num_events', 'event_id_map']
            if all(key in state for key in required_keys):
                logger.info("检查点加载成功。"); return state
            else:
                logger.warning(f"检查点文件缺少必要的键 ({required_keys})，将忽略。"); return None
        except Exception as e:
            logger.exception(f"加载检查点失败 ({filepath})。将从头开始。"); return None
    else:
        logger.info("未找到检查点文件，将从头开始。"); return None

def save_output_file(output_dir: str, run_name: str, event_id: int,
                     plan: Dict[str, Any], final_sum: np.ndarray,
                     actual_counts_per_bin: Dict[str, int], num_layers: int,
                     num_total_events: int, all_energy_bins: List[str] = None, config: Dict[str, Any] = None):
    padding_width = len(str(num_total_events)); filename = f"{run_name}_{time.strftime('%Y%m%d%H%M%S')}_event_{event_id:0{padding_width}d}.txt"; filepath = os.path.join(output_dir, filename)
    try:
        # 不再跳过空的actual_counts_per_bin，而是输出所有能量点
        if all_energy_bins is None or len(all_energy_bins) == 0:
            # 如果没有提供所有能量点，则使用实际的能量点
            if not actual_counts_per_bin:
                logger.warning(f"事件 {event_id} 没有能量点数据且未提供所有能量点列表，跳过。")
                return
            energy_bins = sorted(actual_counts_per_bin.keys(), key=lambda x: float(x.replace('MeV','')))
        else:
            # 使用提供的所有能量点列表
            energy_bins = sorted(all_energy_bins, key=lambda x: float(x.replace('MeV','')))
            
        if not energy_bins:
            logger.warning(f"事件 {event_id} 没有有效的能量点，跳过。")
            return
            
        # 生成不带MeV的能量数值字符串列表
        numeric_energies_str = [energy.replace('MeV', '') for energy in energy_bins]
        
        # 第一行：所有能量点
        line1 = ", ".join(numeric_energies_str)
        
        # 第二行：每个能量点对应的粒子数，如果没有则为0
        line2 = ", ".join(str(actual_counts_per_bin.get(energy, 0)) for energy in energy_bins)
        
        # 第三行：orbit data
        line3 = config['orbit']
        
        # 第四行：random number, 均匀分布随机数，整数
        random_range = config['value_range']
        line4 = str(int(np.random.uniform(random_range[0], random_range[1])))
        
        
        # 第五行：detector response
        line5 = ", ".join(map(str, range(1, num_layers + 1)))
        line6 = np.array2string(final_sum, separator=', ', formatter={'float_kind':lambda x: f"{x:.8e}"},threshold=np.inf,max_line_width=np.inf).replace('[','').replace(']','')
        
        content = f"{line1}\n{line2}\n{line3}\n{line4}\n{line5}\n{line6}"
        logger.debug(f"DEBUG SAVE: 事件 {event_id}, 写入第二行: {line2}") 
        with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
        logger.info(f"已保存事件 {event_id} 的输出文件: {filename}")
    except Exception as e: logger.exception(f"写入输出文件 {filepath} 时发生未知错误")


# --- Worker Function (使用确定性的 n_to_sample) ---
def sample_chunk_task(task_info: Dict[str, Any]) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """工作进程执行的函数：处理单个数据块的抽样任务。(接收确定性的目标数)"""
    part_file_path = task_info['part_file_path']
    chunk_start = task_info['chunk_start_row']
    chunk_end = task_info['chunk_end_row']
    sampling_needs_map = task_info['sampling_needs'] # {event_idx: n_to_sample_deterministic}
    dataset_name = task_info['dataset_name']
    rng_seed = task_info['rng_seed']

    rng = Generator(PCG64(rng_seed))
    partial_sums: Dict[int, np.ndarray] = {}
    actual_counts: Dict[int, int] = {}

    try:
        with h5py.File(part_file_path, 'r') as f:
            if dataset_name not in f:
                 print(f"ERROR WORKER {os.getpid()}: Dataset '{dataset_name}' not found in {part_file_path}", file=sys.stderr, flush=True)
                 return {}, {}
            dset = f[dataset_name]
            dset_shape = dset.shape
            if chunk_start >= dset_shape[0]: return {}, {}
            actual_chunk_end = min(chunk_end, dset_shape[0])
            if chunk_start >= actual_chunk_end: return {}, {}
            chunk_data = dset[chunk_start:actual_chunk_end, :]
            rows_in_chunk = chunk_data.shape[0]
            if rows_in_chunk == 0: return {}, {}

            for event_idx, n_to_sample in sampling_needs_map.items():
                if n_to_sample <= 0: continue
                actual_n_to_sample = min(n_to_sample, rows_in_chunk)
                if actual_n_to_sample <= 0: continue

                if actual_n_to_sample == rows_in_chunk:
                    indices = np.arange(rows_in_chunk)
                else:
                    indices = rng.choice(rows_in_chunk, size=actual_n_to_sample, replace=False)

                if indices.size == 0 and actual_n_to_sample > 0:
                     print(f"WARN WORKER {os.getpid()}: indices empty E:{event_idx} C:[{chunk_start}-{actual_chunk_end}] actual_n:{actual_n_to_sample}", flush=True)
                     continue

                try:
                    selected_rows = chunk_data[indices, :]
                    partial_sums[event_idx] = np.sum(selected_rows, axis=0, dtype=np.float64)
                    # IMPORTANT: Return the number of rows ACTUALLY sampled, which IS actual_n_to_sample
                    actual_counts[event_idx] = actual_n_to_sample
                except IndexError as ie:
                     print(f"ERROR WORKER {os.getpid()}: IndexError E:{event_idx} C:[{chunk_start}-{actual_chunk_end}] {ie}", flush=True)
                     continue

    except FileNotFoundError: print(f"ERROR WORKER {os.getpid()}: HDF5 File Not Found - {part_file_path}", file=sys.stderr, flush=True); return {}, {}
    except KeyError: print(f"ERROR WORKER {os.getpid()}: Dataset '{dataset_name}' not found in {part_file_path}", file=sys.stderr, flush=True); return {}, {}
    except ValueError as e: print(f"ERROR WORKER {os.getpid()}: ValueError processing chunk {chunk_start}-{chunk_end} in {part_file_path}: {e}", file=sys.stderr, flush=True); return {}, {}
    except Exception as e: print(f"ERROR WORKER {os.getpid()}: Unhandled Exception processing chunk {chunk_start}-{chunk_end} in {part_file_path}: {type(e).__name__} - {e}", file=sys.stderr, flush=True); import traceback; traceback.print_exc(file=sys.stderr); return {}, {}

    return partial_sums, actual_counts
# --- End Worker Function ---


# --- 阶段 2: 执行抽样任务 (OOP Class) ---
class SamplingExecutor:
    """负责执行抽样、管理状态和并行处理"""
    def __init__(self, run_info: Dict[str, Any], num_workers: int, resume: bool, force_rerun: bool):
        self.config = run_info['config']; self.hdf5_index = run_info['hdf5_index']; self.sampling_plans = run_info['sampling_plans']
        self.output_dir = run_info['output_dir']; self.num_layers = run_info['num_layers']; self.num_events = run_info['num_valid_events']
        self.event_id_to_index = run_info['event_id_to_index']; self.original_num_events = run_info['original_num_events']
        self.input_data_dir = self.config['input_data_directory']; self.base_seed = self.config.get('random_seed', None)
        self.num_workers = min(num_workers, os.cpu_count()); self.resume = resume; self.force_rerun = force_rerun
        self.checkpoint_path = os.path.join(self.output_dir, CHECKPOINT_FILENAME); self.event_sums: List[np.ndarray] = []
        self.event_rows_collected: List[Dict[str, int]] = []; self.start_bin_index: int = 0; self.execution_successful: bool = True
        self.worker_rngs_seeds = None; self.master_rng_for_tasks = None

        # 如果random_seed为None，则使用当前时间作为种子
        if self.base_seed is None:
            self.base_seed = int(time.time())

    def _initialize_state(self):

        self.event_sums = [np.zeros(self.num_layers, dtype=np.float64) for _ in range(self.num_events)]
        self.event_rows_collected = [{} for _ in range(self.num_events)]
        self.start_bin_index = 0
        if self.resume and not self.force_rerun:
            checkpoint_data = load_checkpoint(self.checkpoint_path)
            if checkpoint_data:
                try:
                    chkpt_orig_events = checkpoint_data.get('original_num_events', -1)
                    loaded_event_sums = checkpoint_data.get('event_sums')
                    loaded_event_rows_collected = checkpoint_data.get('event_rows_collected')
                    loaded_last_completed_bin_index = checkpoint_data.get('last_completed_bin_index', -1)
                    chkpt_event_id_map = checkpoint_data.get('event_id_map')
                    if isinstance(loaded_event_sums, list) and \
                       isinstance(loaded_event_rows_collected, list) and \
                       chkpt_orig_events == self.original_num_events and \
                       isinstance(chkpt_event_id_map, dict) and \
                       len(loaded_event_sums) == len(chkpt_event_id_map) and \
                       len(loaded_event_rows_collected) == len(chkpt_event_id_map):
                        logger.info("检查点结构基本匹配。正在恢复状态...")
                        temp_sums = [np.zeros(self.num_layers, dtype=np.float64) for _ in range(self.num_events)]
                        temp_collected = [{} for _ in range(self.num_events)]
                        valid_restored_count = 0
                        for original_event_id, plan_idx_in_chkpt in chkpt_event_id_map.items():
                             current_plan_idx = self.event_id_to_index.get(original_event_id)
                             if current_plan_idx is not None:
                                 if 0 <= plan_idx_in_chkpt < len(loaded_event_sums): temp_sums[current_plan_idx] = loaded_event_sums[plan_idx_in_chkpt]
                                 if 0 <= plan_idx_in_chkpt < len(loaded_event_rows_collected): temp_collected[current_plan_idx] = loaded_event_rows_collected[plan_idx_in_chkpt]
                                 valid_restored_count += 1
                        if valid_restored_count == self.num_events and len(temp_sums) == self.num_events:
                            self.event_sums = temp_sums; self.event_rows_collected = temp_collected
                            self.start_bin_index = loaded_last_completed_bin_index + 1
                            logger.info(f"从检查点恢复，将从能量分档索引 {self.start_bin_index} 开始。")
                        else: logger.warning(f"检查点映射或数量与当前有效事件不匹配 ({valid_restored_count} vs {self.num_events})，忽略检查点。"); self.start_bin_index = 0
                    else: logger.warning(f"检查点数据与当前配置不匹配，忽略检查点。"); self.start_bin_index = 0
                except Exception as e: logger.exception(f"解析检查点数据时出错。忽略检查点。"); self.start_bin_index = 0
            else: self.start_bin_index = 0
        elif self.force_rerun: logger.info("--force_rerun 指定，忽略检查点。"); self.start_bin_index = 0
        else: logger.info("未指定 --resume 或 --force_rerun，将从头开始。"); self.start_bin_index = 0
        if len(self.event_sums) != self.num_events: self.event_sums = [np.zeros(self.num_layers, dtype=np.float64) for _ in range(self.num_events)]
        if len(self.event_rows_collected) != self.num_events: self.event_rows_collected = [{} for _ in range(self.num_events)]

    def _prepare_seeds(self):
        ss_main = SeedSequence(self.base_seed)
        estimated_tasks = 0
        energy_bins_in_index = sorted(self.hdf5_index.keys(), key=lambda x: float(x.replace('MeV','')))
        for bin_key in energy_bins_in_index[self.start_bin_index:]:
            bin_meta = self.hdf5_index.get(bin_key, {}); max_passes_bin = 0
            rows_in_bin = bin_meta.get('total_rows_in_bin', 0)
            if rows_in_bin <= 0: continue
            for plan_idx, plan in enumerate(self.sampling_plans):
                 original_event_id = plan['event_id']; n_total_required = plan['samples_per_bin'].get(bin_key, 0)
                 n_collected = self.event_rows_collected[plan_idx].get(bin_key, 0)
                 if n_total_required > n_collected: passes_for_this_event = math.ceil(n_total_required / rows_in_bin); max_passes_bin = max(max_passes_bin, passes_for_this_event)
            chunk_r = bin_meta.get('chunk_shape', DEFAULT_CHUNK_SHAPE)[0]
            num_chunks_bin = math.ceil(rows_in_bin / chunk_r) if chunk_r > 0 else 0
            estimated_tasks += max_passes_bin * num_chunks_bin
        logger.debug(f"DEBUG: 预估剩余最大任务数: {estimated_tasks}")
        num_seeds_to_spawn = max(10000, estimated_tasks + 1000)
        self.worker_rngs_seeds = None; self.master_rng_for_tasks = None
        try:
            child_seeds_ss = ss_main.spawn(num_seeds_to_spawn)
            self.worker_rngs_seeds = iter([s.generate_state(1)[0] for s in child_seeds_ss])
            logger.info(f"已预先生成 {num_seeds_to_spawn} 个随机种子。")
        except Exception as e:
            logger.exception("生成大量种子时出错，将使用动态生成。")
            self.worker_rngs_seeds = None; self.master_rng_for_tasks = Generator(PCG64(ss_main.spawn(1)[0]))
            logger.info("将动态生成随机种子。")

    def _get_next_seed(self) -> int:
        try:
            if self.worker_rngs_seeds: return next(self.worker_rngs_seeds)
            else:
                if self.master_rng_for_tasks is None: ss = SeedSequence(self.base_seed); self.master_rng_for_tasks = Generator(PCG64(ss.spawn(1)[0])); logger.warning("Master RNG fallback init!")
                return self.master_rng_for_tasks.integers(low=0, high=2**32)
        except StopIteration: logger.error("严重错误: 预生成的随机种子已用尽！"); sys.exit("Seed exhaustion error.")

    def _distribute_deterministically(self, chunks_in_pass: List[Tuple[str, int, int, int]], p_eff: float, total_target_for_pass: int) -> Dict[Tuple[str, int], int]:
        """使用最大余数法确定性地分配样本到块，同时应用抽样优化
        
        优化策略：
        1. 根据MIN_SAMPLING_RATIO反向计算需要的块数量
        2. 只从随机选择的N个块中抽样，而不是从所有块抽样
        3. 每个选择的块自然达到所需的最低抽样率
        """
        allocations: Dict[Tuple[str, int], int] = {} # {(path, start): count}
        
        # 如果没有块或目标为0，直接返回空分配
        if not chunks_in_pass or total_target_for_pass <= 0:
            return allocations
            
        # 计算原始总行数
        total_rows = sum(rows for _, _, _, rows in chunks_in_pass)
        if total_rows <= 0:
            return allocations
            
        # 计算总体抽样率
        overall_sampling_rate = total_target_for_pass / total_rows
        
        # 根据MIN_SAMPLING_RATIO反向计算需要的块数量
        avg_chunk_size = total_rows / len(chunks_in_pass)
        
        # 仅当总体抽样率小于阈值时才进行优化
        if overall_sampling_rate < MIN_SAMPLING_RATIO:
            # 计算需要抽样的理想块数
            # 公式: 总目标样本数 / (每块大小 * MIN_SAMPLING_RATIO)
            ideal_chunks_needed = math.ceil(total_target_for_pass / (avg_chunk_size * MIN_SAMPLING_RATIO))
            
            # 限制块数在合理范围内
            target_chunks = min(len(chunks_in_pass), max(5, ideal_chunks_needed))
            
            logger.debug(f"优化: 总抽样率低于阈值 ({overall_sampling_rate:.3%} < {MIN_SAMPLING_RATIO:.0%})")
            logger.debug(f"优化: 总块数:{len(chunks_in_pass)}, 需要:{ideal_chunks_needed}, 选择:{target_chunks}")
            
            # 随机选择固定数量的块
            chunks_copy = chunks_in_pass.copy()
            random.shuffle(chunks_copy)
            selected_chunks = chunks_copy[:target_chunks]
            
            # 更新p_eff以反映我们只使用部分块
            selected_total_rows = sum(rows for _, _, _, rows in selected_chunks)
            if selected_total_rows > 0:
                adjusted_p_eff = total_target_for_pass / selected_total_rows
            else:
                return allocations  # 如果选择的块没有行，返回空分配
                
            # 使用正常的分配算法，但只用于选择的块
            fractional_parts: List[Tuple[float, str, int]] = []
            total_base_allocation = 0
            
            # 1. 计算基础分配和分数部分
            for path, start, end, rows_in_chunk in selected_chunks:
                task_key = (path, start)
                if rows_in_chunk <= 0:
                    allocations[task_key] = 0
                    continue
                exact_n = rows_in_chunk * adjusted_p_eff
                n_base = math.floor(exact_n)
                f_part = exact_n - n_base
                allocations[task_key] = n_base
                total_base_allocation += n_base
                if f_part > 1e-9:
                    fractional_parts.append((f_part, path, start))
            
            # 2. 计算需要额外分配的余数
            target_pass_rounded = round(total_target_for_pass)
            n_remainder = target_pass_rounded - total_base_allocation
            n_remainder = max(0, min(n_remainder, len(selected_chunks)))
            
            # 3. 按小数部分大小分配余数
            fractional_parts.sort(key=lambda item: item[0], reverse=True)
            num_to_distribute = min(n_remainder, len(fractional_parts))
            
            for i in range(num_to_distribute):
                _, path, start = fractional_parts[i]
                allocations[(path, start)] += 1
                
            return allocations
            
        else:
            # 常规分配 - 不需要优化，使用所有块
            fractional_parts: List[Tuple[float, str, int]] = []
            total_base_allocation = 0
            
            # 1. 计算基础分配和分数部分
            for path, start, end, rows_in_chunk in chunks_in_pass:
                task_key = (path, start)
                if rows_in_chunk <= 0:
                    allocations[task_key] = 0
                    continue
                exact_n = rows_in_chunk * p_eff
                n_base = math.floor(exact_n)
                f_part = exact_n - n_base
                allocations[task_key] = n_base
                total_base_allocation += n_base
                if f_part > 1e-9:
                    fractional_parts.append((f_part, path, start))
    
            # 2. 计算需要额外分配的余数
            target_pass_rounded = round(total_target_for_pass)
            n_remainder = target_pass_rounded - total_base_allocation
            n_remainder = max(0, min(n_remainder, len(chunks_in_pass)))
    
            # 3. 按小数部分大小分配余数
            fractional_parts.sort(key=lambda item: item[0], reverse=True)
            num_to_distribute = min(n_remainder, len(fractional_parts))
    
            for i in range(num_to_distribute):
                _, path, start = fractional_parts[i]
                allocations[(path, start)] += 1
    
            return allocations

    def _get_files_stats(self, energy_bin):
        """获取能量级别所有文件的状态，用于验证缓存有效性"""
        if energy_bin not in self.hdf5_index:
            return {}
        
        stats = {}
        part_files_info = self.hdf5_index[energy_bin].get('parts', [])
        for part_info in part_files_info:
            path_relative = part_info['path']
            path_abs = os.path.join(self.input_data_dir, path_relative)
            try:
                stats[path_relative] = os.path.getmtime(path_abs)
            except OSError:
                # 如果文件不存在，记录当前时间戳以强制重新生成
                stats[path_relative] = time.time()
        return stats
    
    def _verify_cache_valid(self, cache_data, energy_bin):
        """验证块信息缓存是否仍然有效"""
        if not cache_data or 'file_stats' not in cache_data:
            return False
            
        # 检查文件是否发生变化（通过修改时间）
        current_stats = self._get_files_stats(energy_bin)
        cached_stats = cache_data['file_stats']
        
        # 检查文件数量是否一致
        if len(current_stats) != len(cached_stats):
            logger.info(f"缓存无效：文件数量已改变 ({len(cached_stats)} -> {len(current_stats)})")
            return False
            
        # 检查文件修改时间是否一致
        for file_path, mtime in current_stats.items():
            if file_path not in cached_stats:
                logger.info(f"缓存无效：新文件已添加 ({file_path})")
                return False
            if mtime > cached_stats[file_path]:
                logger.info(f"缓存无效：文件已修改 ({file_path})")
                return False
                
        return True
    
    @staticmethod
    def _collect_chunks_for_file(part_info, chunk_rows, input_data_dir):
        """并行收集单个文件的块信息"""
        file_chunks = []
        part_file_path_relative = part_info['path']
        part_file_path_abs = os.path.join(input_data_dir, part_file_path_relative)
        rows_in_this_part = part_info.get('rows', 0)
        
        if rows_in_this_part <= 0:
            return file_chunks
            
        for chunk_start in range(0, rows_in_this_part, chunk_rows):
            chunk_end = min(chunk_start + chunk_rows, rows_in_this_part)
            actual_rows = chunk_end - chunk_start
            if actual_rows <= 0:
                continue
            file_chunks.append((part_file_path_abs, chunk_start, chunk_end, actual_rows))
            
        return file_chunks
        
    def collect_chunks_info(self, energy_bin, chunk_rows):
        """收集能量级别的块信息，支持缓存和并行处理"""
        # 缓存文件路径
        cache_dir = os.path.join(self.output_dir, "chunks_cache")
        os.makedirs(cache_dir, exist_ok=True)
        sanitized_name = energy_bin.replace('.', '_').replace('/', '_')
        cache_file = os.path.join(cache_dir, f"chunks_cache_{sanitized_name}.pkl")
        
        # 检查缓存是否存在且有效
        if os.path.exists(cache_file) and not self.force_rerun:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # 验证缓存有效性
                    if self._verify_cache_valid(cache_data, energy_bin):
                        logger.info(f"使用缓存的块信息: {energy_bin} ({len(cache_data['chunks_in_pass'])} 个块)")
                        return cache_data['chunks_in_pass']
                    else:
                        logger.info(f"块信息缓存已过期: {energy_bin}")
            except Exception as e:
                logger.warning(f"读取块信息缓存失败: {e}")
        
        # 如果缓存无效或不存在，收集块信息
        logger.info(f"为 {energy_bin} 收集块信息（并行处理）...")
        bin_metadata = self.hdf5_index[energy_bin]
        part_files_info = bin_metadata.get('parts', [])
        
        # 使用并行处理收集块信息
        chunks_in_pass = []
        collection_start = time.time()
        
        # 对于小规模的文件集，直接串行处理
        if len(part_files_info) < 10:
            for part_info in part_files_info:
                chunks = self._collect_chunks_for_file(part_info, chunk_rows, self.input_data_dir)
                chunks_in_pass.extend(chunks)
        else:
            # 对于大规模的文件集，使用并行处理
            with ProcessPoolExecutor(max_workers=min(self.num_workers, len(part_files_info))) as executor:
                futures = []
                for part_info in part_files_info:
                    futures.append(executor.submit(
                        self._collect_chunks_for_file,
                        part_info,
                        chunk_rows,
                        self.input_data_dir
                    ))
                
                # 收集结果
                for i, future in enumerate(as_completed(futures)):
                    try:
                        file_chunks = future.result()
                        chunks_in_pass.extend(file_chunks)
                        # 每处理10%的文件或文件数量少于10时，输出进度
                        if (i+1) % max(1, len(futures)//10) == 0 or i+1 == len(futures):
                            logger.info(f"  块信息收集进度: {i+1}/{len(futures)} 个文件处理完成")
                    except Exception as e:
                        logger.error(f"收集块信息时出错: {e}")
        
        collection_time = time.time() - collection_start
        logger.info(f"块信息收集完成: {energy_bin}，共 {len(chunks_in_pass)} 个块，耗时 {collection_time:.2f}s")
        
        # 保存到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'chunks_in_pass': chunks_in_pass,
                    'timestamp': time.time(),
                    'file_stats': self._get_files_stats(energy_bin)
                }, f)
            logger.info(f"块信息已缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存块信息缓存失败: {e}")
        
        return chunks_in_pass

    def execute(self) -> Optional[Tuple[List[np.ndarray], List[Dict[str, int]]]]:
        """执行主要的抽样循环 (使用并行确定性分配和抽样效率优化)"""
        logger.info("--- 开始阶段 2: 执行抽样任务 (使用并行确定性分配) ---")
        self._initialize_state()
        self._prepare_seeds()

        energy_bins_in_index = sorted(self.hdf5_index.keys(), key=lambda x: float(x.replace('MeV','')))
        if not energy_bins_in_index: logger.warning("HDF5 索引中没有能量分档。"); return None

        self.execution_successful = True

        try:
            for bin_idx in range(self.start_bin_index, len(energy_bins_in_index)):
                energy_bin = energy_bins_in_index[bin_idx]
                bin_start_time = time.time()
                logger.info(f"--- 开始处理能量分档: {energy_bin} ({bin_idx+1}/{len(energy_bins_in_index)}) ---")

                if energy_bin not in self.hdf5_index: continue
                bin_metadata = self.hdf5_index[energy_bin]
                total_rows_in_bin = bin_metadata.get('total_rows_in_bin', 0)
                part_files_info = bin_metadata.get('parts', [])
                dataset_name = bin_metadata.get('dataset_name', HDF5_DATASET_NAME)
                nominal_chunk_shape = tuple(bin_metadata.get('chunk_shape', DEFAULT_CHUNK_SHAPE))
                if len(nominal_chunk_shape) != 2: nominal_chunk_shape = DEFAULT_CHUNK_SHAPE
                chunk_rows = nominal_chunk_shape[0] if nominal_chunk_shape[0] > 0 else DEFAULT_CHUNK_SHAPE[0]

                if total_rows_in_bin <= 0 or not part_files_info:
                    logger.warning(f"能量分档 {energy_bin} 无数据或无文件信息，跳过。"); self._save_current_checkpoint(bin_idx); continue

                active_event_tasks: Dict[int, int] = {}
                max_passes_needed = 0
                for plan_idx, plan in enumerate(self.sampling_plans):
                     original_event_id = plan['event_id']; n_total_required = plan['samples_per_bin'].get(energy_bin, 0)
                     n_collected_so_far = self.event_rows_collected[plan_idx].get(energy_bin, 0); n_still_needed = n_total_required - n_collected_so_far
                     if n_still_needed > 0: active_event_tasks[original_event_id] = n_still_needed; passes_for_this_event = math.ceil(n_total_required / total_rows_in_bin); max_passes_needed = max(max_passes_needed, passes_for_this_event)

                if not active_event_tasks: logger.info(f"能量分档 {energy_bin} 无需处理。"); self._save_current_checkpoint(bin_idx); continue

                logger.info(f"能量分档 {energy_bin}: 需要处理 {len(active_event_tasks)} 个事件，最多需要 {max_passes_needed} 次遍历。")
                current_active_tasks_in_bin = active_event_tasks.copy()
                start_pass_num = 1
                if self.resume and bin_idx == self.start_bin_index -1 :
                    min_collected_for_active = min((self.event_rows_collected[self.event_id_to_index[eid]].get(energy_bin, 0) for eid in current_active_tasks_in_bin), default=0)
                    start_pass_num = math.floor(min_collected_for_active / total_rows_in_bin) + 1 if total_rows_in_bin > 0 else 1
                    logger.info(f"Resuming: 最少收集数 {min_collected_for_active}, 从 Pass {start_pass_num} 开始处理 {energy_bin}")

                # 使用优化后的块信息收集方法
                chunks_in_pass = self.collect_chunks_info(energy_bin, chunk_rows)
                if not chunks_in_pass:
                    logger.warning(f"能量分档 {energy_bin} 未找到有效数据块，跳过。")
                    self._save_current_checkpoint(bin_idx)
                    continue

                for pass_num in range(start_pass_num, max_passes_needed + 1):
                    if not current_active_tasks_in_bin: logger.info(f"  Pass {pass_num}: 无剩余，跳过。"); continue
                    pass_start_time = time.time(); logger.info(f"  开始 Pass {pass_num}/{max_passes_needed} for {energy_bin}...")
                    tasks_for_pool: List[Dict[str, Any]] = []

                    # 2. 计算每个活动事件在此 Pass 的需求（并行优化）
                    event_needs_in_pass: Dict[int, Dict[str, Union[float, int]]] = {}
                    event_needs_calc_start = time.time()
                    active_events = list(current_active_tasks_in_bin.keys())
                    
                    if len(active_events) > 50 and self.num_workers > 1:
                        # 使用并行处理计算事件需求
                        logger.info(f"    并行计算 {len(active_events)} 个事件的需求...")
                        
                        def calc_event_need(event_id, still_needed_total, total_rows_in_bin, pass_num):
                            """计算单个事件的需求信息"""
                            try:
                                plan_for_event = next((p for p in self.sampling_plans if p['event_id'] == event_id), None)
                                if not plan_for_event: 
                                    return event_id, None
                                    
                                n_total_required = plan_for_event['samples_per_bin'].get(energy_bin, 0)
                                plan_idx = self.event_id_to_index.get(event_id, -1)
                                if plan_idx == -1: 
                                    return event_id, None
                                    
                                n_collected_so_far = self.event_rows_collected[plan_idx].get(energy_bin, 0)
                                if n_collected_so_far >= n_total_required:
                                    return event_id, None
                                    
                                current_pass_start_row_target = (pass_num - 1) * total_rows_in_bin
                                rows_target_in_pass = 0
                                if current_pass_start_row_target < n_total_required: 
                                    rows_target_in_pass = min(total_rows_in_bin, n_total_required - current_pass_start_row_target)
                                    
                                if rows_target_in_pass <= 0: 
                                    return event_id, None
                                    
                                p_eff = rows_target_in_pass / total_rows_in_bin
                                return event_id, {
                                    'p_eff': p_eff,
                                    'n_still_needed': still_needed_total,
                                    'target_for_pass': round(rows_target_in_pass)
                                }
                            except Exception as e:
                                logger.error(f"计算事件 {event_id} 需求时出错: {e}")
                                return event_id, None
                        
                        # 使用线程池并行计算事件需求
                        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.num_workers, len(active_events))) as executor:
                            futures = []
                            for event_id in active_events:
                                futures.append(executor.submit(
                                    calc_event_need,
                                    event_id,
                                    current_active_tasks_in_bin[event_id],
                                    total_rows_in_bin,
                                    pass_num
                                ))
                                
                            # 显示进度并收集结果
                            valid_events = 0
                            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                                try:
                                    event_id, result = future.result()
                                    if result is not None:
                                        event_needs_in_pass[event_id] = result
                                        valid_events += 1
                                    
                                    # 每处理20%的事件显示一次进度
                                    if (i+1) % max(1, len(futures)//5) == 0:
                                        logger.info(f"      需求计算进度: {i+1}/{len(futures)} 事件, 有效事件: {valid_events}")
                                        
                                except Exception as e:
                                    logger.error(f"处理事件需求结果时出错: {e}")
                    else:
                        # 串行处理
                        for i, original_event_id in enumerate(active_events):
                            n_still_needed_total = current_active_tasks_in_bin[original_event_id]
                            plan_for_event = next((p for p in self.sampling_plans if p['event_id'] == original_event_id), None)
                            if not plan_for_event: continue
                            n_total_required = plan_for_event['samples_per_bin'].get(energy_bin, 0)
                            plan_idx = self.event_id_to_index.get(original_event_id, -1);
                            if plan_idx == -1: continue
                            n_collected_so_far = self.event_rows_collected[plan_idx].get(energy_bin, 0)
                            if n_collected_so_far >= n_total_required:
                                 if original_event_id in current_active_tasks_in_bin: del current_active_tasks_in_bin[original_event_id]
                                 continue
                            current_pass_start_row_target = (pass_num - 1) * total_rows_in_bin
                            rows_target_in_pass = 0
                            if current_pass_start_row_target < n_total_required: rows_target_in_pass = min(total_rows_in_bin, n_total_required - current_pass_start_row_target)
                            if rows_target_in_pass <= 0: continue
                            p_eff = rows_target_in_pass / total_rows_in_bin
                            
                            # Store needed info for deterministic allocation
                            event_needs_in_pass[original_event_id] = {
                                'p_eff': p_eff,
                                'n_still_needed': n_still_needed_total, # Overall remaining needed
                                'target_for_pass': round(rows_target_in_pass) # Integer target for this pass
                            }
                            
                            # 每处理100个事件显示一次进度
                            if (i+1) % 100 == 0 and i > 0:
                                logger.info(f"      需求计算进度: {i+1}/{len(active_events)} 事件")
                    
                    event_needs_calc_time = time.time() - event_needs_calc_start
                    if len(active_events) > 100:
                        logger.info(f"    需求计算完成: {len(event_needs_in_pass)}/{len(active_events)} 个有效事件，耗时: {event_needs_calc_time:.3f}s")
                    
                    # 如果没有有效事件需要在此Pass处理，跳过剩余操作
                    if not event_needs_in_pass:
                        logger.info(f"    Pass {pass_num}: 无有效事件需要处理，跳过。")
                        continue

                    # 3. 并行确定性分配 - 使用线程池并行处理每个事件的分配
                    final_sampling_needs: Dict[Tuple[str, int], Dict[int, int]] = {} # {(path, start): {event_id: n_to_sample}}
                    chunk_info_map = {(path, start): rows for path, start, end, rows in chunks_in_pass} # Lookup map
                    
                    # 开始并行分配计算
                    chunk_allocation_start_time = time.time()
                    
                    if self.num_workers > 1 and len(event_needs_in_pass) > 1:
                        logger.info(f"    使用 {self.num_workers} 个线程并行计算 {len(event_needs_in_pass)} 个事件的分配方案...")
                        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                            # 为每个事件创建并行任务
                            allocation_futures = {}
                            for original_event_id, needs in event_needs_in_pass.items():
                                future = executor.submit(
                                    self._distribute_deterministically,
                                    chunks_in_pass,
                                    needs['p_eff'],
                                    needs['target_for_pass']
                                )
                                allocation_futures[original_event_id] = future
                            
                            # 显示进度并收集分配结果
                            processed_allocations = 0
                            for i, (original_event_id, future) in enumerate(allocation_futures.items()):
                                try:
                                    chunk_allocations_for_event = future.result()
                                    n_still_needed = event_needs_in_pass[original_event_id]['n_still_needed']
                                    target_for_pass = event_needs_in_pass[original_event_id]['target_for_pass']
                                    
                                    # 应用相同的上限和填充最终分配
                                    temp_still_needed_for_event = n_still_needed
                                    processed_in_event_pass = 0
                                    
                                    for (path, start), n_deterministic in chunk_allocations_for_event.items():
                                        rows_in_chunk = chunk_info_map.get((path, start), 0)
                                        if rows_in_chunk == 0: continue
                                        
                                        # 应用三重上限
                                        n_capped1 = min(n_deterministic, temp_still_needed_for_event)
                                        n_capped2 = min(n_capped1, rows_in_chunk)
                                        n_to_sample = max(0, min(n_capped2, target_for_pass - processed_in_event_pass))
                                        
                                        if n_to_sample > 0:
                                            task_key = (path, start)
                                            if task_key not in final_sampling_needs: final_sampling_needs[task_key] = {}
                                            final_sampling_needs[task_key][original_event_id] = n_to_sample
                                            
                                            # 更新追踪变量
                                            temp_still_needed_for_event -= n_to_sample
                                            processed_in_event_pass += n_to_sample
                                    
                                    processed_allocations += 1
                                    # 每处理20%的分配或至少10个事件显示一次进度
                                    progress_interval = max(1, min(10, len(allocation_futures)//5))
                                    if (i+1) % progress_interval == 0:
                                        logger.info(f"      分配计算进度: {i+1}/{len(allocation_futures)} 事件")
                                        
                                except Exception as e:
                                    logger.exception(f"    计算事件 {original_event_id} 的分配方案时出错: {e}")
                    else:
                        # 退回到串行处理（事件少时更高效）
                        for original_event_id, needs in event_needs_in_pass.items():
                            p_eff = needs['p_eff']
                            n_still_needed = needs['n_still_needed']
                            target_for_pass = needs['target_for_pass']
                            
                            chunk_allocations_for_event = self._distribute_deterministically(
                                chunks_in_pass, p_eff, target_for_pass
                            )
                            
                            # 应用上限和填充最终分配
                            temp_still_needed_for_event = n_still_needed
                            processed_in_event_pass = 0
                            
                            for (path, start), n_deterministic in chunk_allocations_for_event.items():
                                rows_in_chunk = chunk_info_map.get((path, start), 0)
                                if rows_in_chunk == 0: continue
                                
                                # 应用三重上限
                                n_capped1 = min(n_deterministic, temp_still_needed_for_event)
                                n_capped2 = min(n_capped1, rows_in_chunk)
                                n_to_sample = max(0, min(n_capped2, target_for_pass - processed_in_event_pass))
                                
                                if n_to_sample > 0:
                                    task_key = (path, start)
                                    if task_key not in final_sampling_needs: final_sampling_needs[task_key] = {}
                                    final_sampling_needs[task_key][original_event_id] = n_to_sample
                                    
                                    # 更新追踪变量
                                    temp_still_needed_for_event -= n_to_sample
                                    processed_in_event_pass += n_to_sample
                    
                    allocation_time = time.time() - chunk_allocation_start_time
                    if len(event_needs_in_pass) > 1 and len(chunks_in_pass) > 100:
                        logger.info(f"    分配计算完成，耗时: {allocation_time:.3f}s")

                    # 4. 创建最终的任务列表
                    tasks_for_pool.clear() # Clear list before populating for the pass
                    task_creation_start = time.time()
                    total_task_items = len(final_sampling_needs)
                    
                    if total_task_items > 100:
                        logger.info(f"    开始创建 {total_task_items} 个任务...")
                    
                    for i, ((path, start), needs_dict) in enumerate(final_sampling_needs.items()):
                        corresponding_chunk = next((c for c in chunks_in_pass if c[0] == path and c[1] == start), None)
                        if not corresponding_chunk: continue
                        end_row = corresponding_chunk[2]

                        task_seed = self._get_next_seed()
                        task_info = {
                            'part_file_path': path, 'chunk_start_row': start, 'chunk_end_row': end_row,
                            'sampling_needs': needs_dict, # {event_id: n_deterministic_capped}
                            'dataset_name': dataset_name, 'num_layers': self.num_layers, 'rng_seed': task_seed
                        }
                        tasks_for_pool.append(task_info)
                        
                        # 显示进度
                        if (i+1) % max(1, min(10000, total_task_items // 5)) == 0:
                            tasks_percentage = (i+1) / total_task_items * 100
                            logger.info(f"      任务创建进度: {i+1}/{total_task_items} ({tasks_percentage:.1f}%)")
                    
                    task_creation_time = time.time() - task_creation_start
                    if total_task_items > 100:
                        logger.info(f"    任务创建完成: {len(tasks_for_pool)}/{total_task_items} 有效任务，耗时: {task_creation_time:.3f}s")

                    # 5. Execute tasks (rest of the loop as before)
                    if tasks_for_pool:
                        logger.info(f"    Pass {pass_num}: 将 {len(tasks_for_pool)} 个块任务提交到进程池...")
                        processed_chunks_in_pass = 0
                        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                            futures = {executor.submit(sample_chunk_task, task): task for task in tasks_for_pool}
                            for future in as_completed(futures):
                                task_submitted = futures[future]
                                try:
                                    partial_sums, actual_counts = future.result()
                                    for original_event_id, summed_array in partial_sums.items():
                                        plan_idx = self.event_id_to_index.get(original_event_id);
                                        if plan_idx is not None: self.event_sums[plan_idx] += summed_array
                                        else: logger.error(f"聚合错误：无效 event_id {original_event_id} 从 worker 返回。")
                                    for original_event_id, count in actual_counts.items():
                                        if count > 0:
                                            plan_idx = self.event_id_to_index.get(original_event_id)
                                            if plan_idx is not None:
                                                current_collected = self.event_rows_collected[plan_idx].get(energy_bin, 0); new_collected = current_collected + count
                                                self.event_rows_collected[plan_idx][energy_bin] = new_collected
                                                if original_event_id in current_active_tasks_in_bin:
                                                    current_active_tasks_in_bin[original_event_id] -= count
                                                    if current_active_tasks_in_bin[original_event_id] <= 0: logger.debug(f"Event {original_event_id} completed sampling for {energy_bin}."); current_active_tasks_in_bin.pop(original_event_id, None)
                                    processed_chunks_in_pass += 1
                                    if processed_chunks_in_pass % (max(1, len(tasks_for_pool) // 10)) == 0 or processed_chunks_in_pass == len(tasks_for_pool): logger.debug(f"      Pass {pass_num}: 已聚合 {processed_chunks_in_pass}/{len(tasks_for_pool)} 个块的结果...")
                                except Exception as e: failed_task_info = f"chunk ..."; logger.exception(f"    Pass {pass_num}: 处理块任务 ({failed_task_info}) 结果时严重错误"); self.execution_successful = False
                        logger.info(f"    Pass {pass_num}: 完成处理 {processed_chunks_in_pass} 个块。")
                    else: logger.info(f"    Pass {pass_num}: 无需处理的任务。")
                    pass_time = time.time() - pass_start_time
                    logger.info(f"  完成 Pass {pass_num}/{max_passes_needed} for {energy_bin}, 耗时: {pass_time:.2f}s")
                    if not current_active_tasks_in_bin: logger.info(f"  能量分档 {energy_bin} 已完成，提前结束 Passes。"); break

                logger.info(f"完成能量分档 {energy_bin} 的所有处理。正在保存检查点...")
                self._save_current_checkpoint(bin_idx)
                bin_time = time.time() - bin_start_time
                logger.info(f"--- 完成处理能量分档: {energy_bin}, 耗时: {bin_time:.2f}s ---")

        except Exception as e:
             logger.exception("在主要处理循环中发生未捕获的错误")
             self.execution_successful = False
        finally: pass

        logger.info("--- 阶段 2: 执行抽样任务完成 ---")
        if self.execution_successful: return self.event_sums, self.event_rows_collected
        else: logger.error("执行过程中发生错误..."); return self.event_sums, self.event_rows_collected


    def _save_current_checkpoint(self, last_completed_bin_index: int):
        state = {
            'event_sums': self.event_sums, 'event_rows_collected': self.event_rows_collected,
            'last_completed_bin_index': last_completed_bin_index,
            'original_num_events': self.original_num_events, 'event_id_map': self.event_id_to_index
        }
        save_checkpoint(self.checkpoint_path, state)
# --- End SamplingExecutor Class ---

# --- 阶段 3: 完成与输出 (独立函数 - 保持不变) ---
def finalize_run(run_info: Dict[str, Any], final_event_sums: List[np.ndarray], final_event_counts: List[Dict[str, int]], execution_successful: bool, config: Dict[str, Any]):

    logger.info("--- 开始阶段 3: 完成与输出 ---")
    output_dir = run_info['output_dir']; run_name = run_info['run_name']; num_layers = run_info['num_layers']
    sampling_plans = run_info['sampling_plans']; original_num_events = run_info['original_num_events']
    num_valid_events = run_info['num_valid_events']; checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILENAME)
    
    # 获取所有可能的能量点
    all_energy_bins = sorted(run_info['hdf5_index'].keys(), key=lambda x: float(x.replace('MeV','')))
    logger.info(f"将输出 {len(all_energy_bins)} 个能量点: {', '.join(all_energy_bins)}")
    
    output_file_count = 0
    if num_valid_events > 0 and len(final_event_sums) == num_valid_events and len(final_event_counts) == num_valid_events:
        logger.info(f"正在为 {num_valid_events} 个有效事件生成输出文件...")
        event_id_to_index = run_info['event_id_to_index'] # Get map
        for plan_idx, plan in enumerate(sampling_plans):
            original_event_id = plan['event_id']; final_sum = final_event_sums[plan_idx]
            actual_counts_this_event = final_event_counts[plan_idx]; relevant_counts = {}
            for energy_key in plan['samples_per_bin']: relevant_counts[energy_key] = actual_counts_this_event.get(energy_key, 0)
            
            # 即使没有样本，也输出文件
            save_output_file(output_dir, run_name, original_event_id, plan, final_sum, relevant_counts, num_layers, original_num_events, all_energy_bins, config = config)
            output_file_count += 1
    else: logger.error("最终结果列表长度与有效事件数不匹配，无法生成输出文件。")
    logger.info(f"共生成 {output_file_count} 个输出文件。")
    # Clean up checkpoint file
    if execution_successful:
        logger.info("执行成功完成。")
        try:
            if os.path.exists(checkpoint_path): os.remove(checkpoint_path); logger.info(f"成功完成，检查点文件已删除: {checkpoint_path}")
        except OSError as e: logger.warning(f"删除检查点文件失败: {e}")
    else:
         if os.path.exists(checkpoint_path): logger.warning(f"执行出错或未完成，检查点文件 {checkpoint_path} 已保留。")
         else: logger.warning(f"执行出错或未完成，且未找到检查点文件。")
    logger.info("--- 阶段 3: 完成与输出结束 ---")


# --- 主程序入口 (OOP Style Orchestrator) ---
def main_orchestrator(args=None, args_list=None):
    """主协调函数，调用规划、执行和完成阶段"""
    parser = argparse.ArgumentParser(description="根据配置文件，从 HDF5 数据中抽样并计算探测器响应。")
    parser.add_argument("--config", required=True, help="指向 config.json 配置文件的路径")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2), help="并行工作进程数")
    parser.add_argument("--resume", action='store_true', help="尝试从检查点恢复")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--force_rerun", action='store_true', help="强制忽略检查点")
    parser.add_argument("--clean_cache", action='store_true', help="清除所有块信息缓存")

    if args is None and args_list is None:
        args = parser.parse_args()
    elif args_list is not None:
        args = parser.parse_args(args_list)

    logger.setLevel(getattr(logging, args.log_level.upper()))
    if args.log_level == "DEBUG": logger.info("已启用 DEBUG 日志级别。")

    total_start_time = time.time()
    run_info = None
    execution_successful = False
    final_event_sums = None
    final_event_counts = None

    try:
        # --- 检查是否需要清除缓存 ---
        if args.clean_cache:
            try:
                config = load_json(args.config)
                output_dir = config['output_directory']
                cache_dir = os.path.join(output_dir, "chunks_cache")
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir)
                    logger.info(f"已清除块信息缓存目录: {cache_dir}")
                else:
                    logger.info(f"未找到块信息缓存目录: {cache_dir}")
                if not args.force_rerun and not args.resume:
                    logger.info("缓存已清除，退出程序。如需继续执行，请移除 --clean_cache 参数。")
                    return
            except Exception as e:
                logger.error(f"清除缓存时发生错误: {e}")
                if not args.force_rerun and not args.resume:
                    return

        # --- 阶段 1: 规划 ---
        logger.info("=== Stage 1: Planning ===")
        config = load_json(args.config)
        input_data_dir = config['input_data_directory']
        hdf5_index_path = os.path.join(input_data_dir, HDF5_INDEX_FILENAME) # Use constant
        hdf5_index = load_json(hdf5_index_path)
        output_dir = config['output_directory']
        os.makedirs(output_dir, exist_ok=True)

        planner = SamplingPlanner(config, hdf5_index)
        sampling_plans = planner.generate() # 直接调用生成

        if not sampling_plans:
            logger.error("抽样规划阶段未生成任何有效计划，程序退出。")
            sys.exit(1)

        # 保存计划文件
        save_plan_to_file(sampling_plans, output_dir) # 保存计划

        # 准备 run_info
        original_num_events = config['num_datasets_to_generate']
        run_info = {
            'config': config, 'hdf5_index': hdf5_index, 'sampling_plans': sampling_plans,
            'output_dir': output_dir, 'run_name': config['simulation_run_name'],
            'num_layers': config['num_detector_layers'], 'original_num_events': original_num_events,
            'num_valid_events': len(sampling_plans),
            'event_id_to_index': {plan['event_id']: idx for idx, plan in enumerate(sampling_plans)}
        }
        logger.info("--- 阶段 1: 规划抽样任务完成 ---")


        # --- 阶段 2: 执行 ---
        logger.info("=== Stage 2: Execution ===")
        executor = SamplingExecutor(run_info, args.workers, args.resume, args.force_rerun)
        results = executor.execute()
        execution_successful = results is not None and executor.execution_successful # Check flag from executor

        if results:
            final_event_sums, final_event_counts = results
        else:
            # Attempt to load from checkpoint if execution failed early or returned None
            logger.warning("执行阶段未成功返回结果或失败，尝试从检查点恢复用于最终输出。")
            checkpoint_path = os.path.join(run_info['output_dir'], CHECKPOINT_FILENAME)
            checkpoint_data = load_checkpoint(checkpoint_path)
            if checkpoint_data and \
               checkpoint_data.get('original_num_events') == run_info['original_num_events'] and \
               checkpoint_data.get('event_id_map') == run_info['event_id_to_index'] and \
               len(checkpoint_data.get('event_sums',[])) == run_info['num_valid_events'] and \
               len(checkpoint_data.get('event_rows_collected',[])) == run_info['num_valid_events']:
                 final_event_sums = checkpoint_data['event_sums']
                 final_event_counts = checkpoint_data['event_rows_collected']
                 logger.info("已从检查点加载状态用于最终输出。")
                 # Keep execution_successful as False if execute() failed
            else:
                logger.error("无法从检查点加载有效状态用于最终输出。")
                final_event_sums = [np.zeros(run_info['num_layers']) for _ in range(run_info['num_valid_events'])]
                final_event_counts = [{} for _ in range(run_info['num_valid_events'])]
                execution_successful = False # Definitely failed

        # --- 阶段 3: 完成 ---
        logger.info("=== Stage 3: Finalization ===")
        if final_event_sums is not None and final_event_counts is not None:
            finalize_run(run_info, final_event_sums, final_event_counts, execution_successful, config)
        else:
            logger.error("无法获取最终结果，跳过最终处理步骤。")

    except Exception as e:
        logger.exception("主协调函数发生未捕获的错误")
        sys.exit(1) # Exit with error status

    finally:
        total_time = time.time() - total_start_time
        status = "成功" if execution_successful else "有错误或未完成"
        logger.info(f"\n--- 所有阶段完成 ({status}) ---")
        logger.info(f"总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    try:
        import h5py; import numpy; import scipy
    except ImportError as e: print(f"错误: 缺少必要的库: {e}. 请确保 h5py, numpy, scipy 已安装。", file=sys.stderr); sys.exit(1)
    # main_orchestrator()
    main_orchestrator(args_list=["--config", "config_test.json", "--workers", "16", "--force_rerun", "--log_level", "INFO"])