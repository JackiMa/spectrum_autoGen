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
# --- 常量 ---
CHECKPOINT_FILENAME: str = "sampling_checkpoint.pkl"
HDF5_DATASET_NAME: str = 'detector_response'
DEFAULT_CHUNK_SHAPE: Tuple[int, int] = (10000, 20)

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(process)d] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Helper Functions (save_checkpoint, load_checkpoint, save_output_file - 保持不变) ---
# ... (复制 save_checkpoint, load_checkpoint, save_output_file 函数) ...
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
                     num_total_events: int):
    padding_width = len(str(num_total_events)); filename = f"{run_name}_event_{event_id:0{padding_width}d}.txt"; filepath = os.path.join(output_dir, filename)
    try:
        if not actual_counts_per_bin: logger.warning(f"Skipping output for event {event_id} as actual_counts_per_bin is empty."); return
        sampled_energies = sorted(actual_counts_per_bin.keys(), key=lambda x: float(x.replace('MeV','')))
        if not sampled_energies: logger.warning(f"Skipping output for event {event_id} as no valid energies found."); return
        numeric_energies_str = [energy.replace('MeV', '') for energy in sampled_energies]
        line1 = ", ".join(numeric_energies_str)
        line2 = ", ".join(str(actual_counts_per_bin.get(energy, 0)) for energy in sampled_energies)
        line3 = ", ".join(map(str, range(1, num_layers + 1)))
        line4 = np.array2string(final_sum, separator=', ', formatter={'float_kind':lambda x: f"{x:.8e}"},threshold=np.inf,max_line_width=np.inf).replace('[','').replace(']','')
        content = f"{line1}\n{line2}\n{line3}\n{line4}" # No trailing newline
        logger.debug(f"DEBUG SAVE Internal: Event {event_id}, Writing line2: {line2}") # Keep debug
        with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
    except Exception as e: logger.exception(f"写入输出文件 {filepath} 时发生未知错误")


# --- Worker Function (使用确定性的 n_to_sample) ---
def sample_chunk_task(task_info: Dict[str, Any]) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """工作进程执行的函数：处理单个数据块的抽样任务。(接收确定性的目标数)"""
    part_file_path = task_info['part_file_path']
    chunk_start = task_info['chunk_start_row']
    chunk_end = task_info['chunk_end_row']
    sampling_needs_map = task_info['sampling_needs'] # {event_idx: n_to_sample_deterministic}
    dataset_name = task_info['dataset_name']
    num_layers = task_info['num_layers']
    rng_seed = task_info['rng_seed']

    rng = Generator(PCG64(rng_seed))
    partial_sums: Dict[int, np.ndarray] = {}
    actual_counts: Dict[int, int] = {}
    is_target_chunk = '0.6MeV' in part_file_path

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

    # ... (异常处理保持不变) ...
    except FileNotFoundError: print(f"ERROR WORKER {os.getpid()}: HDF5 File Not Found - {part_file_path}", file=sys.stderr, flush=True); return {}, {}
    except KeyError: print(f"ERROR WORKER {os.getpid()}: Dataset '{dataset_name}' not found in {part_file_path}", file=sys.stderr, flush=True); return {}, {}
    except ValueError as e: print(f"ERROR WORKER {os.getpid()}: ValueError processing chunk {chunk_start}-{chunk_end} in {part_file_path}: {e}", file=sys.stderr, flush=True); return {}, {}
    except Exception as e: print(f"ERROR WORKER {os.getpid()}: Unhandled Exception processing chunk {chunk_start}-{chunk_end} in {part_file_path}: {type(e).__name__} - {e}", file=sys.stderr, flush=True); import traceback; traceback.print_exc(file=sys.stderr); return {}, {}

    return partial_sums, actual_counts
# --- End Worker Function ---


# --- 阶段 2: 执行抽样任务 (OOP Class) ---
class SamplingExecutor:
    """负责执行抽样、管理状态和并行处理"""
    # ... (__init__, _initialize_state, _prepare_seeds, _get_next_seed, _save_current_checkpoint 保持不变) ...
    def __init__(self, run_info: Dict[str, Any], num_workers: int, resume: bool, force_rerun: bool):
        # ... (同上) ...
        self.config = run_info['config']; self.hdf5_index = run_info['hdf5_index']; self.sampling_plans = run_info['sampling_plans']
        self.output_dir = run_info['output_dir']; self.num_layers = run_info['num_layers']; self.num_events = run_info['num_valid_events']
        self.event_id_to_index = run_info['event_id_to_index']; self.original_num_events = run_info['original_num_events']
        self.input_data_dir = self.config['input_data_directory']; self.base_seed = self.config.get('random_seed', None)
        self.num_workers = min(num_workers, os.cpu_count()); self.resume = resume; self.force_rerun = force_rerun
        self.checkpoint_path = os.path.join(self.output_dir, CHECKPOINT_FILENAME); self.event_sums: List[np.ndarray] = []
        self.event_rows_collected: List[Dict[str, int]] = []; self.start_bin_index: int = 0; self.execution_successful: bool = True
        self.worker_rngs_seeds = None; self.master_rng_for_tasks = None

    def _initialize_state(self):
        # ... (同上) ...
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
        # ... (修复后的版本) ...
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
        # ... (与上一版本相同) ...
        try:
            if self.worker_rngs_seeds: return next(self.worker_rngs_seeds)
            else:
                if self.master_rng_for_tasks is None: ss = SeedSequence(self.base_seed); self.master_rng_for_tasks = Generator(PCG64(ss.spawn(1)[0])); logger.warning("Master RNG fallback init!")
                return self.master_rng_for_tasks.integers(low=0, high=2**32)
        except StopIteration: logger.error("严重错误: 预生成的随机种子已用尽！"); sys.exit("Seed exhaustion error.")

    def _distribute_deterministically(self, chunks_info: List[Tuple[str, int, int, int]], p_eff: float, total_target_for_pass: int) -> Dict[Tuple[str, int], int]:
        """使用最大余数法确定性地分配样本到块"""
        allocations: Dict[Tuple[str, int], int] = {} # {(path, start): count}
        fractional_parts: List[Tuple[float, str, int]] = [] # [(f_part, path, start)]
        total_base_allocation = 0
        total_exact_sum = 0.0 # For checking

        # 1. 计算基础分配和分数部分
        for path, start, end, rows_in_chunk in chunks_info:
            task_key = (path, start)
            if rows_in_chunk <= 0:
                allocations[task_key] = 0
                continue
            exact_n = rows_in_chunk * p_eff
            total_exact_sum += exact_n
            n_base = math.floor(exact_n)
            f_part = exact_n - n_base
            allocations[task_key] = n_base
            total_base_allocation += n_base
            if f_part > 1e-9:
                fractional_parts.append((f_part, path, start))

        # 2. 计算需要额外分配的余数
        target_pass_rounded = round(total_target_for_pass)
        n_remainder = target_pass_rounded - total_base_allocation
        if n_remainder < 0:
             # logger.warning(...) # Keep warning if needed
             n_remainder = 0
        n_remainder = min(n_remainder, len(chunks_info)) # Cap remainder

        # 3. 按小数部分大小分配余数
        fractional_parts.sort(key=lambda item: item[0], reverse=True)
        num_to_distribute = min(n_remainder, len(fractional_parts))
        for i in range(num_to_distribute):
            _, path, start = fractional_parts[i]
            allocations[(path, start)] += 1

        # Debug check:
        final_sum_check = sum(allocations.values())
        if not math.isclose(final_sum_check, target_pass_rounded):
             logger.debug(f"DEBUG DISTRIBUTE: Pass Target={target_pass_rounded}, Base={total_base_allocation}, Remainder={n_remainder}, Distributed Sum={final_sum_check}, Exact Sum={total_exact_sum:.4f}")

        return allocations


    def execute(self) -> Optional[Tuple[List[np.ndarray], List[Dict[str, int]]]]:
        """执行主要的抽样循环 (使用确定性分配 - 修正版)"""
        logger.info("--- 开始阶段 2: 执行抽样任务 (使用确定性分配) ---")
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

                for pass_num in range(start_pass_num, max_passes_needed + 1):
                    if not current_active_tasks_in_bin: logger.info(f"  Pass {pass_num}: 无剩余，跳过。"); continue
                    pass_start_time = time.time(); logger.info(f"  开始 Pass {pass_num}/{max_passes_needed} for {energy_bin}...")
                    tasks_for_pool: List[Dict[str, Any]] = []

                    # 1. 收集当前 Pass 的所有块信息
                    chunks_in_pass: List[Tuple[str, int, int, int]] = []
                    part_file_counter = 0
                    for part_info in part_files_info:
                        part_file_counter += 1; part_file_path_relative = part_info['path']
                        part_file_path_abs = os.path.join(self.input_data_dir, part_file_path_relative)
                        rows_in_this_part = part_info.get('rows', 0)
                        if rows_in_this_part <= 0: continue
                        for chunk_start_in_part in range(0, rows_in_this_part, chunk_rows):
                            chunk_end_in_part = min(chunk_start_in_part + chunk_rows, rows_in_this_part)
                            actual_rows_in_chunk = chunk_end_in_part - chunk_start_in_part
                            if actual_rows_in_chunk <= 0: continue
                            chunks_in_pass.append((part_file_path_abs, chunk_start_in_part, chunk_end_in_part, actual_rows_in_chunk))

                    # 2. 计算每个活动事件在此 Pass 的需求
                    event_needs_in_pass: Dict[int, Dict[str, Union[float, int]]] = {}
                    for original_event_id in list(current_active_tasks_in_bin.keys()):
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

                    # 3. 确定性地为每个块/事件分配抽样数
                    final_sampling_needs: Dict[Tuple[str, int], Dict[int, int]] = {} # {(path, start): {event_id: n_to_sample}}
                    chunk_info_map = {(path, start): rows for path, start, end, rows in chunks_in_pass} # Lookup map

                    for original_event_id, needs in event_needs_in_pass.items():
                        p_eff = needs['p_eff']
                        n_still_needed = needs['n_still_needed']
                        target_for_pass = needs['target_for_pass']

                        # Get ideal allocation across chunks for this event/pass
                        chunk_allocations_for_event = self._distribute_deterministically(
                            chunks_info=chunks_in_pass, p_eff=p_eff, total_target_for_pass=target_for_pass
                        )

                        # Apply caps (still_needed, rows_in_chunk) and populate final_sampling_needs
                        temp_still_needed_for_event = n_still_needed # Track remaining for this event across chunks
                        processed_in_event_pass = 0 # Track assigned in this pass
                        for (path, start), n_deterministic in chunk_allocations_for_event.items():
                            rows_in_chunk = chunk_info_map.get((path, start), 0)
                            if rows_in_chunk == 0: continue

                            # Cap 1: By total remaining needed for this event overall
                            n_capped1 = min(n_deterministic, temp_still_needed_for_event)
                            # Cap 2: By rows available in this specific chunk
                            n_capped2 = min(n_capped1, rows_in_chunk)
                            # Cap 3: By remaining target specifically for this pass
                            n_to_sample = max(0, min(n_capped2, target_for_pass - processed_in_event_pass))

                            if n_to_sample > 0:
                                task_key = (path, start)
                                if task_key not in final_sampling_needs: final_sampling_needs[task_key] = {}
                                final_sampling_needs[task_key][original_event_id] = n_to_sample

                                # Update tracking variables
                                temp_still_needed_for_event -= n_to_sample
                                processed_in_event_pass += n_to_sample

                    # 4. 创建最终的任务列表
                    tasks_for_pool.clear() # Clear list before populating for the pass
                    for (path, start), needs_dict in final_sampling_needs.items():
                        corresponding_chunk = next((c for c in chunks_in_pass if c[0] == path and c[1] == start), None)
                        if not corresponding_chunk: continue
                        end_row = corresponding_chunk[2]

                        # *** 保留 TaskGen 调试日志 ***
                        if energy_bin == '0.6MeV' and 0 in needs_dict and start < 3 * chunk_rows:
                             logger.debug(f"DEBUG TaskGen 0.6MeV E:0 P:{pass_num} Chunk:[{start}:{end_row}] n_final(deterministic):{needs_dict[0]}")

                        task_seed = self._get_next_seed()
                        task_info = {
                            'part_file_path': path, 'chunk_start_row': start, 'chunk_end_row': end_row,
                            'sampling_needs': needs_dict, # {event_id: n_deterministic_capped}
                            'dataset_name': dataset_name, 'num_layers': self.num_layers, 'rng_seed': task_seed
                        }
                        tasks_for_pool.append(task_info)

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
                                    # ... (结果聚合逻辑不变) ...
                                    if energy_bin == '0.6MeV' and 0 in actual_counts:
                                         chunk_id = f"{os.path.basename(task_submitted['part_file_path'])}[{task_submitted['chunk_start_row']}:{task_submitted['chunk_end_row']}]"; logger.debug(f"DEBUG AGGREGATE: Received count={actual_counts[0]} for E:0 Bin:0.6MeV from worker (Chunk: {chunk_id}).")
                                    for original_event_id, summed_array in partial_sums.items():
                                        plan_idx = self.event_id_to_index.get(original_event_id);
                                        if plan_idx is not None: self.event_sums[plan_idx] += summed_array
                                        else: logger.error(f"聚合错误：无效 event_id {original_event_id} 从 worker 返回。")
                                    for original_event_id, count in actual_counts.items():
                                        if count > 0:
                                            plan_idx = self.event_id_to_index.get(original_event_id)
                                            if plan_idx is not None:
                                                current_collected = self.event_rows_collected[plan_idx].get(energy_bin, 0); new_collected = current_collected + count
                                                if energy_bin == '0.6MeV' and original_event_id == 0: logger.debug(f"DEBUG AGGREGATE: Updating 0.6MeV E:{original_event_id} (idx:{plan_idx}). Adding count={count} to current={current_collected}. New total={new_collected}")
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
        # ... (同上) ...
        state = {
            'event_sums': self.event_sums, 'event_rows_collected': self.event_rows_collected,
            'last_completed_bin_index': last_completed_bin_index,
            'original_num_events': self.original_num_events, 'event_id_map': self.event_id_to_index
        }
        save_checkpoint(self.checkpoint_path, state)
# --- End SamplingExecutor Class ---

# --- 阶段 3: 完成与输出 (独立函数 - 保持不变) ---
def finalize_run(run_info: Dict[str, Any], final_event_sums: List[np.ndarray], final_event_counts: List[Dict[str, int]], execution_successful: bool):
    # ... (同上) ...
    logger.info("--- 开始阶段 3: 完成与输出 ---")
    output_dir = run_info['output_dir']; run_name = run_info['run_name']; num_layers = run_info['num_layers']
    sampling_plans = run_info['sampling_plans']; original_num_events = run_info['original_num_events']
    num_valid_events = run_info['num_valid_events']; checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILENAME)
    output_file_count = 0
    if num_valid_events > 0 and len(final_event_sums) == num_valid_events and len(final_event_counts) == num_valid_events:
        logger.info(f"正在为 {num_valid_events} 个有效事件生成输出文件...")
        event_id_to_index = run_info['event_id_to_index'] # Get map
        for plan_idx, plan in enumerate(sampling_plans):
            original_event_id = plan['event_id']; final_sum = final_event_sums[plan_idx]
            actual_counts_this_event = final_event_counts[plan_idx]; relevant_counts = {}
            for energy_key in plan['samples_per_bin']: relevant_counts[energy_key] = actual_counts_this_event.get(energy_key, 0)
            if original_event_id == 0: logger.debug(f"DEBUG SAVE: Final counts dict passed to save_output_file for E0: {relevant_counts}")
            if any(v > 0 for v in relevant_counts.values()):
                save_output_file(output_dir, run_name, original_event_id, plan, final_sum, relevant_counts, num_layers, original_num_events)
                output_file_count += 1
            else: logger.warning(f"事件 {original_event_id} (idx {plan_idx}) 未抽取数据，不生成文件。")
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
            finalize_run(run_info, final_event_sums, final_event_counts, execution_successful)
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
    main_orchestrator(args_list=["--config", "config_test.json", "--workers", "16", "--resume", "--log_level", "INFO"])