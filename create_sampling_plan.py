# sampling_planner.py
import os
import sys
import json
import math
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from numpy.random import Generator, PCG64, SeedSequence
from scipy.stats import norm
import argparse

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(process)d] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- 常量 ---

SAMPLING_PLAN_FILENAME = 'sampling_plan.json'
HDF5_INDEX_FILENAME: str = "hdf5_index.json"

def load_json(path: str) -> Dict[str, Any]:
    logger.info(f"正在加载 JSON 文件: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"错误: 文件未找到 - {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"错误: 解析 JSON 文件失败 - {path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载 JSON 文件时发生未知错误 - {path}")
        sys.exit(1)

class SamplingPlanner:
    """负责根据配置生成抽样计划"""
    # ... (复制 SamplingPlanner 类的 __init__ 和 _parse_methodX, generate, _log_debug_plan 方法) ...
    def __init__(self, config: Dict[str, Any], hdf5_index: Dict[str, Any]):
        # ... (同上) ...
        self.config = config
        self.hdf5_index = hdf5_index
        self.num_events = config['num_datasets_to_generate']
        self.method = config['generation_method']
        self.method_config = config.get(self.method)
        self.base_seed = config.get('random_seed', None)
        if not self.method_config:
            raise ValueError(f"配置中找不到方法 '{self.method}' 的详细信息")

    # (_parse_method1, _parse_method2, _parse_method3 as methods)
    def _parse_method1(self, rng: Generator) -> List[Dict[str, int]]:
        # ... (同上) ...
        plans = []
        method_config = self.method_config
        levels = method_config['energy_levels_MeV']
        min_counts = method_config['min_particles_per_level']
        max_counts = method_config['max_particles_per_level']
        if not (len(levels) == len(min_counts) == len(max_counts)):
            raise ValueError("Method 1 配置错误: energy_levels_MeV, min_particles_per_level, max_particles_per_level 长度必须一致")
        for i in range(self.num_events):
            event_plan = {}
            for j, level in enumerate(levels):
                level_name = f"{level:.1f}MeV"
                actual_min = min(min_counts[j], max_counts[j])
                actual_max = max(min_counts[j], max_counts[j])
                if actual_min < 0 or actual_max < 0: logger.warning(f"Method 1, E{i}, L{level_name}: Count range invalid ({actual_min}, {actual_max}), using max(0,...)"); actual_min = max(0, actual_min); actual_max = max(0, actual_max)
                count = int(rng.integers(actual_min, actual_max + 1))
                event_plan[level_name] = count
            plans.append(event_plan)
        return plans

    def _parse_method2(self, rng: Generator) -> List[Dict[str, int]]:
        # ... (同上) ...
        plans = []
        method_config = self.method_config
        levels = method_config['energy_levels_MeV']; probs = np.array(method_config['probabilities'])
        total_min = method_config['total_particles_min']; total_max = method_config['total_particles_max']
        actual_total_min = max(0, min(total_min, total_max)); actual_total_max = max(0, total_min, total_max)
        if len(levels) != len(probs): raise ValueError("Method 2 lengths mismatch")
        if np.any(probs < 0): probs = np.abs(probs); logger.warning("Method 2 negative probs found")
        probs_sum = np.sum(probs)
        if not math.isclose(probs_sum, 1.0, abs_tol=1e-9):
            if probs_sum > 1e-9: probs = probs / probs_sum
            else: probs = np.ones_like(levels) / len(levels); logger.warning("Method 2 probs sum zero")
        probs = np.maximum(probs, 0)
        if not math.isclose(np.sum(probs), 1.0, abs_tol=1e-9): probs = probs / np.sum(probs)
        for i in range(self.num_events):
            total_particles = int(rng.integers(actual_total_min, actual_total_max + 1))
            if total_particles == 0: counts = np.zeros(len(probs), dtype=int)
            else:
                if not math.isclose(np.sum(probs), 1.0): probs = probs / np.sum(probs)
                try: counts = rng.multinomial(total_particles, probs)
                except ValueError as e:
                     logger.error(f"Multinomial error event {i}: {e}, probs={probs}, total={total_particles}. Using fallback.")
                     probs_fallback = np.ones_like(levels) / len(levels); counts = rng.multinomial(total_particles, probs_fallback)
            event_plan = {}
            for j, level in enumerate(levels): event_plan[f"{level:.1f}MeV"] = int(counts[j])
            plans.append(event_plan)
        return plans

    def _parse_method3(self, rng: Generator) -> List[Dict[str, int]]:
        # ... (同上) ...
        plans = []
        method_config = self.method_config
        levels_mev = np.array(method_config['energy_levels_MeV'])
        total_min = method_config['total_particles_min']; total_max = method_config['total_particles_max']
        actual_total_min = max(0, min(total_min, total_max)); actual_total_max = max(0, total_min, total_max)
        dist_info = method_config['distribution_function']; dist_type = dist_info.get('type', '').lower()
        if dist_type != 'gaussian': raise NotImplementedError(f"Method 3 only supports 'gaussian', not '{dist_type}'")
        mean_range = dist_info['parameters']['mean']; stddev_range = dist_info['parameters']['stddev']
        mean_min = min(mean_range['min'], mean_range['max']); mean_max = max(mean_range['min'], mean_range['max'])
        stddev_min = min(stddev_range['min'], stddev_range['max']); stddev_max = max(stddev_range['min'], stddev_range['max'])
        if stddev_min <= 0: stddev_min = 1e-6; logger.warning("Method 3 stddev min <= 0, using 1e-6")
        for i in range(self.num_events):
            mean = rng.uniform(mean_min, mean_max); stddev = rng.uniform(stddev_min, stddev_max)
            weights = norm.pdf(levels_mev, loc=mean, scale=stddev)
            if np.sum(weights) < 1e-9 or not np.all(np.isfinite(weights)): weights = np.ones_like(levels_mev); logger.warning(f"Event {i} Method 3 using equal weights")
            probs_sum = np.sum(weights)
            if probs_sum > 1e-9: probs = weights / probs_sum
            else: probs = np.ones_like(levels_mev) / len(levels_mev)
            probs = np.maximum(probs, 0)
            if not math.isclose(np.sum(probs), 1.0): probs = probs / np.sum(probs)
            total_particles = int(rng.integers(actual_total_min, actual_total_max + 1))
            if total_particles == 0: counts = np.zeros(len(probs), dtype=int)
            else:
                if not math.isclose(np.sum(probs), 1.0): probs = probs / np.sum(probs)
                try: counts = rng.multinomial(total_particles, probs)
                except ValueError as e:
                     logger.error(f"Multinomial error event {i}: {e}, probs={probs}, total={total_particles}. Using fallback.")
                     probs_fallback = np.ones_like(levels_mev) / len(levels_mev); counts = rng.multinomial(total_particles, probs_fallback)
            event_plan = {}
            for j, level in enumerate(levels_mev): event_plan[f"{level:.1f}MeV"] = int(counts[j])
            plans.append(event_plan)
        return plans

    def generate(self) -> List[Dict[str, Any]]:
        # ... (同上) ...
        ss = SeedSequence(self.base_seed)
        rng = Generator(PCG64(ss))
        logger.info(f"正在为 {self.num_events} 个抽样事件生成计划，使用方法: {self.method}")
        if self.method == 'method1': plans_per_bin = self._parse_method1(rng)
        elif self.method == 'method2': plans_per_bin = self._parse_method2(rng)
        elif self.method == 'method3': plans_per_bin = self._parse_method3(rng)
        else: raise NotImplementedError(f"未实现的方法: {self.method}")
        final_plans = []; all_required_levels = set(); original_num_events = self.num_events
        for i in range(original_num_events):
            filtered_plan = {k: v for k, v in plans_per_bin[i].items() if v > 0}
            plan = {'event_id': i, 'samples_per_bin': filtered_plan}
            final_plans.append(plan)
            if filtered_plan: all_required_levels.update(plan['samples_per_bin'].keys())
            else: logger.warning(f"Event {i}: 生成的抽样计划为空。")
        missing_levels = all_required_levels - set(self.hdf5_index.keys())
        if missing_levels:
            hdf5_index_file = os.path.join(self.config.get('input_data_directory', './'), HDF5_INDEX_FILENAME) # Use constant
            raise ValueError(f"配置中需要的能量级别在 HDF5 索引 ({hdf5_index_file}) 中缺失: {sorted(list(missing_levels))}")
        valid_final_plans = [p for p in final_plans if p['samples_per_bin']]
        num_valid_events = len(valid_final_plans)
        if num_valid_events < original_num_events: logger.warning(f"{original_num_events - num_valid_events} 个事件因抽样计划为空而被移除。")
        if num_valid_events == 0: logger.error("没有有效的抽样计划生成。")
        logger.info(f"成功生成 {num_valid_events} 个有效的抽样计划。")
        self._log_debug_plan(valid_final_plans, all_required_levels)
        return valid_final_plans

    def _log_debug_plan(self, valid_plans, required_levels):
        # ... (同上) ...
        if '0.6MeV' in required_levels and valid_plans:
            first_event_id_with_06 = -1; target_count_06 = 'N/A'
            for plan in valid_plans:
                if '0.6MeV' in plan['samples_per_bin']: first_event_id_with_06 = plan['event_id']; target_count_06 = plan['samples_per_bin']['0.6MeV']; break
            if first_event_id_with_06 != -1: logger.debug(f"DEBUG PLAN: 抽样计划 Event {first_event_id_with_06} 中 '0.6MeV' 的目标抽样数 (N_sample_bin) = {target_count_06}")
            else: logger.debug("DEBUG PLAN: 在有效的抽样计划中未找到需要抽取 0.6MeV 的事件。")


def analyze_plan(sampling_plans: List[Dict[str, Any]], config: Dict[str, Any]):
    """对生成的抽样计划进行统计分析"""
    # ... (同上一个脚本) ...
    logger.info("\n--- 开始抽样计划统计分析 ---")
    num_events = len(sampling_plans)
    if num_events == 0: logger.info("无有效计划可供分析。"); return
    all_energies = set(); total_particles_per_event = []; particles_per_bin: Dict[str, List[int]] = {}
    for plan in sampling_plans:
        event_total = sum(plan['samples_per_bin'].values()); total_particles_per_event.append(event_total)
        for energy, count in plan['samples_per_bin'].items():
            all_energies.add(energy)
            if energy not in particles_per_bin: particles_per_bin[energy] = []
            particles_per_bin[energy].append(count)
    total_particles_per_event_np = np.array(total_particles_per_event)
    logger.info(f"分析的总事件数: {num_events}")
    logger.info(f"每事件总粒子数:")
    logger.info(f"  - 最小值: {np.min(total_particles_per_event_np):,}")
    logger.info(f"  - 最大值: {np.max(total_particles_per_event_np):,}")
    logger.info(f"  - 平均值: {np.mean(total_particles_per_event_np):,.2f}")
    logger.info(f"  - 标准差: {np.std(total_particles_per_event_np):,.2f}")
    logger.info(f"  - 中位数: {np.median(total_particles_per_event_np):,}")
    logger.info("各能量点抽样数统计:")
    sorted_energies = sorted(list(all_energies), key=lambda x: float(x.replace('MeV','')))
    for energy in sorted_energies:
        counts_this_bin = particles_per_bin.get(energy);
        if not counts_this_bin: logger.info(f"  - {energy}: 未在任何事件中抽样。"); continue
        counts_np = np.array(counts_this_bin)
        logger.info(f"  - {energy} (出现 {len(counts_np)} 次):")
        logger.info(f"    - 最小值: {np.min(counts_np):,}"); logger.info(f"    - 最大值: {np.max(counts_np):,}")
        logger.info(f"    - 平均值: {np.mean(counts_np):,.2f}"); logger.info(f"    - 标准差: {np.std(counts_np):,.2f}")
        logger.info(f"    - 中位数: {np.median(counts_np):,}")
        if energy == '0.6MeV': # 特别统计 8999 (如果 chunk 数是 8999)
             # 假设块数是 8999
             count_eq_chunks = np.sum(counts_np == 8999)
             if count_eq_chunks > 0: logger.info(f"    - *** 等于 8999 (块数?) 的次数: {count_eq_chunks} / {len(counts_np)} ({count_eq_chunks / len(counts_np) * 100:.1f}%) ***")
    logger.info("--- 抽样计划统计分析结束 ---")


def save_plan_to_file(plans: List[Dict[str, Any]], output_dir: str):
    """将抽样计划保存到 JSON 文件"""
    # ... (同上一个脚本) ...
    plan_filepath = os.path.join(output_dir, SAMPLING_PLAN_FILENAME)
    try:
        os.makedirs(output_dir, exist_ok=True) # 确保目录存在
        with open(plan_filepath, 'w', encoding='utf-8') as f:
            serializable_plans = json.loads(json.dumps(plans, default=int)) # Convert numpy ints
            json.dump(serializable_plans, f, indent=4, ensure_ascii=False)
        logger.info(f"抽样计划已保存至: {plan_filepath}")
    except Exception as e:
        logger.error(f"无法保存抽样计划到 {plan_filepath}: {e}")


def main_plan_generator(args=None, args_list=None):
    """主函数：生成、分析并保存抽样计划
    
    Args:
        args: 预先解析好的参数对象
        args_list: 字符串列表形式的参数，将被parser解析
    """
    parser = argparse.ArgumentParser(description="生成抽样计划并进行统计分析。")
    parser.add_argument("--config", required=True, help="指向 config.json 配置文件的路径")
    parser.add_argument("--output_dir", help="保存抽样计划文件的目录（默认使用config中的output_directory）")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")

    # 如果提供了预解析的参数，直接使用；否则解析命令行或提供的参数列表
    if args is None:
        if args_list is not None:
            args = parser.parse_args(args_list)
        else:
            args = parser.parse_args()

    # --- 用户测试代码 (可选, 用于独立测试此脚本) ---
    # args.config = "config_test.json"
    # --- 结束用户测试代码 ---

    logger.setLevel(getattr(logging, args.log_level.upper()))
    if args.log_level == "DEBUG": logger.info("已启用 DEBUG 日志级别。")

    try:
        config = load_json(args.config)
        input_data_dir = config['input_data_directory']
        hdf5_index_path = os.path.join(input_data_dir, HDF5_INDEX_FILENAME)
        hdf5_index = load_json(hdf5_index_path)

        output_dir = args.output_dir if args.output_dir else config['output_directory']
        # 确保输出目录存在，以便保存计划
        os.makedirs(output_dir, exist_ok=True)

        # 实例化 Planner 并生成计划
        planner = SamplingPlanner(config, hdf5_index)
        sampling_plans = planner.generate() # 生成计划

        if sampling_plans:
             analyze_plan(sampling_plans, config) # 分析计划
             save_plan_to_file(sampling_plans, output_dir) # 保存计划
             return sampling_plans  # 返回生成的计划，便于程序化调用
        else:
             logger.error("未能生成有效的抽样计划。")
             sys.exit(1) # Exit if no plans generated

    except Exception as e:
        logger.exception("生成抽样计划过程中发生错误")
        sys.exit(1)

if __name__ == "__main__":
    # 这个脚本只做规划和分析
    try:
        import numpy
        import scipy # For method 3
    except ImportError as e:
        print(f"错误: 缺少必要的库: {e}. 请确保 numpy, scipy 已安装。", file=sys.stderr)
        sys.exit(1)

    # 以下是几种调用方式的示例
    
    # 1. 原始方式：直接解析命令行参数
    # main_plan_generator()
    
    # 2. 传入字符串列表方式（取消注释使用）
    main_plan_generator(args_list=["--config", "config_test.json", "--log_level", "DEBUG"])
    
    # 3. 传入已解析的参数对象方式（取消注释使用）
    # import argparse
    # parsed_args = argparse.Namespace()
    # parsed_args.config = "config.json"
    # parsed_args.output_dir = None
    # parsed_args.log_level = "INFO"
    # main_plan_generator(args=parsed_args)