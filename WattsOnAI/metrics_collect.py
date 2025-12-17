import os
import subprocess
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from resources_consumption_record import timing_decorator

@timing_decorator
def get_gpu_info(indices=[]):
    """
    获取基本GPU信息，返回一个字典列表，每个字典包含一个GPU的信息
    """
    command = [
        "nvidia-smi",
        "--query-gpu=name,index,power.draw,utilization.gpu,utilization.memory,"
        "pcie.link.gen.current,pcie.link.width.current,temperature.gpu,"
        "temperature.memory,clocks.gr,clocks.mem,clocks.current.sm",
        "--format=csv"
    ]

    if indices:  # 如果indices不为空
        id_str = ",".join(map(str, indices))  # 将indices转换为逗号分隔的字符串
        command.extend(["-i=" + id_str])  # 添加--id参数
    
    try:
        result = subprocess.check_output(command, shell=False).decode('utf-8')
        lines = result.strip().split("\n")
        headers = lines[0].split(", ")
        gpu_data_list = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split(", ")
            gpu_data = {}
            for i, header in enumerate(headers):
                gpu_data[header] = values[i]
            gpu_data_list.append(gpu_data)
        return gpu_data_list
    except subprocess.CalledProcessError as e:
        print(f"Error running basic command: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

@timing_decorator
def get_dcgm_metrics_group(indices=None):
    """
    使用 dcgmi dmon 命令获取 GPU 的高级性能指标（组1）
    包括：设备名称、SM活跃度、SM占用率、Tensor Core活跃度、DRAM活跃度、
    PCIe发送字节数、PCIe接收字节数、NVLink发送字节数、NVLink接收字节数、
    内存控制器利用率、GPU利用率、内存利用率、温度、时钟频率

    返回:
    list: 包含每个GPU指标的字典列表
    """
    if indices is None:
        indices = []

    # 构造命令
    command = [
        "dcgmi", "dmon",
        "-e", "50,155,203,252,251,237,238,150,140,100,101,1002,1003,1004,1005,1009,1010,1011,1012,204",
        "-c", "1"
    ]
    if indices:
        id_str = ",".join(map(str, indices))
        command.extend(["-i", id_str])

    try:
        # 执行命令并获取输出
        output = subprocess.check_output(command, shell=False).decode('utf-8')
        lines = [l for l in output.strip().splitlines() if l.strip()]

        # 定位表头行
        header_line = next(i for i, l in enumerate(lines) if "SMACT" in l and "SMOCC" in l)
        raw_headers = lines[header_line].split()

        # 定义从原始字段到新字段的映射，包括 DVNAM
        header_map = {
            'DVNAM': 'name',                  # 设备名称
            'POWER': 'power.draw [W]',
            'GPUTL': 'utilization.gpu [%]',
            'FBUSD': None,  # 用于计算内存利用率，不单独输出
            'FBFRE': None,
            'PCILG': 'pcie.link.gen.current',
            'PCILW': 'pcie.link.width.current',
            'TMPTR': 'temperature.gpu',
            'MMTMP': 'temperature.memory',
            'SMCLK': 'clocks.current.sm [MHz]',
            'MMCLK': 'clocks.current.memory [MHz]',
            'SMACT': 'sm_active',
            'SMOCC': 'sm_occupancy',
            'TENSO': 'tensor_active',
            'DRAMA': 'dram_active',
            'PCITX': 'pcie_tx_bytes',
            'PCIRX': 'pcie_rx_bytes',
            'NVLTX': 'nvlink_tx_bytes',
            'NVLRX': 'nvlink_rx_bytes',
            'MCUTL': 'utilization.memory [%]' 
        }

        gpu_list = []
        for row in lines[header_line + 1:]:
            parts = row.split()
            if not parts or not parts[0].startswith('GPU'):
                continue
            # GPU索引
            gpu_idx = int(parts[1])
            # 查找首个数值字段位置，以区分DVNAM
            num_idx = None
            for j in range(2, len(parts)):
                try:
                    float(parts[j])
                    num_idx = j
                    break
                except ValueError:
                    continue
            if num_idx is None:
                continue
            # 设备名称
            dvnam = " ".join(parts[2:num_idx])
            # 剩余全部数值字段
            numeric_vals = parts[num_idx:]
            # 构建原始值字典
            raw_vals = {'DVNAM': dvnam}
            needed = len(raw_headers) - 2
            for k, key in enumerate(raw_headers[2:2 + needed]):
                if k < len(numeric_vals):
                    raw_vals[key] = numeric_vals[k]

            # 构建输出数据
            gpu_data = {
                'index': gpu_idx
            }
            # 遍历映射
            for raw_key, field in header_map.items():
                if field is None or raw_key not in raw_vals:
                    continue
                val = raw_vals[raw_key]
                try:
                    f = float(val)
                except ValueError:
                    gpu_data[field] = val
                    continue
                # 单位格式化
                if field == 'utilization.gpu [%]':
                    gpu_data[field] = f"{f:.2f} %"
                elif field in ['sm_active', 'sm_occupancy', 'tensor_active', 'dram_active']:
                    gpu_data[field] = f"{f * 100:.2f} %"
                elif field in ['pcie_tx_bytes', 'pcie_rx_bytes', 'nvlink_tx_bytes', 'nvlink_rx_bytes']:
                    gpu_data[field] = f"{f / 1024**3:.2f} GB/s"
                elif field in ['temperature.gpu', 'temperature.memory']:
                    gpu_data[field] = f"{f:.2f} °C"
                elif field in ['clocks.current.sm [MHz]', 'clocks.current.memory [MHz]']:
                    gpu_data[field] = f"{f:.0f} MHz"
                elif field in ['power.draw [W]']:
                    gpu_data[field] = f"{f:.2f} W"
                elif field in ['utilization.memory [%]']:
                    gpu_data[field] = f"{f:.2f} %"
                else:
                    gpu_data[field] = val

            # 计算内存利用率
            try:
                used = float(raw_vals.get('FBUSD', 0))
                free = float(raw_vals.get('FBFRE', 0))
                mem_util = used / (used + free) * 100 if (used + free) > 0 else 0
                gpu_data['usage.memory [%]'] = f"{mem_util:.2f} %"
            except Exception:
                pass

            # 新增分组时钟，与SMCLK相同
            gpu_data['clocks.current.graphics [MHz]'] = gpu_data.get('clocks.current.sm [MHz]')

            gpu_list.append(gpu_data)

        return gpu_list

    except subprocess.CalledProcessError as e:
        print(f"执行 dcgmi dmon 命令时出错: {e}")
        return []
    except Exception as e:
        print(f"处理 dcgmi dmon 输出时发生意外错误: {e}")
        return []


@timing_decorator
def get_dcgm_fp64_active(indices=[]):
    """
    使用 dcgmi dmon 命令获取 GPU 的 FP64 活跃度指标
    返回:
    list: 包含每个GPU的 FP64 活跃度指标的字典列表
    """
    command = [
        "dcgmi", "dmon", "-e", "1006", "-c", "1"
    ]

    if indices:  # 如果indices不为空
        id_str = ",".join(map(str, indices))  # 将indices转换为逗号分隔的字符串
        command.extend(["-i " + id_str])  # 添加--id参数

    try:
        result = subprocess.check_output(command, shell=False).decode('utf-8')
        lines = result.strip().split("\n")
        # 查找表头行和数据开始行
        header_line = None
        data_start_line = None
        for i, line in enumerate(lines):
            if "FP64A" in line:
                header_line = i
            if "GPU " in line and header_line is not None:
                data_start_line = i
                break
        if header_line is None or data_start_line is None:
            print("无法解析 dcgmi dmon FP64 活跃度输出的表头或数据")
            return []
        # 解析数据行
        gpu_data_list = []
        for i in range(data_start_line, len(lines)):
            line = lines[i].strip()
            if not line or "---" in line:
                continue
            parts = line.split()
            if len(parts) < 3 or not parts[0].startswith("GPU"):
                continue  
            gpu_index = int(parts[1])
            value = parts[2]
            # 创建GPU数据字典
            gpu_data = {'index': str(gpu_index)}
            # 添加FP64活跃度指标
            try:
                fp64_value = float(value) * 100
                gpu_data['fp64_active'] = f"{fp64_value:.2f} %"
            except ValueError:
                gpu_data['fp64_active'] = "N/A"
            gpu_data_list.append(gpu_data)
        return gpu_data_list
    
    except subprocess.CalledProcessError as e:
        print(f"执行 dcgmi dmon FP64 活跃度命令时出错: {e}")
        return []
    except Exception as e:
        print(f"处理 dcgmi dmon FP64 活跃度输出时发生意外错误: {e}")
        return []

@timing_decorator
def get_dcgm_fp32_active(indices=[]):
    """
    使用 dcgmi dmon 命令获取 GPU 的 FP32 活跃度指标
    返回:
    list: 包含每个GPU的 FP32 活跃度指标的字典列表
    """
    command = [
        "dcgmi", "dmon", "-e", "1007", "-c", "1"
    ]

    if indices:  # 如果indices不为空
        id_str = ",".join(map(str, indices))  # 将indices转换为逗号分隔的字符串
        command.extend(["-i " + id_str])  # 添加--id参数

    try:
        result = subprocess.check_output(command, shell=False).decode('utf-8')
        lines = result.strip().split("\n")
        # 查找表头行和数据开始行
        header_line = None
        data_start_line = None
        for i, line in enumerate(lines):
            if "FP32A" in line:
                header_line = i
            if "GPU " in line and header_line is not None:
                data_start_line = i
                break
        if header_line is None or data_start_line is None:
            print("无法解析 dcgmi dmon FP32 活跃度输出的表头或数据")
            return []
        # 解析数据行
        gpu_data_list = []
        for i in range(data_start_line, len(lines)):
            line = lines[i].strip()
            if not line or "---" in line:
                continue  
            parts = line.split()
            if len(parts) < 3 or not parts[0].startswith("GPU"):
                continue  
            gpu_index = int(parts[1])
            value = parts[2]
            # 创建GPU数据字典
            gpu_data = {'index': str(gpu_index)}
            # 添加FP32活跃度指标
            try:
                fp32_value = float(value) * 100
                gpu_data['fp32_active'] = f"{fp32_value:.2f} %"
            except ValueError:
                gpu_data['fp32_active'] = "N/A"
            
            gpu_data_list.append(gpu_data)
        return gpu_data_list
    
    except subprocess.CalledProcessError as e:
        print(f"执行 dcgmi dmon FP32 活跃度命令时出错: {e}")
        return []
    except Exception as e:
        print(f"处理 dcgmi dmon FP32 活跃度输出时发生意外错误: {e}")
        return []
    
@timing_decorator
def get_dcgm_fp16_active(indices=[]):
    """
    使用 dcgmi dmon 命令获取 GPU 的 FP32 活跃度指标
    返回:
    list: 包含每个GPU的 FP32 活跃度指标的字典列表
    """
    command = [
        "dcgmi", "dmon", "-e", "1008", "-c", "1"
    ]
    if indices:  # 如果indices不为空
        id_str = ",".join(map(str, indices))  # 将indices转换为逗号分隔的字符串
        command.extend(["-i " + id_str])  # 添加--id参数
    try:
        result = subprocess.check_output(command, shell=False).decode('utf-8')
        lines = result.strip().split("\n")
        # 查找表头行和数据开始行
        header_line = None
        data_start_line = None
        for i, line in enumerate(lines):
            if "FP16A" in line:
                header_line = i
            if "GPU " in line and header_line is not None:
                data_start_line = i
                break
        if header_line is None or data_start_line is None:
            print("无法解析 dcgmi dmon FP16 活跃度输出的表头或数据")
            return []
        # 解析数据行
        gpu_data_list = []
        for i in range(data_start_line, len(lines)):
            line = lines[i].strip()
            if not line or "---" in line:
                continue  
            parts = line.split()
            if len(parts) < 3 or not parts[0].startswith("GPU"):
                continue  
            gpu_index = int(parts[1])
            value = parts[2]
            # 创建GPU数据字典
            gpu_data = {'index': str(gpu_index)}
            # 添加FP16活跃度指标
            try:
                fp16_value = float(value) * 100
                gpu_data['fp16_active'] = f"{fp16_value:.2f} %"
            except ValueError:
                gpu_data['fp16_active'] = "N/A"
            
            gpu_data_list.append(gpu_data)
        return gpu_data_list
    
    except subprocess.CalledProcessError as e:
        print(f"执行 dcgmi dmon FP16 活跃度命令时出错: {e}")
        return []
    except Exception as e:
        print(f"处理 dcgmi dmon FP16 活跃度输出时发生意外错误: {e}")
        return []

@timing_decorator
def get_cpu_usage_info(sample_interval=1.0):
    """
    获取CPU信息
    返回:
    float: CPU使用率
    """
    try:
        if sample_interval <= 0:
            sample_interval = 1.0
        cpu_usage = psutil.cpu_percent(interval=sample_interval)
        
        # ✅ Fallback: If first call returns 0, try non-blocking mode
        if cpu_usage == 0.0:
            # Initialize with a quick call
            psutil.cpu_percent(interval=0.1)
            time.sleep(0.1)
            # Then get actual value
            cpu_usage = psutil.cpu_percent(interval=0)
        
        return str(cpu_usage) + " %"
    except Exception as e:
        print(f"Error getting CPU usage info: {e}")
        return None

@timing_decorator
def get_dram_usage_info():
    """
    获取DRAM使用情况
    返回:
    float: DRAM使用率
    """
    try:
        info = psutil.virtual_memory()
        dram_usage = info.percent
        return str(dram_usage) + " %"
    except Exception as e:
        print(f"Error getting DRAM usage info: {e}")
        return None


@timing_decorator
def get_cpu_power_info(sample_interval=1):
    """
    高级 CPU 功耗估算（基于频率和使用率）
    更接近实际功耗，适合容器环境
    """
    try:
        powercap_path = "/sys/class/powercap"
        
        # 首先尝试 RAPL
        if os.path.exists(powercap_path):
            domains = []
            for entry in os.listdir(powercap_path):
                if entry.startswith("intel-rapl:") and ":" not in entry[len("intel-rapl:"):]:
                    domain_path = os.path.join(powercap_path, entry)
                    energy_path = os.path.join(domain_path, "energy_uj")
                    if os.path.exists(energy_path):
                        try:
                            with open(energy_path, "r") as f:
                                energy_start = int(f.read().strip())
                            timestamp_start = time.time()
                            domains.append({
                                "path": energy_path,
                                "energy_start": energy_start,
                                "timestamp_start": timestamp_start
                            })
                        except (PermissionError, ValueError):
                            continue
            
            if domains:
                time.sleep(sample_interval)
                total_power_w = 0.0
                valid_domains = 0
                
                for domain in domains:
                    try:
                        with open(domain["path"], "r") as f:
                            energy_end = int(f.read().strip())
                        timestamp_end = time.time()
                        delta_time = timestamp_end - domain["timestamp_start"]
                        
                        if delta_time <= 0:
                            continue
                        
                        delta_energy_uj = energy_end - domain["energy_start"]
                        
                        # 处理计数器溢出
                        if delta_energy_uj < 0:
                            max_energy_path = os.path.join(
                                os.path.dirname(domain["path"]),
                                "max_energy_range_uj"
                            )
                            if os.path.exists(max_energy_path):
                                with open(max_energy_path, "r") as f:
                                    max_energy = int(f.read().strip())
                                delta_energy_uj += max_energy + 1
                            else:
                                continue
                        
                        power_w = (delta_energy_uj * 1e-6) / delta_time
                        
                        # 合理性检查 (0-500W per domain)
                        if 0 <= power_w <= 500:
                            total_power_w += power_w
                            valid_domains += 1
                            
                    except Exception:
                        continue
                
                if valid_domains > 0 and total_power_w > 0:
                    return f"{total_power_w:.2f} W"
        
        # ============================================================
        # 高级估算：考虑 CPU 频率
        # ============================================================
        
        # 获取 CPU 信息
        cpu_count_physical = psutil.cpu_count(logical=False) or 1
        cpu_count_logical = psutil.cpu_count(logical=True) or 1
        
        # 获取 CPU 频率
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                current_freq_mhz = cpu_freq.current
                max_freq_mhz = cpu_freq.max if cpu_freq.max > 0 else cpu_freq.current
            else:
                # 如果无法获取频率，使用默认值
                current_freq_mhz = 2400
                max_freq_mhz = 3500
        except Exception:
            current_freq_mhz = 2400
            max_freq_mhz = 3500
        
        # 频率比例
        freq_ratio = current_freq_mhz / max_freq_mhz if max_freq_mhz > 0 else 1.0
        
        # 获取 CPU 使用率（每核心）
        per_core_percent = psutil.cpu_percent(interval=sample_interval, percpu=True)
        avg_cpu_percent = sum(per_core_percent) / len(per_core_percent)
        
        # 估算参数
        base_tdp_per_core = 15.0  # 基础 TDP（瓦特/核心）
        
        # 功耗估算公式：
        # P = P_idle + (P_max - P_idle) * utilization * (freq/freq_max)^3
        # 频率的立方关系来自 CMOS 功耗公式：P ∝ CV²f
        
        idle_power_per_core = base_tdp_per_core * 0.15  # 空闲时约 15% TDP
        max_power_per_core = base_tdp_per_core
        
        # 每个核心的功耗
        core_powers = []
        for core_percent in per_core_percent:
            utilization = core_percent / 100.0
            freq_factor = freq_ratio ** 2.5  # 使用 2.5 次方作为近似
            
            core_power = idle_power_per_core + \
                        (max_power_per_core - idle_power_per_core) * \
                        utilization * freq_factor
            
            core_powers.append(core_power)
        
        # 总功耗
        total_estimated_power = sum(core_powers)
        
        return f"{total_estimated_power:.2f} W (freq-based estimate)"
    
    except Exception as e:
        print(f"Error in frequency-based estimation: {e}")
        import traceback
        traceback.print_exc()
        return "N/A"


@timing_decorator
def get_dram_power_info(sample_interval=1.0):
    """
    获取 DRAM 功耗（优先使用 RAPL，容器环境中使用估算）
    参数: sample_interval (float): 采样间隔（秒）
    返回: str: 功耗（瓦特）或估算值
    """
    try:
        # ============================================================
        # 方法 1: 尝试 RAPL（裸机环境）
        # ============================================================
        powercap_path = "/sys/class/powercap"
        if os.path.exists(powercap_path):
            domains = []
            for entry in os.listdir(powercap_path):
                domain_path = os.path.join(powercap_path, entry)
                name_path = os.path.join(domain_path, "name")
                if os.path.exists(name_path):
                    try:
                        with open(name_path, "r") as f:
                            name = f.read().strip()
                        if name == "dram":
                            energy_path = os.path.join(domain_path, "energy_uj")
                            if os.path.exists(energy_path):
                                with open(energy_path, "r") as f:
                                    energy_start = int(f.read().strip())
                                domains.append({
                                    "path": energy_path,
                                    "energy_start": energy_start,
                                    "timestamp_start": time.time()
                                })
                    except Exception:
                        continue
            
            if domains:
                time.sleep(sample_interval)
                total_power_w = 0.0
                valid_domains = 0
                
                for domain in domains:
                    try:
                        with open(domain["path"], "r") as f:
                            energy_end = int(f.read().strip())
                        timestamp_end = time.time()
                        delta_time = timestamp_end - domain["timestamp_start"]
                        
                        if delta_time <= 0:
                            continue
                        
                        delta_energy_uj = energy_end - domain["energy_start"]
                        
                        # 处理计数器溢出
                        if delta_energy_uj < 0:
                            max_energy_path = os.path.join(
                                os.path.dirname(domain["path"]),
                                "max_energy_range_uj"
                            )
                            if os.path.exists(max_energy_path):
                                with open(max_energy_path, "r") as f:
                                    max_energy = int(f.read().strip())
                                delta_energy_uj += max_energy + 1
                            else:
                                continue
                        
                        power_w = (delta_energy_uj * 1e-6) / delta_time
                        
                        # 合理性检查 (DRAM 通常 < 100W)
                        if 0 <= power_w <= 100:
                            total_power_w += power_w
                            valid_domains += 1
                            
                    except Exception:
                        continue
                
                if valid_domains > 0 and total_power_w > 0:
                    return f"{total_power_w:.2f} W"
        
        # ============================================================
        # 方法 2: 基于内存使用率估算（容器环境）
        # ============================================================
        
        # 获取内存信息
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024**3)
        used_memory_gb = memory_info.used / (1024**3)
        memory_percent = memory_info.percent
        
        # DRAM 功耗估算参数（基于典型值）
        # DDR4: ~3W per 8GB stick at idle, ~5W under load
        # 估算公式：每 8GB 内存在空闲时约 3W，满载时约 5W
        
        # 计算内存模块数量（假设每个模块 8GB）
        num_memory_modules = max(1, int(total_memory_gb / 8))
        
        # 基础功耗（空闲）和最大功耗
        idle_power_per_module = 3.0  # W
        max_power_per_module = 5.0   # W
        
        # 根据内存使用率计算功耗
        idle_power_total = idle_power_per_module * num_memory_modules
        max_power_total = max_power_per_module * num_memory_modules
        
        # 线性插值基于内存使用率
        estimated_power = idle_power_total + \
                         (max_power_total - idle_power_total) * (memory_percent / 100.0)
        
        # 调试信息（可选）
        # print(f"   Total memory: {total_memory_gb:.2f} GB")
        # print(f"   Used memory: {used_memory_gb:.2f} GB ({memory_percent:.1f}%)")
        # print(f"   Estimated modules: {num_memory_modules}")
        # print(f"   Estimated DRAM power: {estimated_power:.2f} W")
        
        return f"{estimated_power:.2f} W (est)"
    
    except Exception as e:
        print(f"Error getting/estimating DRAM power: {e}")
        return "N/A"


@timing_decorator
def parallel_collect_metrics(additional_metrics, indices=[], sample_interval=1):

    """
    并行收集硬件指标
    参数:
    additional_metrics (list): 额外需要收集的指标列表，可能包含 'fp64', 'fp32', 'fp16'
    返回:
    dict: 包含所有收集到的指标的字典
    """
    metrics = {}
    gpu_data_list = []

    # 确保是列表，避免 None
    additional_metrics = additional_metrics or []

    # 先根据要提交的任务算出总任务数
    num_tasks = 1  

    if 'CPU' in additional_metrics:
        num_tasks += 2  # cpu_usage + cpu_power
    if 'DRAM' in additional_metrics:
        num_tasks += 2  # dram_power + dram_usage
    # 每个 fp* 都是一个任务
    for m in ('fp64', 'fp32', 'fp16'):
        if m in additional_metrics:
            num_tasks += 1
    workers = min(max(num_tasks, 1), 8)
    
    # ✅ Pre-initialize CPU measurement (avoid first-call 0.0 issue)
    if 'CPU' in additional_metrics:
        try:
            _ = psutil.cpu_percent(interval=0.1)
        except Exception as e:
            print(f"   ⚠️ CPU initialization warning: {e}")

    # 创建任务映射
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:

        if 'Gdetails' not in additional_metrics:
            futures[executor.submit(get_gpu_info, indices)] = "gpu_info"

        # 根据 additional_metrics 添加额外的任务
        if 'CPU' in additional_metrics:
            futures[executor.submit(get_cpu_usage_info, sample_interval)] = "cpu_usage"
            futures[executor.submit(get_cpu_power_info, sample_interval)] = "cpu_power"
        if 'DRAM' in additional_metrics:
            futures[executor.submit(get_dram_power_info, sample_interval)] = "dram_power"
            futures[executor.submit(get_dram_usage_info)] = "dram_usage"

        if 'Gdetails' in additional_metrics:
            futures[executor.submit(get_dcgm_metrics_group, indices)] = "dcgm_metrics"

        if 'fp64' in additional_metrics:
            futures[executor.submit(get_dcgm_fp64_active, indices)] = "fp64_active"
        if 'fp32' in additional_metrics:
            futures[executor.submit(get_dcgm_fp32_active, indices)] = "fp32_active"
        if 'fp16' in additional_metrics:
            futures[executor.submit(get_dcgm_fp16_active, indices)] = "fp16_active"

        # 等待所有任务完成（带超时保护）
        # If sample_interval=1, need at least 2+ seconds for CPU/DRAM power
        timeout_value = max(20, sample_interval * 3 + 10)  # Dynamic timeout
        
        for future in as_completed(futures, timeout=timeout_value):
            key = futures[future]
            try:
                result = future.result()
                if key == 'gpu_info':
                    gpu_data_list = result
                elif key == 'dcgm_metrics':
                    gpu_data_list = result
                else:
                    metrics[key] = result
            except Exception as e:
                print(f"Failed to collect metric: {key} - {e}")
                metrics[key] = None

        # 后续和原来一样：把各类指标合并到 gpu_data_list 中
        if gpu_data_list:

            if metrics.get("fp64_active") is not None:
                for gpu_data in gpu_data_list:
                    gid = gpu_data.get('index')
                    for rec in metrics["fp64_active"]:
                        if rec.get('index') == str(gid):
                            gpu_data['fp64_active'] = rec.get('fp64_active', 'N/A')
                metrics.pop("fp64_active", None)

            if metrics.get("fp32_active") is not None:
                for gpu_data in gpu_data_list:
                    gid = gpu_data.get('index')
                    for rec in metrics["fp32_active"]:
                        if rec.get('index') == str(gid):
                            gpu_data['fp32_active'] = rec.get('fp32_active', 'N/A')
                metrics.pop("fp32_active", None)

            if metrics.get("fp16_active") is not None:
                for gpu_data in gpu_data_list:
                    gid = gpu_data.get('index')
                    for rec in metrics["fp16_active"]:
                        if rec.get('index') == str(gid):
                            gpu_data['fp16_active'] = rec.get('fp16_active', 'N/A')
                metrics.pop("fp16_active", None)

        metrics['gpu_info'] = gpu_data_list

    return metrics