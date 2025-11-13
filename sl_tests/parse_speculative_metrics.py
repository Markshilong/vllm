#!/usr/bin/env python3
"""
解析 vllm_run.log 中的 Speculative metrics 信息并做统计
"""

import re
import sys
from statistics import mean, median, stdev

def parse_log_file(log_file_path):
    """解析日志文件，提取 Speculative metrics 信息"""
    pattern = re.compile(
        r'Speculative metrics: '
        r'Draft acceptance rate: ([\d.]+), '
        r'System efficiency: ([\d.]+), '
        r'Number of speculative tokens: (\d+), '
        r'Number of accepted tokens: (\d+), '
        r'Number of draft tokens: (\d+), '
        r'Number of emitted tokens: (\d+)\.'
    )
    
    metrics = {
        'draft_acceptance_rate': [],
        'system_efficiency': [],
        'speculative_tokens': [],
        'accepted_tokens': [],
        'draft_tokens': [],
        'emitted_tokens': []
    }
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            match = pattern.search(line)
            if match:
                draft_acceptance_rate = float(match.group(1))
                system_efficiency = float(match.group(2))
                speculative_tokens = int(match.group(3))
                accepted_tokens = int(match.group(4))
                draft_tokens = int(match.group(5))
                emitted_tokens = int(match.group(6))
                
                metrics['draft_acceptance_rate'].append(draft_acceptance_rate)
                metrics['system_efficiency'].append(system_efficiency)
                metrics['speculative_tokens'].append(speculative_tokens)
                metrics['accepted_tokens'].append(accepted_tokens)
                metrics['draft_tokens'].append(draft_tokens)
                metrics['emitted_tokens'].append(emitted_tokens)
    
    return metrics

def calculate_statistics(values):
    """计算统计信息"""
    if not values:
        return None
    
    stats = {
        'count': len(values),
        'min': min(values),
        'max': max(values),
        'mean': mean(values),
        'median': median(values),
    }
    
    if len(values) > 1:
        stats['stdev'] = stdev(values)
    else:
        stats['stdev'] = 0.0
    
    return stats

def print_statistics(metrics):
    """打印统计信息"""
    print("=" * 80)
    print("Speculative Metrics 统计报告")
    print("=" * 80)
    print()
    
    total_count = len(metrics['draft_acceptance_rate'])
    print(f"总记录数: {total_count}")
    print()
    
    # 定义指标的中文名称和单位
    metric_info = {
        'draft_acceptance_rate': ('Draft Acceptance Rate', '%', True),  # True表示需要转换为百分比
        'accepted_tokens': ('Number of Accepted Tokens', '个', False),
        'draft_tokens': ('Number of Draft Tokens', '个', False),
        'emitted_tokens': ('Number of Emitted Tokens', '个', False)
    }
    
    for key, (name, unit, is_percentage) in metric_info.items():
        values = metrics[key]
        stats = calculate_statistics(values)
        
        if stats:
            print(f"{name}:")
            print(f"  记录数: {stats['count']}")
            
            # 如果是百分比类型，乘以100显示
            if is_percentage:
                print(f"  最小值: {stats['min'] * 100:.2f} {unit}")
                print(f"  最大值: {stats['max'] * 100:.2f} {unit}")
                print(f"  平均值: {stats['mean'] * 100:.2f} {unit}")
                print(f"  中位数: {stats['median'] * 100:.2f} {unit}")
                if stats['stdev'] > 0:
                    print(f"  标准差: {stats['stdev'] * 100:.2f} {unit}")
            else:
                print(f"  最小值: {stats['min']:.4f} {unit}")
                print(f"  最大值: {stats['max']:.4f} {unit}")
                print(f"  平均值: {stats['mean']:.4f} {unit}")
                print(f"  中位数: {stats['median']:.4f} {unit}")
                if stats['stdev'] > 0:
                    print(f"  标准差: {stats['stdev']:.4f} {unit}")
            
            # 对于累计值（tokens），也显示总和
            if 'tokens' in key:
                print(f"  总和: {sum(values):,} {unit}")
            
            print()
    
    # 计算一些衍生指标
    if metrics['accepted_tokens'] and metrics['draft_tokens']:
        print("衍生指标:")
        print("-" * 80)
        
        # 总体接受率（基于累计值）
        total_accepted = sum(metrics['accepted_tokens'])
        total_draft = sum(metrics['draft_tokens'])
        if total_draft > 0:
            overall_acceptance_rate = total_accepted / total_draft
            print(f"总体 Draft Acceptance Rate (累计): {overall_acceptance_rate * 100:.2f}%")
        
        print()

def main():
    log_file = '/home/shilong/shilong_projects/vllm/sl_tests/vllm_run.log'
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    print(f"正在解析日志文件: {log_file}")
    print()
    
    try:
        metrics = parse_log_file(log_file)
        
        if not metrics['draft_acceptance_rate']:
            print("未找到任何 Speculative metrics 记录！")
            return
        
        print_statistics(metrics)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_file}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

