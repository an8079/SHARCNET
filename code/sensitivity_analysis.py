# sensitivity_analysis.py
import os
import pandas as pd
import time
import json
import copy
import argparse  # 新增导入

# 从我们修改过的文件中导入关键函数
from main import main as run_single_experiment
from parser import parse_args


def run_sensitivity_analysis(cli_args):
    """
    为PPI2Complex模型执行超参数敏感性分析的主函数。
    """
    # 1. 定义实验配置
    datasets_to_run = ['HuRI', 'yeast']#'c_elegans',

    # 使用命令行传入的数据根目录
    output_base_dir = cli_args.output_dir
    if output_base_dir is None:
        # 默认保存到项目根目录下的 data/result (相对于 code/ 目录)
        output_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'result')
    base_data_path = cli_args.base_data_path
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"使用的根数据目录: {base_data_path}")
    print(f"结果将保存到: {output_base_dir}")

    # 定义要测试的超参数及其候选值
    params_to_test = {
        'min_clique_size': [1,2,3,4],
        'num_gcn_layers': [1,3,2,4],
        # 'num_hgnn_layers': [1, 2, 3, 4],
        # 'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        #'embedding_dim': [512,1024,64,32,128,256],
        # 'ricci_cutoff_min': [-0.9, -0.7, -0.5, -0.3, -0.1],
        # 'lambda_tcl': [0.01, 0.1, 0.5, 1.0, 5.0],
        # 'fusion_gamma': [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    # 2. 循环遍历每个数据集
    for dataset in datasets_to_run:
        print(f"\n{'=' * 20} 开始处理数据集: {dataset} {'=' * 20}\n")

        # 获取模型的默认参数配置
        default_args = parse_args([])

        # 强制更新数据根路径为命令行指定的值
        default_args.data_path = base_data_path

        all_results_for_dataset = []
        results_csv_path = os.path.join(output_base_dir, f'sensitivity_results_{dataset}.csv')

        # 3. 循环遍历每个要测试的超参数
        for param_name, param_values in params_to_test.items():
            print(f"\n--- 正在测试参数: {param_name} ---\n")

            # 4. 循环遍历该超参数的每个候选值
            for value in param_values:
                start_time = time.time()
                current_args = copy.deepcopy(default_args)
                current_args.dataset_name = dataset
                setattr(current_args, param_name, value)

                # 特殊处理：如果改变 embedding_dim, 同步更新 gcn_hidden_dims 和 hgnn_hidden_dims 的最后一层
                # 这是基于 parser.py 中的逻辑进行的更稳健的实现
                if param_name == 'embedding_dim':
                    # 更新 GCN 隐藏层
                    if current_args.num_gcn_layers > 1 and current_args.gcn_hidden_dims:
                        # 保持中间层不变，只改变最终输出前的维度（如果适用）
                        # 注意：我们模型的结构是最后一层直接输出 embedding_dim，所以这里不需要改
                        pass
                    elif current_args.num_gcn_layers == 1:
                        current_args.gcn_hidden_dims = []

                    # 更新 HGNN 隐藏层
                    if current_args.num_hgnn_layers > 1 and current_args.hgnn_hidden_dims:
                        pass
                    elif current_args.num_hgnn_layers == 1:
                        current_args.hgnn_hidden_dims = []

                print(f"*** 开始运行: Dataset={dataset}, Parameter={param_name}, Value={value} ***")

                performance_metrics = run_single_experiment(current_args)

                end_time = time.time()
                duration = end_time - start_time

                result_row = {
                    'dataset': dataset,
                    'param_tested': param_name,
                    'param_value': value,
                    'duration_seconds': round(duration, 2)
                }

                if performance_metrics:  # 检查是否成功返回结果
                    result_row.update(performance_metrics)
                else:
                    print("警告：本次运行未能返回有效的性能指标。")

                all_results_for_dataset.append(result_row)

                try:
                    df = pd.DataFrame(all_results_for_dataset)
                    df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
                    print(f"*** 结果已更新到: {results_csv_path} ***\n")
                except Exception as e:
                    print(f"警告：无法写入CSV文件: {e}")

        print(f"\n{'=' * 20} 数据集 {dataset} 的所有参数测试完成 {'=' * 20}\n")

    print("所有数据集的敏感性分析全部完成！")


if __name__ == '__main__':
    # 为 sensitivity_analysis.py 脚本自身创建参数解析器
    cli_parser = argparse.ArgumentParser(description="运行PPI2Complex超参数敏感性分析")
    cli_parser.add_argument('--base_data_path', type=str, required=True,
                            help='包含所有数据集子目录 (c_elegans, HuRI, yeast) 的根目录的绝对路径。')
    cli_parser.add_argument('--output_dir', type=str, default=None,
                            help='保存结果CSV文件的目录。默认为项目根目录下的 data/result。')

    args = cli_parser.parse_args()
    run_sensitivity_analysis(args)