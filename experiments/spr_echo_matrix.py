import os
import itertools
import subprocess

def generate_and_submit_matrix():
    """
    负责生成 $3 \times 3 \times 3$ 的矩阵测试任务，并将其提交给 q.py 调度器。
    用于详尽测试深度、维度和特征聚合算法（包含解答深度 MLP 是否退化的疑问）。
    """
    depths = [3, 5, 7]
    dims = [64, 128, 256]
    agg_methods = ['complex_mul', 'simple_add', 'mlp_add']
    
    # 获取当前执行目录和调度器路径
    curr_dir = "/home/nio/log/holds/SameTime/experiments"
    q_script = "/home/nio/log/holds/SameTime/benchmark/wmt/q.py"
    
    tasks = list(itertools.product(depths, dims, agg_methods))
    print(f"总计规划了 {len(tasks)} 组矩阵测试任务。")
    
    for depth, dim, agg in tasks:
        # 定义任务名称和执行命令
        run_name = f"echo_L{depth}_D{dim}_{agg}"
        # TODO: 下一步我们需要编写 spr_echo.py 来接收这些参数
        cmd = f"python {curr_dir}/spr_echo.py --depth {depth} --dim {dim} --agg_method {agg} --run_name {run_name}"
        
        # 构建提交给 q.py 的命令 (假设 q.py 支持 'submit' 或直接跟命令，具体视 q.py 接口而定)
        # 这里先以打印或基础调用的方式演示
        submit_cmd = f"python {q_script} run '{cmd}'"
        
        print(f"[{run_name}] 准备提交命令: {submit_cmd}")
        # 在真正运行前，我们可以先生成脚本，之后你核对无误再放开 os.system
        # os.system(submit_cmd)

if __name__ == "__main__":
    generate_and_submit_matrix()
