import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(
    rewards,
    algo_name,
    env_name,
    window=10,
    save_path=None,
    show_plot=True
):
    if not isinstance(rewards, (list, np.ndarray)) or len(rewards) == 0:
        print("BBQ了")
        return

    smoothed_rewards = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window + 1)
        window_rewards = rewards[start_idx:i+1]
        smoothed_rewards.append(np.mean(window_rewards))

    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False   
    plt.figure(figsize=(10, 6))

    plt.plot(rewards, label="单回合回报", alpha=0.4, color="#1f77b4")
    plt.plot(smoothed_rewards, label=f"滑动平均（窗口={window}）", color="#ff4b5c")
    plt.xlabel("训练回合 (Episode)", fontsize=12)
    plt.ylabel("单回合回报 (Episode Reward)", fontsize=12)
    plt.title(f"{algo_name} 算法在 {env_name} 环境的训练曲线", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"奖励曲线已保存至：{save_path}")
        except Exception as e:
            print(f"保存图片失败：{e}")

    if show_plot:
        plt.show()

    plt.close()