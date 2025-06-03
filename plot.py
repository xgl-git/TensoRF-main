import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 指定 TensorBoard 日志路径（包含 events.out.tfevents... 的文件）
log_dir = './logs'  # 根据你的路径修改

# 加载日志数据
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# 查看所有可以读取的 scalar 标签（如：'train/loss', 'val/loss'）
print("Available tags:", ea.Tags()['scalars'])

# 读取特定的 loss 曲线（如 'train/loss'）
tag = 'train/mse'  # 根据你的 tag 名称修改
events = ea.Scalars(tag)

# 提取 steps 和对应的 loss 值
steps = [e.step for e in events]
loss_values = [e.value for e in events]

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(steps, loss_values, label=tag, color='blue')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


tag = 'train/PSNR'  # 根据你的 tag 名称修改
events = ea.Scalars(tag)

# 提取 steps 和对应的 loss 值
steps = [e.step for e in events]
loss_values = [e.value for e in events]

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(steps, loss_values, label=tag, color='blue')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Test PSNR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
