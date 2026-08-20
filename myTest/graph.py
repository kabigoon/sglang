import torch

# 确保 CUDA 可用
if not torch.cuda.is_available():
    raise RuntimeError("本示例需要 GPU 环境运行")

device = torch.device("cuda") # 默认指向0号GPU，device仅指向一个GPU

# 1. 定义一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10).to(device)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
model.eval()  # 设为评估模式

# 2. 分配【静态输入张量】（Static Input）
# 它的内存地址在整个生命周期内必须保持不变
static_input = torch.empty((1, 10), device=device)

# 3. 预热（Warmup）
# PyTorch 的 CUDA 显存分配器（Caching Allocator）在第一次运行算子时会动态申请内存。
# 在捕获 CUDA Graph 之前，必须进行几次“预热”运行，确保显存分配稳定，避免在捕获时触发新的内存分配。
for _ in range(10):
    static_output = model(static_input)

# 4. 捕获阶段（Capture）
g = torch.cuda.CUDAGraph()

# 使用 torch.cuda.graph 启动捕获上下文
with torch.cuda.graph(g):
    # 在这个上下文中执行的所有 CUDA kernel 都会被记录到 g 中
    # 注意：这里必须使用我们在步骤 2 中定义的 static_input
    static_output = model(static_input)

# 此时，CUDA Graph 已经将 static_input 的内存地址和 static_output 的内存地址绑定在了图中。

# 5. 重放阶段（Replay）
# 假设在实际推理中，我们不断收到新的动态输入数据：
dynamic_input_1 = torch.randn(1, 10, device=device)
dynamic_input_2 = torch.randn(1, 10, device=device)

print("--- 开始使用 CUDA Graph 进行推理 ---")

# ---- 推理数据 1 ----
# 错误做法：static_input = dynamic_input_1  <-- 这会改变 Python 变量指向的内存地址！
# 正确做法：使用 .copy_() 将数据拷贝到静态输入内存中
static_input.copy_(dynamic_input_1)

# 重放图（不再通过 Python 解释器逐个发射 Kernel，而是直接在 GPU 上执行整个图）
g.replay()

# 此时，计算结果已经自动写入了静态输出 static_output 中
# 我们可以直接读取或者拷贝出来
output_1 = static_output.clone()
print("输入 1 的推理结果：", output_1)

# ---- 推理数据 2 ----
static_input.copy_(dynamic_input_2)
g.replay()
output_2 = static_output.clone()
print("输入 2 的推理结果：", output_2)