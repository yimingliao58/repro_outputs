"""
放在 Wan2.2 repo 根目录(和 generate.py 同级),或者放在任意目录并把该目录加进 PYTHONPATH。
Python 解释器启动时会自动 import sitecustomize(如果它在 sys.path 上),
这样不用改 Wan2.2 的源码就能在 CUDA/cuDNN 初始化之前把确定性开关打开。

如果这几行开关导致某些 kernel 报 "not implemented for deterministic algorithms" 的报错,
把 warn_only 保持 True 即可(仅警告不报错),代价是那几个 op 不保证 bit-exact,
但通常不影响视觉一致性。
"""
import os
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
