import logging
import sys,os

import torch
import numpy as np
import tvm
from tvm import te, topi, testing, auto_scheduler
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing

# the module is called `autotvm`
from tvm import autotvm

from numpy import median
import timeit
from functools import partial, reduce
from operator import mul

@auto_scheduler.register_workload
def my_conv2d_depthwise(N, H, W, CO, CI, KH, KW, strides, padding, groups):
    assert N == 1, "Only consider batch_size = 1 in this template"

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI//groups, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

N, H, W, CO, CI, KH, KW, strides, padding, groups = 1, 7, 7, 576, 576, 5, 5, (1, 1), (2, 2), 576

target = tvm.target.Target("llvm")
task = auto_scheduler.SearchTask(
        func = my_conv2d_depthwise, args=(N, H, W, CO, CI, KH, KW, strides, padding, groups), target=target
)

log_file = "conv2d_eval.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=100,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
# Run auto-tuning (search)
task.tune(tune_option)

sch, args = task.apply_best(log_file)
print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))
print("\n\n Equivalent python schedule:")
print(task.print_best(log_file, print_mode="schedule"))
func = tvm.build(sch, args, target)

##### check correctness

#input data and weight
data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
kernel_np = np.random.uniform(size=(CO, CI//groups, KH, KW)).astype(np.float32)

data_py = torch.from_numpy(data_np)
kernel_py = torch.from_numpy(kernel_np)
with torch.no_grad():
    out_torch = torch.nn.functional.conv2d(data_py, kernel_py, stride=strides, padding=padding, groups=groups)

print("output shape:", list(out_torch.size()))
out_tvm = tvm.nd.empty(list(out_torch.size()))
func(tvm.nd.array(data_np), tvm.nd.array(kernel_np), out_tvm)

tvm.testing.assert_allclose(out_torch.detach().numpy(), out_tvm.asnumpy(), rtol=1e-4)

#### measure performance
times, repeat = 5000, 50
def torch_conv(data, kernel, strides, padding, groups):
    with torch.no_grad():
        out_torch = torch.nn.functional.conv2d(data, kernel, stride=strides, padding=padding, groups=groups)
data_tvm, kernel_tvm = tvm.nd.array(data_np), tvm.nd.array(kernel_np)
sec1 = median(timeit.repeat(lambda: torch_conv(data_py, kernel_py, strides, padding, groups), number=times, repeat=repeat))
sec2 = median(timeit.repeat(lambda: func(data_tvm, kernel_tvm, out_tvm), number=times, repeat=repeat))
product = partial(reduce, mul)
gflops = 2* product(list(out_torch.size()) +
                  [CI//groups, KH, KW]) * times / 1000000000.0
print(f"fglops={gflops}")
print(f"pytorch: {sec1}ms, autotuned: {sec2}ms")
pytorch, autotuned, speedup = gflops / sec1, gflops / sec2, sec1 / sec2
print(f"{pytorch:.1f} gflops => {autotuned:.1f} gflops ({speedup:.2f}x)")
import traceback, threading
thread_names = {t.ident: t.name for t in threading.enumerate()}
os._exit(0)

#### tvm evaluator
# Evaluate execution time
#evaluator = func.time_evaluator(func.entry_name, tvm.cpu(), min_repeat_ms=500)
#print(
#    "Execution time of this operator: %.3f ms"
#    % (np.median(evaluator(data_tvm, kernel_tvm, out_tvm).results) * 1000)
#)
