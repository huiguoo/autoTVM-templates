import logging
import sys

import torch
import numpy as np
import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing

# the module is called `autotvm`
from tvm import autotvm

from numpy import median
import timeit
from functools import partial, reduce
from operator import mul

from numbers import Integral
def get_pad_tuple(padding, kernel):
    """Common code to get the pad option
    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    kernel : tuple of int
        Conv kernel size
    Returns
    -------
    pad_top : int
        Padding size on top
    pad_left : int
        Padding size on left
    pad_down : int
        Padding size on down.
    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left

def equal_const_int(expr, value):
    """Returns if expr equals value.
    Parameters
    ----------
    expr : tvm.Expr
        The input expression.
    Returns
    -------
    equal : bool
        Whether they equals.
    """
    if isinstance(expr, Integral):
        return expr == value
    if not isinstance(expr, tvm.tir.IntImm):
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
        return False
    return expr.value == value

def get_const_int(expr):
    """Verifies expr is integer and get the constant value.
    Parameters
    ----------
    expr : tvm.Expr or int
        The input expression.
    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(expr, Integral):
        return expr
    if not isinstance(expr, tvm.tir.IntImm):
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
        raise ValueError("Expect value to be constant int")
    return int(expr.value)

def pad(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput"):
    """Pad Input with zeros.
    Parameters
    ----------
    data : tvm.te.Tensor
        n-D input, can be any layout.
    pad_before : list / tuple of n ints
        Pad width on each dimension to pad the before the axis begin.
    pad_after : list / tuple of n ints, optional
        Pad width each dimension to pad the after the axis end.
    pad_value : float, optional
        The value to be padded.
    name : str, optional
        The name prefix operators generated
    Returns
    -------
    Output : tvm.te.Tensor
        n-D, the same layout as Input.
    """
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError(
            "Input dimension and pad_before dismatch : %d vs %d" % (n, len(pad_before))
        )
    if len(pad_after) != n:
        raise ValueError("Input dimension and pad_after dismatch : %d vs %d" % (n, len(pad_before)))
    ana = tvm.arith.Analyzer()
    dshape = []
    for dim in data.shape:
        if isinstance(dim, tvm.tir.Any):
            dshape.append(tvm.te.size_var("dim"))
        else:
            dshape.append(dim)
    out_shape = tuple(ana.simplify(dshape[i] + pad_before[i] + pad_after[i]) for i in range(n))
    pad_value = (
        pad_value
        if isinstance(pad_value, tvm.tir.PrimExpr)
        else tvm.tir.const(pad_value, data.dtype)
    )

    def _pad(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                index_tuple.append(indices[i])
            else:
                index_tuple.append(indices[i] - pad_before[i])
                not_zero.append(indices[i] >= pad_before[i])
                not_zero.append(indices[i] < data.shape[i] + pad_before[i])
        if not_zero:
            not_zero = tvm.tir.all(*not_zero)
            return tvm.tir.if_then_else(not_zero, data(*index_tuple), pad_value)
        return data(*index_tuple)

    return te.compute(out_shape, _pad, name=name)

def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm or Var, returns tuple of int or Var.
    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.
    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    ret = []
    ana = None
    for elem in in_tuple:
        if isinstance(elem, (tvm.tir.Var, tvm.tir.expr.Any)):
            ret.append(elem)
        elif not isinstance(elem, (tvm.tir.IntImm, int)):
            ana = tvm.arith.Analyzer() if ana is None else ana
            elem = ana.simplify(elem)
            if not isinstance(elem, tvm.tir.IntImm):
                ret.append(elem)
            else:
                ret.append(get_const_int(elem))
        else:
            ret.append(get_const_int(elem))
    return tuple(ret)

def unpack_NCHWc_to_nchw(packed_out, out_dtype):
    """Unpack conv2d_NCHWc output from layout NCHWc to NCHW
    Parameters
    ----------
    packed_out : tvm.te.Tensor
        The output tensor of conv2d_NCHWc.
    out_dtype : str
        The output dtype.
    Returns
    -------
    unpacked_out : tvm.te.Tensor
        The unpacked output tensor in NCHW layout.
    """
    n, oc_chunk, oh, ow, oc_bn = get_const_tuple(packed_out.shape)

    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv

    oshape = (n, oc_chunk * oc_bn, oh, ow)
    unpacked_out = te.compute(
        oshape,
        lambda n, c, h, w: packed_out[n, idxdiv(c, oc_bn), h, w, idxmod(c, oc_bn)].astype(
            out_dtype
        ),
        name="output_unpack",
        tag="unpack_nchwc",
    )
    return unpacked_out

@autotvm.template("local/conv")  # 1. use a decorator
def conv2d_depthwise(N, H, W, CO, CI, KH, KW, strides, padding, groups):
    assert N == 1, "Only consider batch_size = 1 in this template"

    in_data = te.placeholder((N, CI, H, W), name="data")
    in_kernel = te.placeholder((CO, CI//groups, KH, KW), name="kernel")
    #conv = topi.nn.group_conv2d_nchw(data, kernel, strides, padding, dilation=1, groups=groups, out_dtype="float32")

    ### conv args
    batch, in_channel, in_height, in_width = N, CI, H, W
    out_channel, channel_multiplier, filter_height, filter_width = CO, CI//groups, KH, KW

    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HSTR, WSTR = strides
    dilation=1
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)

    dilated_kernel_h = (filter_height - 1) * dh + 1
    dilated_kernel_w = (filter_width - 1) * dw + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    out_height = (in_height + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (in_width + WPAD - dilated_kernel_w) // WSTR + 1
    out_dtype="float32"

    ### search space in compute begin
    cfg = autotvm.get_config()
    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", out_channel, num_outputs=2)
    cfg.define_split("tile_ow", out_width, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    cfg.define_knob("unroll_kw", [True, False])
    ### search space in compute end

    pr_ic, pr_oc, pr_ow, pr_unroll_kw = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1], cfg["tile_ow"].size[-1], cfg["unroll_kw"].val
    print(f"tile_ic = {pr_ic}, tile_oc = {pr_oc}, tile_ow = {pr_ow}, unroll_kw = {pr_unroll_kw}")

    in_channel_block = cfg["tile_ic"].size[-1]
    in_channel_chunk = in_channel // in_channel_block
    out_channel_block = cfg["tile_oc"].size[-1]
    out_channel_chunk = out_channel // out_channel_block
    dshape = (batch, in_channel_chunk, in_height, in_width, in_channel_block)
    kshape = (out_channel_chunk, 1, filter_height, filter_width, 1, out_channel_block)

    data = te.compute(
        (batch, in_channel_chunk, in_height, in_width, in_channel_block),
        lambda b, ico, ih, iw, ici: in_data[
            b,
            ico* in_channel_block + ici,
            ih,
            iw
            ].astype(in_data.dtype),
            name="data_NCHWc",
            tag="pack_data_as_NCHWc",
            )
    kernel = te.compute(
        (out_channel_chunk, 1, filter_height, filter_width, 1, out_channel_block),
        lambda oco, cst1, h, w, cst2, oci: in_kernel[
            oco* out_channel_block + oci,
            0,
            h,
            w
            ].astype(in_kernel.dtype),
            name="kernel_NCHWc",
            tag="pack_kernel_as_NCHWc",
            )

    # padding stage
    DOPAD = pad_top != 0 or pad_left != 0 or pad_down != 0 or pad_right != 0
    if DOPAD:
        pad_before = [0, 0, pad_top, pad_left, 0]
        pad_after = [0, 0, pad_down, pad_right, 0]
        data_pad = pad(data, pad_before, pad_after, name="PaddedInput")
    else:
        data_pad = data


    # depthconv stage
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    kh = te.reduce_axis((0, filter_height), name="kh")
    kw = te.reduce_axis((0, filter_width), name="kw")
    Output = te.compute(
        (batch, out_channel_chunk, out_height, out_width, out_channel_block),
        lambda b, oco, oh, ow, oci: te.sum(
            (
                data_pad[
                    b,
                    idxdiv(
                        idxdiv(oco * out_channel_block + oci, channel_multiplier), in_channel_block
                    ),
                    oh * HSTR + kh * dh,
                    ow * WSTR + kw * dw,
                    idxmod(
                        idxdiv(oco * out_channel_block + oci, channel_multiplier), in_channel_block
                    ),
                ].astype(out_dtype)
                * kernel[oco, 0, kh, kw, 0, oci].astype(out_dtype)
            ),
            axis=[kh, kw],
        ),
        name="DepthwiseConv2d",
        tag="depthwise_conv2d_NCHWc",
    )
    conv = unpack_NCHWc_to_nchw(Output, out_dtype)
    s = te.create_schedule([conv.op])

    print("Original compute stmt:")
    print(tvm.lower(s, [in_data, in_kernel, conv], simple_mode=True))

    #### scheduling started
    tile_ow, oc_bn = cfg["tile_ow"].size[-1], cfg["tile_oc"].size[-1]
    unroll_kw = cfg["unroll_kw"].val

    #### schedule data_pad
    batch, ic_chunk, ih, iw, ic_block = s[data_pad].op.axis
    s[data_pad].vectorize(ic_block)
    parallel_axis = s[data_pad].fuse(batch, ic_chunk, ih)
    s[data_pad].parallel(parallel_axis)
    #print("scheduled data_pad:")
    #print(tvm.lower(s, [in_data, in_kernel, conv], simple_mode=True))

    #### scheule conv_NCHWc
    C, O = Output, conv
    CC = s.cache_write(C, "global")

    _, ic_chunk, oh, ow, ic_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=tile_ow)
    s[C].reorder(ic_chunk, oh, ow_chunk, ow_block, ic_block)
    s[C].vectorize(ic_block)
    parallel_axis = s[C].fuse(ic_chunk, oh)
    s[C].parallel(parallel_axis)
    #print("scheduled conv_NCHWc:")
    #print(tvm.lower(s, [in_data, in_kernel, conv], simple_mode=True))

    s[CC].compute_at(s[C], ow_chunk)
    #print("cache write conv_NCHWc:")
    #print(tvm.lower(s, [in_data, in_kernel, conv], simple_mode=True))

    # the ow axis in the cached block CC is the ow_block in C
    _, ic_chunk, oh, ow, ic_block = s[CC].op.axis
    kh, kw = s[CC].op.reduce_axis
    s[CC].reorder(ic_chunk, oh, kh, kw, ow, ic_block)
    if unroll_kw:
        s[CC].unroll(kw)
    s[CC].vectorize(ic_block)
    s[CC].unroll(ow)
    #print("conv_NCHWc schedule:")
    #print(tvm.lower(s, [in_data, in_kernel, conv], simple_mode=True))

    #### scheule conv_NCHW
    batch, oc, oh, ow = s[O].op.axis
    ow_chunk, ow_block = s[O].split(ow, factor=tile_ow)
    oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
    s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[O].fuse(oc_chunk, oh)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)
    s[O].parallel(parallel_axis)
    #print("Final schedule:")
    #print(tvm.lower(s, [in_data, in_kernel, conv], simple_mode=True))

    return s, [in_data, in_kernel, conv]

# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# the last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding, groups = 1, 14, 14, 120, 120, 5, 5, (1, 1), (2, 2), 120

#(96, 96, (5, 5), (2, 2), (2, 2), (1, 1), 96, False, 'zeros'),"(1, 96, 28, 28)"
#(240, 240, (5, 5), (1, 1), (2, 2), (1, 1), 240, False, 'zeros'),"(1, 240, 14, 14)"
#(120, 120, (5, 5), (1, 1), (2, 2), (1, 1), 120, False, 'zeros')","(1, 120, 14, 14)"
#(144, 144, (5, 5), (1, 1), (2, 2), (1, 1), 144, False, 'zeros'),"(1, 144, 14, 14)"
#(288, 288, (5, 5), (2, 2), (2, 2), (1, 1), 288, False, 'zeros'),"(1, 288, 14, 14)"
#(576, 576, (5, 5), (1, 1), (2, 2), (1, 1), 576, False, 'zeros'),"(1, 576, 7, 7)"
#N, H, W, CO, CI, KH, KW, strides, padding, groups = 1, 28, 28, 96, 96, 5, 5, (2, 2), (2, 2), 96
#N, H, W, CO, CI, KH, KW, strides, padding, groups = 1, 14, 14, 240, 240, 5, 5, (1, 1), (2, 2), 240
#N, H, W, CO, CI, KH, KW, strides, padding, groups = 1, 14, 14, 120, 120, 5, 5, (1, 1), (2, 2), 120
#N, H, W, CO, CI, KH, KW, strides, padding, groups = 1, 14, 14, 144, 144, 5, 5, (1, 1), (2, 2), 144
#N, H, W, CO, CI, KH, KW, strides, padding, groups = 1, 14, 14, 288, 288, 5, 5, (2, 2), (2, 2), 288
#N, H, W, CO, CI, KH, KW, strides, padding, groups = 1, 7, 7, 576, 576, 5, 5, (1, 1), (2, 2), 576

task = autotvm.task.create(
    "local/conv", args=(N, H, W, CO, CI, KH, KW, strides, padding, groups), target="llvm"
)
print(task.config_space)


#measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))
# Use local gpu, measure 10 times for every config to reduce variance
# The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
)

# Begin tuning, log records to file `conv2d.log`
# During tuning we will also try many invalid configs, so you are expected to
# see many error reports. As long as you can see non-zero GFLOPS, it is okay.
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=200,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("conv2d.log")],
)

autotvm.record.pick_best("conv2d.log", "result")
# apply history best from log file
with autotvm.apply_history_best("conv2d.log"):
    with tvm.target.Target("llvm"):
        s, arg_bufs = conv2d_depthwise(N, H, W, CO, CI, KH, KW, strides, padding, groups)
        print("Autotuner transformed stmt:")
        print(tvm.lower(s, arg_bufs, simple_mode=True))
        func = tvm.build(s, arg_bufs)

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

#### tvm measure performance
# Evaluate execution time
#evaluator = func.time_evaluator(func.entry_name, tvm.cpu(), min_repeat_ms=500)
#print(
#    "Execution time of this operator: %.3f ms"
#    % (np.median(evaluator(data_tvm, kernel_tvm, out_tvm).results) * 1000)
#)
