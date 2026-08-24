"""Fused continuous-optimizer update kernels exposed through a C ABI."""

from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float64]()
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def scale_range(noise: Ptr, dst: Ptr, start: Int, end: Int, sigma: Float64):
    var scale = SIMD[DType.float64, W](sigma)
    var i = start
    while i + W <= end:
        dst.store(i, noise.load[width=W](i) * scale)
        i += W
    while i < end:
        dst[i] = sigma * noise[i]
        i += 1


@export("mng_scale")
def mng_scale(noise_addr: Int, dst_addr: Int, n: Int, sigma: Float64) abi("C"):
    if n <= 0:
        return
    var noise = p(noise_addr)
    var dst = p(dst_addr)

    scale_range(noise, dst, 0, n, sigma)


def copy_range(src: Ptr, dst: Ptr, start: Int, end: Int):
    var i = start
    while i + W <= end:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < end:
        dst[i] = src[i]
        i += 1


def de_range(
    parent: Ptr,
    a: Ptr,
    b: Ptr,
    best: Ptr,
    dst: Ptr,
    start: Int,
    end: Int,
    f1: Float64,
    f2: Float64,
):
    var f1_vec = SIMD[DType.float64, W](f1)
    var f2_vec = SIMD[DType.float64, W](f2)
    var i = start
    while i + W <= end:
        var parent_vec = parent.load[width=W](i)
        dst.store(
            i,
            parent_vec
            + f1_vec * (a.load[width=W](i) - b.load[width=W](i))
            + f2_vec * (best.load[width=W](i) - parent_vec),
        )
        i += W
    while i < end:
        dst[i] = (
            parent[i]
            + f1 * (a[i] - b[i])
            + f2 * (best[i] - parent[i])
        )
        i += 1


@export("mng_de_binomial")
def mng_de_binomial(
    parent_addr: Int,
    a_addr: Int,
    b_addr: Int,
    best_addr: Int,
    random_addr: Int,
    dst_addr: Int,
    n: Int,
    f1: Float64,
    f2: Float64,
    cr: Float64,
    forced: Int,
) abi("C"):
    if n <= 0:
        return
    var parent = p(parent_addr)
    var a = p(a_addr)
    var b = p(b_addr)
    var best = p(best_addr)
    var random = p(random_addr)
    var dst = p(dst_addr)
    for i in range(n):
        if i != forced and random[i] > cr:
            dst[i] = parent[i]
        else:
            dst[i] = (
                parent[i]
                + f1 * (a[i] - b[i])
                + f2 * (best[i] - parent[i])
            )


@export("mng_de_twopoints")
def mng_de_twopoints(
    parent_addr: Int,
    a_addr: Int,
    b_addr: Int,
    best_addr: Int,
    dst_addr: Int,
    n: Int,
    f1: Float64,
    f2: Float64,
    lower: Int,
    upper: Int,
    parent_inside: Int,
) abi("C"):
    if n <= 0:
        return
    var parent = p(parent_addr)
    var a = p(a_addr)
    var b = p(b_addr)
    var best = p(best_addr)
    var dst = p(dst_addr)
    if parent_inside != 0:
        de_range(parent, a, b, best, dst, 0, lower, f1, f2)
        copy_range(parent, dst, lower, upper)
        de_range(parent, a, b, best, dst, upper, n, f1, f2)
    else:
        copy_range(parent, dst, 0, lower)
        de_range(parent, a, b, best, dst, lower, upper, f1, f2)
        copy_range(parent, dst, upper, n)


@export("mng_pso_update")
def mng_pso_update(
    x_addr: Int,
    speed_addr: Int,
    parent_best_addr: Int,
    global_best_addr: Int,
    rp_addr: Int,
    rg_addr: Int,
    new_speed_addr: Int,
    boxed_addr: Int,
    n: Int,
    omega: Float64,
    phip: Float64,
    phig: Float64,
) abi("C"):
    if n <= 0:
        return
    var x = p(x_addr)
    var speed = p(speed_addr)
    var parent_best = p(parent_best_addr)
    var global_best = p(global_best_addr)
    var rp = p(rp_addr)
    var rg = p(rg_addr)
    var new_speed = p(new_speed_addr)
    var boxed = p(boxed_addr)
    var omega_vec = SIMD[DType.float64, W](omega)
    var phip_vec = SIMD[DType.float64, W](phip)
    var phig_vec = SIMD[DType.float64, W](phig)
    var zero = SIMD[DType.float64, W](0.0)
    var one = SIMD[DType.float64, W](1.0)
    var i = 0
    while i + W <= n:
        var x_vec = x.load[width=W](i)
        var value = (
            omega_vec * speed.load[width=W](i)
            + phip_vec
            * rp.load[width=W](i)
            * (parent_best.load[width=W](i) - x_vec)
            + phig_vec
            * rg.load[width=W](i)
            * (global_best.load[width=W](i) - x_vec)
        )
        new_speed.store(i, value)
        boxed.store(i, min(max(x_vec + value, zero), one))
        i += W
    while i < n:
        var value = (
            omega * speed[i]
            + phip * rp[i] * (parent_best[i] - x[i])
            + phig * rg[i] * (global_best[i] - x[i])
        )
        new_speed[i] = value
        var position = x[i] + value
        if position < 0.0:
            position = 0.0
        elif position > 1.0:
            position = 1.0
        boxed[i] = position
        i += 1
