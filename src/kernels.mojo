"""Numerical kernels exported through a small C ABI.

All arrays are caller-owned contiguous buffers. Addresses cross the ABI as
integers so exported functions remain non-parametric.
"""

from std.math import cos, exp, isnan, log, sin, sqrt
from std.sys.info import simd_width_of

comptime PI = 3.14159265358979323846264338327950288
comptime W = simd_width_of[DType.float64]()
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def _sum(x: FPtr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += x.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += x[i]
        i += 1
    return total


def _nan() -> Float64:
    var zero = 0.0
    return zero / zero


def _inf() -> Float64:
    var zero = 0.0
    return 1.0 / zero


@export("msc_convolve")
def msc_convolve(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int, m: Int) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    for k in range(n + m - 1):
        var lo = max(0, k - (m - 1))
        var hi = min(n - 1, k)
        var total = 0.0
        for i in range(lo, hi + 1):
            total += a[i] * b[k - i]
        dst[k] = total


@export("msc_lfilter")
def msc_lfilter(
    b_addr: Int,
    a_addr: Int,
    x_addr: Int,
    dst_addr: Int,
    nb: Int,
    na: Int,
    n: Int,
) abi("C"):
    var b = fp(b_addr)
    var a = fp(a_addr)
    var x = fp(x_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        var total = 0.0
        var j = 0
        while j < nb and j <= i:
            total += b[j] * x[i - j]
            j += 1
        j = 1
        while j < na and j <= i:
            total -= a[j] * dst[i - j]
            j += 1
        dst[i] = total / a[0]


def _is_power_two(n: Int) -> Bool:
    return n > 0 and (n & (n - 1)) == 0


def _fft_radix2(src: FPtr, dst: FPtr, n: Int, inverse: Bool):
    var j = 0
    for i in range(n):
        dst[2 * j] = src[2 * i]
        dst[2 * j + 1] = src[2 * i + 1]
        var bit = n >> 1
        while bit > 0 and (j & bit) != 0:
            j ^= bit
            bit >>= 1
        j ^= bit

    var length = 2
    while length <= n:
        var angle = (2.0 if inverse else -2.0) * PI / Float64(length)
        var wlen_r = cos(angle)
        var wlen_i = sin(angle)
        var half = length >> 1
        var base = 0
        while base < n:
            var wr0 = 1.0
            var wi0 = 0.0
            var wr1 = wlen_r
            var wi1 = wlen_i
            var step_r = wlen_r * wlen_r - wlen_i * wlen_i
            var step_i = 2.0 * wlen_r * wlen_i
            var k = 0
            while k + 2 <= half:
                var even = base + k
                var odd = even + half
                var orr = dst[2 * odd]
                var oii = dst[2 * odd + 1]
                var tr = wr0 * orr - wi0 * oii
                var ti = wr0 * oii + wi0 * orr
                var er = dst[2 * even]
                var ei = dst[2 * even + 1]
                dst[2 * even] = er + tr
                dst[2 * even + 1] = ei + ti
                dst[2 * odd] = er - tr
                dst[2 * odd + 1] = ei - ti
                var next_wr0 = wr0 * step_r - wi0 * step_i
                wi0 = wr0 * step_i + wi0 * step_r
                wr0 = next_wr0

                even += 1
                odd += 1
                orr = dst[2 * odd]
                oii = dst[2 * odd + 1]
                tr = wr1 * orr - wi1 * oii
                ti = wr1 * oii + wi1 * orr
                er = dst[2 * even]
                ei = dst[2 * even + 1]
                dst[2 * even] = er + tr
                dst[2 * even + 1] = ei + ti
                dst[2 * odd] = er - tr
                dst[2 * odd + 1] = ei - ti
                var next_wr1 = wr1 * step_r - wi1 * step_i
                wi1 = wr1 * step_i + wi1 * step_r
                wr1 = next_wr1
                k += 2
            while k < half:
                var even = base + k
                var odd = even + half
                var orr = dst[2 * odd]
                var oii = dst[2 * odd + 1]
                var tr = wr0 * orr - wi0 * oii
                var ti = wr0 * oii + wi0 * orr
                var er = dst[2 * even]
                var ei = dst[2 * even + 1]
                dst[2 * even] = er + tr
                dst[2 * even + 1] = ei + ti
                dst[2 * odd] = er - tr
                dst[2 * odd + 1] = ei - ti
                k += 1
            base += length
        length <<= 1


def _dft(src: FPtr, dst: FPtr, n: Int, inverse: Bool):
    var direction = 1.0 if inverse else -1.0
    for k in range(n):
        var sr = 0.0
        var si = 0.0
        for t in range(n):
            var angle = direction * 2.0 * PI * Float64(k * t) / Float64(n)
            var c = cos(angle)
            var s = sin(angle)
            sr += src[2 * t] * c - src[2 * t + 1] * s
            si += src[2 * t] * s + src[2 * t + 1] * c
        dst[2 * k] = sr
        dst[2 * k + 1] = si


@export("msc_fft")
def msc_fft(src_addr: Int, dst_addr: Int, n: Int, inverse_flag: Int) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var inverse = inverse_flag != 0
    if _is_power_two(n):
        _fft_radix2(src, dst, n, inverse)
    else:
        _dft(src, dst, n, inverse)
    if inverse:
        var scale = 1.0 / Float64(n)
        for i in range(2 * n):
            dst[i] *= scale


@export("msc_lu_solve")
def msc_lu_solve(
    a_addr: Int, b_addr: Int, n: Int, nrhs: Int
) abi("C") -> Int:
    var a = fp(a_addr)
    var b = fp(b_addr)
    for k in range(n):
        var pivot = k
        var best = abs(a[k * n + k])
        for i in range(k + 1, n):
            var value = abs(a[i * n + k])
            if value > best:
                best = value
                pivot = i
        if best <= 1.0e-15:
            return 0
        if pivot != k:
            var j = 0
            while j + W <= n:
                var tmp = a.load[width=W](k * n + j)
                a.store(k * n + j, a.load[width=W](pivot * n + j))
                a.store(pivot * n + j, tmp)
                j += W
            while j < n:
                var tmp = a[k * n + j]
                a[k * n + j] = a[pivot * n + j]
                a[pivot * n + j] = tmp
                j += 1
            var r = 0
            while r + W <= nrhs:
                var rhs_tmp = b.load[width=W](k * nrhs + r)
                b.store(k * nrhs + r, b.load[width=W](pivot * nrhs + r))
                b.store(pivot * nrhs + r, rhs_tmp)
                r += W
            while r < nrhs:
                var tmp = b[k * nrhs + r]
                b[k * nrhs + r] = b[pivot * nrhs + r]
                b[pivot * nrhs + r] = tmp
                r += 1
        for i in range(k + 1, n):
            var factor = a[i * n + k] / a[k * n + k]
            a[i * n + k] = factor
            var j = k + 1
            while j + W <= n:
                var updated = a.load[width=W](i * n + j) - (
                    SIMD[DType.float64, W](factor)
                    * a.load[width=W](k * n + j)
                )
                a.store(i * n + j, updated)
                j += W
            while j < n:
                a[i * n + j] -= factor * a[k * n + j]
                j += 1
            var r = 0
            while r + W <= nrhs:
                var rhs_updated = b.load[width=W](i * nrhs + r) - (
                    SIMD[DType.float64, W](factor)
                    * b.load[width=W](k * nrhs + r)
                )
                b.store(i * nrhs + r, rhs_updated)
                r += W
            while r < nrhs:
                b[i * nrhs + r] -= factor * b[k * nrhs + r]
                r += 1
    for rev in range(n):
        var i = n - 1 - rev
        var r = 0
        while r + W <= nrhs:
            var totals = b.load[width=W](i * nrhs + r)
            for j in range(i + 1, n):
                totals -= (
                    SIMD[DType.float64, W](a[i * n + j])
                    * b.load[width=W](j * nrhs + r)
                )
            totals /= SIMD[DType.float64, W](a[i * n + i])
            b.store(i * nrhs + r, totals)
            r += W
        while r < nrhs:
            var total = b[i * nrhs + r]
            for j in range(i + 1, n):
                total -= a[i * n + j] * b[j * nrhs + r]
            b[i * nrhs + r] = total / a[i * n + i]
            r += 1
    return 1


@export("msc_cholesky")
def msc_cholesky(a_addr: Int, n: Int) abi("C") -> Int:
    var a = fp(a_addr)
    for i in range(n):
        for j in range(i + 1):
            var total = a[i * n + j]
            for k in range(j):
                total -= a[i * n + k] * a[j * n + k]
            if i == j:
                if total <= 0.0:
                    return 0
                a[i * n + j] = sqrt(total)
            else:
                a[i * n + j] = total / a[j * n + j]
        for j in range(i + 1, n):
            a[i * n + j] = 0.0
    return 1


@export("msc_triangular_solve")
def msc_triangular_solve(
    a_addr: Int,
    b_addr: Int,
    n: Int,
    nrhs: Int,
    lower_flag: Int,
    transpose_flag: Int,
    unit_flag: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var effective_lower = (lower_flag != 0) != (transpose_flag != 0)
    if effective_lower:
        for i in range(n):
            for r in range(nrhs):
                var total = b[i * nrhs + r]
                for j in range(i):
                    var av = a[j * n + i] if transpose_flag != 0 else a[i * n + j]
                    total -= av * b[j * nrhs + r]
                if unit_flag == 0:
                    total /= a[i * n + i]
                b[i * nrhs + r] = total
    else:
        for rev in range(n):
            var i = n - 1 - rev
            for r in range(nrhs):
                var total = b[i * nrhs + r]
                for j in range(i + 1, n):
                    var av = a[j * n + i] if transpose_flag != 0 else a[i * n + j]
                    total -= av * b[j * nrhs + r]
                if unit_flag == 0:
                    total /= a[i * n + i]
                b[i * nrhs + r] = total


@export("msc_linear_assignment")
def msc_linear_assignment(
    cost_addr: Int,
    cols_addr: Int,
    u_addr: Int,
    v_addr: Int,
    p_addr: Int,
    way_addr: Int,
    minv_addr: Int,
    used_addr: Int,
    n: Int,
    m: Int,
) abi("C"):
    var cost = fp(cost_addr)
    var cols = ip(cols_addr)
    var u = fp(u_addr)
    var v = fp(v_addr)
    var p = ip(p_addr)
    var way = ip(way_addr)
    var minv = fp(minv_addr)
    var used = ip(used_addr)
    for j in range(m + 1):
        v[j] = 0.0
        p[j] = 0
        way[j] = 0
    for i in range(n + 1):
        u[i] = 0.0
    for ii in range(1, n + 1):
        p[0] = Int64(ii)
        var used_count = 0
        var j0 = 0
        for j in range(m + 1):
            minv[j] = 1.7976931348623157e308
            used[j] = 0
        while True:
            used[j0] = 1
            cols[used_count] = Int64(j0)
            used_count += 1
            var i0 = Int(p[j0])
            var delta = 1.7976931348623157e308
            var j1 = 0
            var j = 1
            while j + W <= m + 1:
                var unused = used.load[width=W](j).eq(0)
                var previous = minv.load[width=W](j)
                var current = (
                    cost.load[width=W]((i0 - 1) * m + (j - 1))
                    - SIMD[DType.float64, W](u[i0])
                    - v.load[width=W](j)
                )
                var improved = current.lt(previous) & unused
                var candidates = min(current, previous)
                minv.store(j, unused.select(candidates, previous))
                way.store(
                    j,
                    improved.select(
                        SIMD[DType.int64, W](Int64(j0)),
                        way.load[width=W](j),
                    ),
                )
                var eligible = unused.select(
                    candidates,
                    SIMD[DType.float64, W](1.7976931348623157e308),
                )
                if eligible.reduce_min() < delta:
                    for lane in range(W):
                        if eligible[lane] < delta:
                            delta = eligible[lane]
                            j1 = j + lane
                j += W
            while j < m + 1:
                if used[j] == 0:
                    var cur = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = Int64(j0)
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                j += 1
            j = 0
            var delta_vec = SIMD[DType.float64, W](delta)
            while j + W <= m + 1:
                var is_used = used.load[width=W](j).ne(0)
                minv.store(
                    j,
                    is_used.select(
                        minv.load[width=W](j),
                        minv.load[width=W](j) - delta_vec,
                    ),
                )
                j += W
            while j < m + 1:
                if used[j] == 0:
                    minv[j] -= delta
                j += 1
            for visited in range(used_count):
                var used_j = Int(cols[visited])
                u[Int(p[used_j])] += delta
                v[used_j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            var j1 = Int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    for i in range(n):
        cols[i] = -1
    for j in range(1, m + 1):
        if p[j] != 0:
            cols[Int(p[j]) - 1] = Int64(j - 1)


@export("msc_rosen")
def msc_rosen(x_addr: Int, n: Int) abi("C") -> Float64:
    var x = fp(x_addr)
    var total = 0.0
    for i in range(n - 1):
        var d = x[i + 1] - x[i] * x[i]
        var e = 1.0 - x[i]
        total += 100.0 * d * d + e * e
    return total


@export("msc_rosen_der")
def msc_rosen_der(x_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var x = fp(x_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        dst[i] = 0.0
    for i in range(n - 1):
        var d = x[i + 1] - x[i] * x[i]
        dst[i] += -400.0 * x[i] * d - 2.0 * (1.0 - x[i])
        dst[i + 1] += 200.0 * d


def _locate(x: FPtr, n: Int, q: Float64) -> Int:
    if q <= x[0]:
        return 0
    if q >= x[n - 1]:
        return n - 2
    var lo = 0
    var hi = n - 1
    while hi - lo > 1:
        var mid = (lo + hi) >> 1
        if x[mid] <= q:
            lo = mid
        else:
            hi = mid
    return lo


@export("msc_interp_linear")
def msc_interp_linear(
    x_addr: Int, y_addr: Int, q_addr: Int, dst_addr: Int, n: Int, m: Int
) abi("C"):
    var x = fp(x_addr)
    var y = fp(y_addr)
    var q = fp(q_addr)
    var dst = fp(dst_addr)
    for k in range(m):
        var i = _locate(x, n, q[k])
        var t = (q[k] - x[i]) / (x[i + 1] - x[i])
        dst[k] = y[i] + t * (y[i + 1] - y[i])


@export("msc_interp_nearest")
def msc_interp_nearest(
    x_addr: Int, y_addr: Int, q_addr: Int, dst_addr: Int, n: Int, m: Int
) abi("C"):
    var x = fp(x_addr)
    var y = fp(y_addr)
    var q = fp(q_addr)
    var dst = fp(dst_addr)
    for k in range(m):
        var i = _locate(x, n, q[k])
        dst[k] = y[i] if q[k] <= 0.5 * (x[i] + x[i + 1]) else y[i + 1]


@export("msc_pchip_eval")
def msc_pchip_eval(
    x_addr: Int,
    y_addr: Int,
    d_addr: Int,
    q_addr: Int,
    dst_addr: Int,
    n: Int,
    m: Int,
) abi("C"):
    var x = fp(x_addr)
    var y = fp(y_addr)
    var d = fp(d_addr)
    var q = fp(q_addr)
    var dst = fp(dst_addr)
    for k in range(m):
        var i = _locate(x, n, q[k])
        var h = x[i + 1] - x[i]
        var t = (q[k] - x[i]) / h
        var t2 = t * t
        var t3 = t2 * t
        dst[k] = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * y[i]
            + (t3 - 2.0 * t2 + t) * h * d[i]
            + (-2.0 * t3 + 3.0 * t2) * y[i + 1]
            + (t3 - t2) * h * d[i + 1]
        )


@export("msc_stats_summary")
def msc_stats_summary(
    x_addr: Int, result_addr: Int, n: Int, omit_nan: Int
) abi("C") -> Int:
    var x = fp(x_addr)
    var result = fp(result_addr)
    var count = 0
    var total = 0.0
    for i in range(n):
        if isnan(x[i]):
            if omit_nan == 0:
                for j in range(4):
                    result[j] = _nan()
                return 0
        else:
            total += x[i]
            count += 1
    if count == 0:
        for j in range(4):
            result[j] = _nan()
        return 0
    var mean = total / Float64(count)
    var m2 = 0.0
    var m3 = 0.0
    var m4 = 0.0
    for i in range(n):
        if not isnan(x[i]):
            var z = x[i] - mean
            var z2 = z * z
            m2 += z2
            m3 += z2 * z
            m4 += z2 * z2
    result[0] = mean
    result[1] = m2 / Float64(count)
    result[2] = m3 / Float64(count)
    result[3] = m4 / Float64(count)
    return count


@export("msc_zscore")
def msc_zscore(
    x_addr: Int, dst_addr: Int, n: Int, mean: Float64, scale: Float64
) abi("C"):
    var x = fp(x_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        dst[i] = (x[i] - mean) / scale


@export("msc_raw_moment")
def msc_raw_moment(
    x_addr: Int, n: Int, order: Int, center: Float64, omit_nan: Int
) abi("C") -> Float64:
    var x = fp(x_addr)
    var total = 0.0
    var count = 0
    for i in range(n):
        if isnan(x[i]):
            if omit_nan == 0:
                return _nan()
        else:
            var value = 1.0
            var delta = x[i] - center
            var exponent = 0
            while exponent < order:
                value *= delta
                exponent += 1
            total += value
            count += 1
    return total / Float64(count)


@export("msc_entropy")
def msc_entropy(pk_addr: Int, qk_addr: Int, n: Int, has_qk: Int) abi("C") -> Float64:
    var pk = fp(pk_addr)
    var qk = fp(qk_addr)
    var sp = _sum(pk, n)
    var sq = _sum(qk, n) if has_qk != 0 else 1.0
    var total = 0.0
    for i in range(n):
        var p = pk[i] / sp
        if p > 0.0:
            if has_qk != 0:
                var q = qk[i] / sq
                if q == 0.0:
                    return _inf()
                total += p * log(p / q)
            else:
                total -= p * log(p)
    return total


def _sift_down(values: FPtr, indices: IPtr, start: Int, end: Int):
    var root = start
    while root * 2 + 1 <= end:
        var child = root * 2 + 1
        if child + 1 <= end and values[child] < values[child + 1]:
            child += 1
        if values[root] < values[child]:
            var tv = values[root]
            values[root] = values[child]
            values[child] = tv
            var ti = indices[root]
            indices[root] = indices[child]
            indices[child] = ti
            root = child
        else:
            return


@export("msc_rankdata")
def msc_rankdata(
    values_addr: Int, indices_addr: Int, ranks_addr: Int, n: Int, method: Int
) abi("C"):
    var values = fp(values_addr)
    var indices = ip(indices_addr)
    var ranks = fp(ranks_addr)
    for i in range(n):
        indices[i] = Int64(i)
    var start = (n - 2) >> 1
    while start >= 0:
        _sift_down(values, indices, start, n - 1)
        start -= 1
    var end = n - 1
    while end > 0:
        var tv = values[end]
        values[end] = values[0]
        values[0] = tv
        var ti = indices[end]
        indices[end] = indices[0]
        indices[0] = ti
        end -= 1
        _sift_down(values, indices, 0, end)
    if method == 4:
        for i in range(n):
            var begin = i
            while begin > 0 and values[begin - 1] == values[i]:
                begin -= 1
            var finish = i + 1
            while finish < n and values[finish] == values[i]:
                finish += 1
            var rank = begin + 1
            for k in range(begin, finish):
                if indices[k] < indices[i]:
                    rank += 1
            ranks[Int(indices[i])] = Float64(rank)
        return
    var i = 0
    var dense_rank = 1
    while i < n:
        var j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        var rank = 0.5 * Float64(i + j + 1)
        if method == 1:
            rank = Float64(i + 1)
        elif method == 2:
            rank = Float64(j)
        elif method == 3:
            rank = Float64(dense_rank)
        for k in range(i, j):
            ranks[Int(indices[k])] = rank
        dense_rank += 1
        i = j
