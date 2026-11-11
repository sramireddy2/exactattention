// Certified windowed ExactAttention, CPU kernel.
//
// Implements the same algorithm as src/exactattn/certified.py, the bit-accurate model,
// structured the way the GPU kernels will be:
//
//   scores      windowed int64 accumulation of exact bf16 products, then a certified
//               t = RN32(alpha * s)
//   weights     w = RN24(2^t) with an unbounded exponent; a double fast path with a margin
//               check, and a long double second try
//   accumulate  per-tile partials (reference exponent R, Z, N_c, error counts), merged in
//               any order
//   finalize    rounds both ends of the error interval; returns a value only if they agree
//
// Any row whose result is not certified is flagged. The caller recomputes flagged rows with
// the exact reference, so returned bits always equal the spec (docs/SPEC.md).
//
// Build: see csrc/Makefile. Needs __int128, so gcc or clang.

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace {

using i64 = int64_t;
using u64 = uint64_t;
using i128 = __int128;
using u128 = unsigned __int128;

// ---------------------------------------------------------------------------
// Minimal fixed-width unsigned bigint (320 bits), used only for exact final rounding.
// ---------------------------------------------------------------------------
struct U320 {
    u64 w[5] = {0, 0, 0, 0, 0};  // little-endian limbs

    static U320 from_u128(u128 x) {
        U320 r;
        r.w[0] = (u64)x;
        r.w[1] = (u64)(x >> 64);
        return r;
    }
    bool is_zero() const { return !(w[0] | w[1] | w[2] | w[3] | w[4]); }
    int bitlen() const {
        for (int i = 4; i >= 0; --i)
            if (w[i]) return 64 * i + (64 - __builtin_clzll(w[i]));
        return 0;
    }
    U320 shl(int s) const {
        U320 r;
        if (s >= 320) return r;
        int limbs = s / 64, bits = s % 64;
        for (int i = 4; i >= 0; --i) {
            int src = i - limbs;
            if (src < 0) continue;
            u64 v = w[src] << bits;
            if (bits && src - 1 >= 0) v |= w[src - 1] >> (64 - bits);
            r.w[i] = v;
        }
        return r;
    }
    int cmp(const U320& o) const {
        for (int i = 4; i >= 0; --i) {
            if (w[i] != o.w[i]) return w[i] < o.w[i] ? -1 : 1;
        }
        return 0;
    }
    void sub(const U320& o) {  // *this -= o, requires *this >= o
        u64 borrow = 0;
        for (int i = 0; i < 5; ++i) {
            u128 a = (u128)w[i];
            u128 b = (u128)o.w[i] + borrow;
            if (a >= b) {
                w[i] = (u64)(a - b);
                borrow = 0;
            } else {
                w[i] = (u64)(((u128)1 << 64) + a - b);
                borrow = 1;
            }
        }
    }
};

struct Fmt {
    int p;       // precision
    int qmin;    // min quantum exponent
    int emax;    // max exponent of the leading bit
};
constexpr Fmt FP32{24, -149, 127};
constexpr Fmt BF16{8, -133, 127};

// Result of a rounding: value = sign * m * 2^e. m == 0 means zero. overflow set on overflow.
struct Rounded {
    int sign = 1;
    u64 m = 0;
    int e = 0;
    bool overflow = false;
    bool operator==(const Rounded& o) const {
        if (overflow || o.overflow) return false;
        if (m == 0 && o.m == 0) return true;
        return sign == o.sign && m == o.m && e == o.e;
    }
};

// RN-even of (a / b) * 2^E into fmt, sign applied. a, b < 2^128, b > 0.
Rounded round_ratio(u128 a_, u128 b_, int E, int sign, const Fmt& f) {
    Rounded r;
    r.sign = sign;
    if (a_ == 0) return r;
    U320 a = U320::from_u128(a_), b = U320::from_u128(b_);
    int k = a.bitlen() - b.bitlen();
    if (k >= 0) {
        if (a.cmp(b.shl(k)) < 0) --k;
    } else {
        if (a.shl(-k).cmp(b) < 0) --k;
    }
    long K = (long)k + E;                 // leading-bit exponent of the value
    if (K < (long)f.qmin - 1) return r;   // below half the smallest subnormal: rounds to 0
    long qe = K - (f.p - 1);
    if (qe < f.qmin) qe = f.qmin;
    long s = (long)E - qe;                // q = RN(a/b * 2^s)
    U320 n = a, d = b;
    if (s >= 0) n = a.shl((int)s);
    else d = b.shl((int)-s);
    // q < 2^(p+1). Restoring division, bit by bit.
    u64 q = 0;
    for (int i = f.p + 1; i >= 0; --i) {
        U320 ds = d.shl(i);
        if (n.cmp(ds) >= 0) {
            n.sub(ds);
            q |= (u64)1 << i;
        }
    }
    // Remainder n: round to nearest, ties to even.
    U320 twice_r = n.shl(1);
    int c = twice_r.cmp(d);
    if (c > 0 || (c == 0 && (q & 1))) ++q;
    if (q == ((u64)1 << f.p)) {
        q >>= 1;
        ++qe;
    }
    // Overflow: q * 2^qe > (2^p - 1) * 2^(emax - p + 1)
    long top = qe + (64 - __builtin_clzll(q)) - 1;
    if (top > f.emax) r.overflow = true;
    r.m = q;
    r.e = (int)qe;
    return r;
}

Rounded round_dyadic_i128(i128 m, int E, const Fmt& f) {
    int sign = m < 0 ? -1 : 1;
    u128 a = m < 0 ? (u128)(-m) : (u128)m;
    return round_ratio(a, 1, E, sign, f);
}

// ---------------------------------------------------------------------------
// bf16 / fp32 decoding and encoding
// ---------------------------------------------------------------------------
struct Dy {
    int32_t m;  // signed significand
    int32_t e;
};

inline bool decode_bf16(uint16_t b, Dy& out) {
    int exp = (b >> 7) & 0xFF;
    int frac = b & 0x7F;
    int sign = (b & 0x8000) ? -1 : 1;
    if (exp == 0xFF) return false;
    if (exp == 0) {
        out.m = sign * frac;
        out.e = -133;
    } else {
        out.m = sign * (frac | 0x80);
        out.e = exp - 134;
    }
    return true;
}

inline bool decode_fp32(float x, int32_t& m, int32_t& e) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    int exp = (b >> 23) & 0xFF;
    int32_t frac = b & 0x7FFFFF;
    int sign = (b & 0x80000000u) ? -1 : 1;
    if (exp == 0xFF) return false;
    if (exp == 0) {
        m = sign * frac;
        e = -149;
    } else {
        m = sign * (frac | 0x800000);
        e = exp - 150;
    }
    return true;
}

inline float rounded_to_float(const Rounded& r) {
    if (r.m == 0) return 0.0f;
    double v = std::ldexp((double)r.m, r.e);  // exact
    return (float)(r.sign * v);              // exact: the value is representable
}

inline int ilog2_u64(u64 x) { return 63 - __builtin_clzll(x); }

// ---------------------------------------------------------------------------
// Scores and logits
// ---------------------------------------------------------------------------
constexpr int SCORE_BITS = 64;

// Returns false if the logit is not certified; the row then falls back.
bool certified_logit(const Dy* q, const Dy* k, int d, int fs, int32_t am, int32_t ae, float& t_out) {
    int Lmax = INT_MIN;
    bool any = false;
    for (int i = 0; i < d; ++i) {
        int32_t p = q[i].m * k[i].m;
        if (!p) continue;
        any = true;
        int lead = q[i].e + k[i].e + ilog2_u64((u64)(p < 0 ? -p : p));
        Lmax = std::max(Lmax, lead);
    }
    if (!any || am == 0) {
        t_out = 0.0f;
        return true;
    }
    int base = Lmax + 1 - fs;
    i64 acc = 0;
    u64 err = 0;
    for (int i = 0; i < d; ++i) {
        i64 p = (i64)q[i].m * k[i].m;
        if (!p) continue;
        int sh = q[i].e + k[i].e - base;
        if (sh >= 0) {
            acc += p << sh;
        } else {
            int s = -sh;
            if (s >= 63) {
                acc += p < 0 ? -1 : 0;
                err += 1;
            } else {
                i64 v = p >> s;
                acc += v;
                if ((v << s) != p) err += 1;
            }
        }
    }
    Rounded lo = round_dyadic_i128((i128)am * acc, ae + base, FP32);
    Rounded hi = err ? round_dyadic_i128((i128)am * (acc + (i64)err), ae + base, FP32) : lo;
    if (lo.overflow || hi.overflow || !(lo == hi)) return false;
    t_out = rounded_to_float(lo);
    return true;
}

// ---------------------------------------------------------------------------
// Weights: w = RN24(2^t), unbounded exponent. M in [2^23, 2^24), value M * 2^E.
// ---------------------------------------------------------------------------
bool weight(float t, int32_t& M, i64& E) {
    double td = t;
    if (!(std::fabs(td) < 1073741824.0)) return false;  // |t| >= 2^30: leave to fallback
    double n = std::floor(td);
    double f = td - n;  // exact
    if (f == 0.0) {
        M = 1 << 23;
        E = (i64)n - 23;
        return true;
    }
    double scaled = std::exp2(f) * 8388608.0;  // 2^23
    double fl = std::floor(scaled);
    double frac = scaled - fl;
    if (std::fabs(frac - 0.5) < 0x1p-18) {
        // Second try in long double (x86: 64-bit significand).
        long double sl = std::exp2l((long double)f) * 8388608.0L;
        long double fll = std::floor(sl);
        long double fr = sl - fll;
        if (std::fabs((double)(fr - 0.5L)) < 0x1p-30) return false;  // unresolved
        fl = (double)fll;
        frac = (double)fr;
    }
    i64 q = (i64)fl + (frac >= 0.5 ? 1 : 0);
    i64 e = (i64)n - 23;
    if (q == (1 << 24)) {
        q >>= 1;
        e += 1;
    }
    M = (int32_t)q;
    E = e;
    return true;
}

// ---------------------------------------------------------------------------
// Partials
// ---------------------------------------------------------------------------
template <typename Acc>
struct Traits;
template <>
struct Traits<i64> {
    static constexpr int bits = 64;
};
template <>
struct Traits<i128> {
    static constexpr int bits = 128;
};

template <typename Acc>
inline Acc floor_shift(Acc m, i64 sh, bool& lost) {
    constexpr int B = Traits<Acc>::bits;
    if (sh >= 0) {
        lost = false;
        return m << sh;  // headroom is guaranteed by construction
    }
    i64 s = -sh;
    if (s >= B - 1) {
        lost = m != 0;
        return m < 0 ? (Acc)-1 : (Acc)0;
    }
    Acc v = m >> s;
    lost = (Acc)(v << s) != m;
    return v;
}

// Re-express [acc, acc + err] in units 2^s times larger (s >= 0), rigorously:
// acc' = floor(acc / 2^s), err' = ceil(err / 2^s) + (1 if bits were lost).
template <typename Acc>
inline void rescale(Acc& acc, u64& err, i64 s) {
    if (s <= 0) return;
    bool lost;
    acc = floor_shift<Acc>(acc, -s, lost);
    u64 e = s >= 64 ? (err ? 1 : 0) : (err >> s) + ((err & (((u64)1 << s) - 1)) != 0);
    err = e + (lost ? 1 : 0);
}

template <typename Acc>
inline bool fits(Acc v, u64 err) {
    constexpr int B = Traits<Acc>::bits;
    using U = std::conditional_t<B == 64, u64, u128>;
    U a = v < 0 ? (U)(-(v + 1)) + 1 : (U)v;
    U lim = (U)1 << (B - 1);
    return a < lim && (U)err < lim - a;
}

template <typename Acc>
struct Partial {
    bool empty = true;
    i64 R = 0;
    Acc Z = 0;
    u64 Zerr = 0;
    u64 count = 0;
    std::vector<i64> NR;  // INT64_MIN = channel is exactly zero
    std::vector<Acc> N;
    std::vector<u64> Nerr;
    explicit Partial(int dv) : NR(dv, INT64_MIN), N(dv, 0), Nerr(dv, 0) {}
};

struct Ctx {
    int d, dv, fz, fn, key_log2;
};

template <typename Acc>
bool tile_partial(const Ctx& c, const int32_t* M, const i64* Ew, const Dy* const* vrows, int n,
                  Partial<Acc>& p) {
    if (n == 0) return true;
    p.empty = false;
    p.count = n;
    i64 R = INT64_MIN;
    for (int j = 0; j < n; ++j) R = std::max(R, Ew[j] + 23);
    p.R = R;
    for (int j = 0; j < n; ++j) {
        bool lost;
        p.Z += floor_shift<Acc>((Acc)M[j], Ew[j] - (R - c.fz), lost);
        p.Zerr += lost;
    }
    if (!fits<Acc>(p.Z, p.Zerr)) return false;
    for (int ch = 0; ch < c.dv; ++ch) {
        i64 NR = INT64_MIN;
        for (int j = 0; j < n; ++j) {
            int32_t m = vrows[j][ch].m;
            if (!m) continue;
            i64 L = Ew[j] + 23 + 1 + vrows[j][ch].e + ilog2_u64((u64)(m < 0 ? -m : m));
            NR = std::max(NR, L);
        }
        if (NR == INT64_MIN) continue;
        p.NR[ch] = NR;
        i64 base = NR - c.fn;
        Acc acc = 0;
        u64 err = 0;
        for (int j = 0; j < n; ++j) {
            int32_t m = vrows[j][ch].m;
            if (!m) continue;
            bool lost;
            acc += floor_shift<Acc>((Acc)((i64)M[j] * m), Ew[j] + vrows[j][ch].e - base, lost);
            err += lost;
        }
        if (!fits<Acc>(acc, err)) return false;
        p.N[ch] = acc;
        p.Nerr[ch] = err;
    }
    return true;
}

template <typename Acc>
bool merge_into(const Ctx& c, Partial<Acc>& a, const Partial<Acc>& b) {
    if (b.empty) return true;
    if (a.empty) {
        a = b;
        return true;
    }
    a.count += b.count;
    if (a.count > ((u64)1 << c.key_log2)) return false;
    i64 R = std::max(a.R, b.R);
    Acc za = a.Z, zb = b.Z;
    u64 ea = a.Zerr, eb = b.Zerr;
    rescale<Acc>(za, ea, R - a.R);
    rescale<Acc>(zb, eb, R - b.R);
    a.R = R;
    a.Z = za + zb;
    a.Zerr = ea + eb;
    if (!fits<Acc>(a.Z, a.Zerr)) return false;
    for (int ch = 0; ch < c.dv; ++ch) {
        i64 ra = a.NR[ch], rb = b.NR[ch];
        if (rb == INT64_MIN) continue;
        if (ra == INT64_MIN) {
            a.NR[ch] = rb;
            a.N[ch] = b.N[ch];
            a.Nerr[ch] = b.Nerr[ch];
            continue;
        }
        i64 ref = std::max(ra, rb);
        Acc xa = a.N[ch], xb = b.N[ch];
        u64 eA = a.Nerr[ch], eB = b.Nerr[ch];
        rescale<Acc>(xa, eA, ref - ra);
        rescale<Acc>(xb, eB, ref - rb);
        a.NR[ch] = ref;
        a.N[ch] = xa + xb;
        a.Nerr[ch] = eA + eB;
        if (!fits<Acc>(a.N[ch], a.Nerr[ch])) return false;
    }
    return true;
}

template <typename Acc>
inline u128 mag(Acc v) {
    return v < 0 ? (u128)(-(i128)v) : (u128)(i128)v;
}

// Certified rounding of N_c / Z for every channel. Returns false if any channel is not
// certified.
template <typename Acc>
bool finalize(const Ctx& c, const Partial<Acc>& p, bool fp32_out, void* out_row) {
    uint16_t* o16 = (uint16_t*)out_row;
    float* o32 = (float*)out_row;
    for (int ch = 0; ch < c.dv; ++ch) {
        if (fp32_out) o32[ch] = 0.0f;
        else o16[ch] = 0;
    }
    if (p.empty) return true;
    const Fmt& f = fp32_out ? FP32 : BF16;
    i64 zunit = p.R - c.fz;
    i128 Zlo = (i128)p.Z, Zhi = (i128)p.Z + (i128)p.Zerr;
    if (Zlo <= 0) return false;
    for (int ch = 0; ch < c.dv; ++ch) {
        if (p.NR[ch] == INT64_MIN) continue;
        i64 nunit = p.NR[ch] - c.fn;
        i128 Nlo = (i128)p.N[ch], Nhi = (i128)p.N[ch] + (i128)p.Nerr[ch];
        i128 ln, ld, hn, hd;
        if (Nlo >= 0) {
            ln = Nlo; ld = Zhi; hn = Nhi; hd = Zlo;
        } else if (Nhi <= 0) {
            ln = Nlo; ld = Zlo; hn = Nhi; hd = Zhi;
        } else {
            ln = Nlo; ld = Zlo; hn = Nhi; hd = Zlo;
        }
        i64 sh = nunit - zunit;
        if (sh > 100000 || sh < -100000) return false;
        Rounded rl = round_ratio(ln < 0 ? (u128)(-ln) : (u128)ln, (u128)ld, (int)sh, ln < 0 ? -1 : 1, f);
        Rounded rh = round_ratio(hn < 0 ? (u128)(-hn) : (u128)hn, (u128)hd, (int)sh, hn < 0 ? -1 : 1, f);
        if (!(rl == rh)) return false;
        float v = rounded_to_float(rl);
        if (fp32_out) {
            o32[ch] = v;
        } else {
            uint32_t b;
            std::memcpy(&b, &v, 4);
            o16[ch] = (uint16_t)(b >> 16);  // exact: the value is bf16-representable
        }
    }
    return true;
}

struct Args {
    const uint16_t* q;
    int64_t rows;
    int d;
    const uint16_t* k;
    const uint16_t* v;
    int64_t nkeys;
    int dv;
    float alpha;
    const int64_t* row_ptr;
    const int64_t* key_idx;
    int fp32_out;
    int tile;
    int splits;
    uint64_t order_seed;
    void* out;
    uint8_t* flags;
};

template <typename Acc>
void run_rows(const Args& A, const Ctx& c, int64_t r0, int64_t r1, const std::vector<Dy>& Kd,
              const std::vector<Dy>& Vd, int fs, int32_t am, int32_t ae, int64_t* stats) {
    std::vector<Dy> qd(A.d);
    std::vector<float> t;
    std::vector<int32_t> M;
    std::vector<i64> E;
    std::vector<const Dy*> vrows;
    const size_t osz = A.fp32_out ? 4 : 2;
    for (int64_t r = r0; r < r1; ++r) {
        uint8_t flag = 0;
        void* orow = (char*)A.out + (size_t)r * A.dv * osz;
        for (int i = 0; i < A.d; ++i)
            if (!decode_bf16(A.q[r * A.d + i], qd[i])) flag = 2;
        int64_t b = A.row_ptr[r], e = A.row_ptr[r + 1];
        int64_t n = e - b;
        t.resize(n);
        M.resize(n);
        E.resize(n);
        for (int64_t j = 0; j < n && !flag; ++j) {
            int64_t key = A.key_idx[b + j];
            if (!certified_logit(qd.data(), &Kd[key * A.d], A.d, fs, am, ae, t[j])) {
                flag = 1;
                stats[0]++;
                break;
            }
            if (!weight(t[j], M[j], E[j])) {
                flag = 1;
                stats[1]++;
                break;
            }
        }
        if (!flag) {
            // Split the key list into `splits` contiguous chunks, process each in tiles,
            // then merge the chunk partials in a seeded random order.
            int S = std::max<int64_t>(1, std::min<int64_t>(A.splits, std::max<int64_t>(n, 1)));
            std::vector<Partial<Acc>> parts(S, Partial<Acc>(A.dv));
            for (int s = 0; s < S && !flag; ++s) {
                int64_t cb = n * s / S, ce = n * (s + 1) / S;
                for (int64_t t0 = cb; t0 < ce && !flag; t0 += A.tile) {
                    int64_t t1 = std::min<int64_t>(t0 + A.tile, ce);
                    vrows.resize(t1 - t0);
                    for (int64_t j = t0; j < t1; ++j) vrows[j - t0] = &Vd[A.key_idx[b + j] * A.dv];
                    Partial<Acc> tp(A.dv);
                    if (!tile_partial<Acc>(c, &M[t0], &E[t0], vrows.data(), (int)(t1 - t0), tp) ||
                        !merge_into<Acc>(c, parts[s], tp))
                        flag = 1;
                }
            }
            std::vector<int> order(S);
            std::iota(order.begin(), order.end(), 0);
            std::mt19937_64 rng(A.order_seed ^ (uint64_t)r * 0x9E3779B97F4A7C15ull);
            std::shuffle(order.begin(), order.end(), rng);
            Partial<Acc> total(A.dv);
            for (int s : order)
                if (!flag && !merge_into<Acc>(c, total, parts[s])) flag = 1;
            if (!flag && !finalize<Acc>(c, total, A.fp32_out, orow)) {
                flag = 1;
                stats[2]++;
            }
        }
        A.flags[r] = flag;
    }
}

}  // namespace

extern "C" {

// Returns 0 on success, or a negative code on invalid arguments. Per-row flags:
//   0 = certified (bits equal the spec)
//   1 = not certified (recompute with the reference)
//   2 = non-finite input (the reference raises)
// stats[0..2] += score, weight and final-rounding fallback counts.
int ea_attention_cpu(const uint16_t* q, int64_t rows, int d, const uint16_t* k, const uint16_t* v,
                     int64_t nkeys, int dv, float alpha, const int64_t* row_ptr, const int64_t* key_idx,
                     int fp32_out, int acc_bits, int tile, int splits, uint64_t order_seed, int nthreads,
                     void* out, uint8_t* flags, int64_t* stats) {
    if (d <= 0 || d > 1024 || dv <= 0 || tile <= 0 || splits <= 0) return -1;
    if (acc_bits != 64 && acc_bits != 128) return -2;
    int32_t am, ae;
    if (!decode_fp32(alpha, am, ae)) return -3;
    // Decode K and V once. These are shared by every row (in the GPU design, by every
    // query in the tile).
    std::vector<Dy> Kd((size_t)nkeys * d), Vd((size_t)nkeys * dv);
    std::vector<uint8_t> kbad(nkeys, 0);
    for (int64_t i = 0; i < nkeys * d; ++i)
        if (!decode_bf16(k[i], Kd[i])) kbad[i / d] = 1;
    for (int64_t i = 0; i < nkeys * dv; ++i)
        if (!decode_bf16(v[i], Vd[i])) kbad[i / dv] = 1;
    int dlog = 0;
    while ((1 << dlog) < d) ++dlog;
    int fs = SCORE_BITS - 1 - dlog;
    int64_t maxn = 1;
    for (int64_t r = 0; r < rows; ++r) maxn = std::max<int64_t>(maxn, row_ptr[r + 1] - row_ptr[r]);
    int key_log2 = 0;
    while (((int64_t)1 << key_log2) < maxn) ++key_log2;
    Ctx c{d, dv, acc_bits - 2 - key_log2, acc_bits - 2 - key_log2, key_log2};
    Args A{q, rows, d, k, v, nkeys, dv, alpha, row_ptr, key_idx, fp32_out, tile, splits, order_seed, out, flags};
    int T = std::max(1, nthreads);
    std::vector<std::thread> th;
    std::vector<std::vector<int64_t>> st(T, std::vector<int64_t>(3, 0));
    for (int i = 0; i < T; ++i) {
        int64_t r0 = rows * i / T, r1 = rows * (i + 1) / T;
        th.emplace_back([&, i, r0, r1] {
            if (acc_bits == 64) run_rows<i64>(A, c, r0, r1, Kd, Vd, fs, am, ae, st[i].data());
            else run_rows<i128>(A, c, r0, r1, Kd, Vd, fs, am, ae, st[i].data());
        });
    }
    for (auto& x : th) x.join();
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = row_ptr[r]; j < row_ptr[r + 1]; ++j)
            if (kbad[key_idx[j]]) flags[r] = 2;
    }
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < 3; ++j) stats[j] += st[i][j];
    return 0;
}

// Exposed for unit tests: RN-even of (a / b) * 2^E into fp32 (p = 24) or bf16 (p = 8).
// a and b are given as (hi, lo) u64 halves. Returns the float, sets *overflow.
float ea_round_ratio(uint64_t a_hi, uint64_t a_lo, uint64_t b_hi, uint64_t b_lo, int E, int neg, int fp32,
                     int* overflow) {
    u128 a = ((u128)a_hi << 64) | a_lo, b = ((u128)b_hi << 64) | b_lo;
    Rounded r = round_ratio(a, b, E, neg ? -1 : 1, fp32 ? FP32 : BF16);
    *overflow = r.overflow;
    return rounded_to_float(r);
}

}  // extern "C"
