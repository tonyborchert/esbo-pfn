// Small dense linear algebra for the Furuta LQR design: a scaling-and-squaring Pade matrix
// exponential (scipy.linalg.expm) and a port of scipy.linalg.solve_discrete_are (balanced extended
// pencil, QR deflation, dgges, dtgsen reordering, stable subspace) that calls the same LAPACK
// routines scipy calls, so the same plants are undesignable as in scipy. Every scipy failure mode
// is a plain `false` return, which the simulator turns into a crash at the first step.
// Matrices are column-major so they go to LAPACK unchanged.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace esbo {
namespace la {


// --------------------------------------------------------------------------- LAPACK (LP64)
// Character arguments carry the hidden Fortran length arguments at the end, which gfortran-built
// LAPACK (OpenBLAS) expects and MKL and Accelerate ignore.
extern "C" {
void dgebal_(const char* job, const int* n, double* a, const int* lda, int* ilo, int* ihi,
             double* scale, int* info, std::size_t job_len);
void dgeqrf_(const int* m, const int* n, double* a, const int* lda, double* tau, double* work,
             const int* lwork, int* info);
void dorgqr_(const int* m, const int* n, const int* k, double* a, const int* lda, const double* tau,
             double* work, const int* lwork, int* info);
void dgges_(const char* jobvsl, const char* jobvsr, const char* sort,
            int (*selctg)(const double*, const double*, const double*), const int* n, double* a,
            const int* lda, double* b, const int* ldb, int* sdim, double* alphar, double* alphai,
            double* beta, double* vsl, const int* ldvsl, double* vsr, const int* ldvsr, double* work,
            const int* lwork, int* bwork, int* info, std::size_t l1, std::size_t l2,
            std::size_t l3);
void dtgsen_(const int* ijob, const int* wantq, const int* wantz, const int* select, const int* n,
             double* a, const int* lda, double* b, const int* ldb, double* alphar, double* alphai,
             double* beta, double* q, const int* ldq, double* z, const int* ldz, int* m, double* pl,
             double* pr, double* dif, double* work, const int* lwork, int* iwork, const int* liwork,
             int* info);
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b,
            const int* ldb, int* info);
void dgesdd_(const char* jobz, const int* m, const int* n, double* a, const int* lda, double* s,
             double* u, const int* ldu, double* vt, const int* ldvt, double* work, const int* lwork,
             int* iwork, int* info, std::size_t jobz_len);
}

// --------------------------------------------------------------------------- tiny matrix type
struct Mat {
    int r = 0, c = 0;
    std::vector<double> d;   // column-major: d[i + j*r]

    Mat() = default;
    Mat(int rows, int cols) : r(rows), c(cols), d(std::size_t(rows) * cols, 0.0) {}

    double& operator()(int i, int j) { return d[std::size_t(i) + std::size_t(j) * r]; }
    double operator()(int i, int j) const { return d[std::size_t(i) + std::size_t(j) * r]; }
    double* data() { return d.data(); }
    const double* data() const { return d.data(); }
};

inline Mat eye(int n) {
    Mat m(n, n);
    for (int i = 0; i < n; ++i) m(i, i) = 1.0;
    return m;
}

// C = op(A) * op(B), where `at`/`bt` transpose the respective operand.
inline Mat mul(const Mat& A, const Mat& B, bool at = false, bool bt = false) {
    const int ar = at ? A.c : A.r, ak = at ? A.r : A.c;
    const int bk = bt ? B.c : B.r, bc = bt ? B.r : B.c;
    (void)bk;   // ak == bk by construction at every call site
    Mat C(ar, bc);
    for (int j = 0; j < bc; ++j)
        for (int k = 0; k < ak; ++k) {
            const double b = bt ? B(j, k) : B(k, j);
            if (b == 0.0) continue;
            for (int i = 0; i < ar; ++i) C(i, j) += (at ? A(k, i) : A(i, k)) * b;
        }
    return C;
}

inline Mat transpose(const Mat& A) {
    Mat T(A.c, A.r);
    for (int j = 0; j < A.c; ++j)
        for (int i = 0; i < A.r; ++i) T(j, i) = A(i, j);
    return T;
}

// numpy/scipy `norm(M, 1)` for a matrix: the maximum absolute COLUMN sum.
inline double norm1(const Mat& A) {
    double best = 0.0;
    for (int j = 0; j < A.c; ++j) {
        double s = 0.0;
        for (int i = 0; i < A.r; ++i) s += std::fabs(A(i, j));
        best = std::max(best, s);
    }
    return best;
}

inline bool all_finite(const Mat& A) {
    for (double v : A.d)
        if (!std::isfinite(v)) return false;
    return true;
}

// Solve A X = B in place of B (LAPACK dgesv). Returns false on a singular/failed factorization.
inline bool solve(Mat A, Mat& B) {
    const int n = A.r, nrhs = B.c;
    std::vector<int> ipiv(n);
    int info = 0;
    dgesv_(&n, &nrhs, A.data(), &n, ipiv.data(), B.data(), &n, &info);
    return info == 0;
}

// --------------------------------------------------------------------------- expm
// Higham's scaling-and-squaring Pade algorithm -- the same family scipy.linalg.expm uses, so the
// discretised (Ad, Bd) agree with scipy's `cont2discrete(..., method="zoh")` to ~1e-15 relative.
inline Mat _pade(const Mat& A, const double* b, int m) {
    const int n = A.r;
    Mat U(n, n), V(n, n);
    if (m == 13) {
        const Mat A2 = mul(A, A), A4 = mul(A2, A2), A6 = mul(A2, A4);
        Mat W1(n, n), Z1(n, n);
        for (int i = 0; i < n * n; ++i) {
            W1.d[i] = b[13] * A6.d[i] + b[11] * A4.d[i] + b[9] * A2.d[i];
            Z1.d[i] = b[12] * A6.d[i] + b[10] * A4.d[i] + b[8] * A2.d[i];
        }
        Mat W = mul(A6, W1), Z = mul(A6, Z1);
        for (int i = 0; i < n * n; ++i) {
            W.d[i] += b[7] * A6.d[i] + b[5] * A4.d[i] + b[3] * A2.d[i];
            Z.d[i] += b[6] * A6.d[i] + b[4] * A4.d[i] + b[2] * A2.d[i];
        }
        for (int i = 0; i < n; ++i) { W(i, i) += b[1]; Z(i, i) += b[0]; }
        U = mul(A, W);
        V = Z;
    } else {
        // U = A * sum_{odd k} b_k A^(k-1),  V = sum_{even k} b_k A^k
        Mat P = eye(n);                       // A^0, A^2, A^4, ...
        const Mat A2 = mul(A, A);
        Mat Uinner(n, n);
        for (int k = 0; k <= m; k += 2) {
            for (int i = 0; i < n * n; ++i) {
                Uinner.d[i] += b[k + 1] * P.d[i];
                V.d[i] += b[k] * P.d[i];
            }
            if (k + 2 <= m) P = mul(P, A2);
        }
        U = mul(A, Uinner);
    }
    Mat num(n, n), den(n, n);
    for (int i = 0; i < n * n; ++i) { num.d[i] = V.d[i] + U.d[i]; den.d[i] = V.d[i] - U.d[i]; }
    if (!solve(den, num)) return Mat();       // empty -> caller treats as failure
    return num;
}

inline Mat expm(const Mat& A) {
    static const double b3[]  = {120., 60., 12., 1.};
    static const double b5[]  = {30240., 15120., 3360., 420., 30., 1.};
    static const double b7[]  = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};
    static const double b9[]  = {17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                                 2162160., 110880., 3960., 90., 1.};
    static const double b13[] = {64764752532480000., 32382376266240000., 7771770303897600.,
                                 1187353796428800., 129060195264000., 10559470521600., 670442572800.,
                                 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.};
    const double nrm = norm1(A);
    if (!std::isfinite(nrm)) return Mat();
    if (nrm <= 1.495585217958292e-2) return _pade(A, b3, 3);
    if (nrm <= 2.539398330063230e-1) return _pade(A, b5, 5);
    if (nrm <= 9.504178996162932e-1) return _pade(A, b7, 7);
    if (nrm <= 2.097847961257068e0)  return _pade(A, b9, 9);

    const int s = std::max(0, (int)std::ceil(std::log2(nrm / 5.371920351148152e0)));
    Mat As = A;
    const double f = std::ldexp(1.0, -s);
    for (double& v : As.d) v *= f;
    Mat X = _pade(As, b13, 13);
    for (int i = 0; i < s && X.r; ++i) X = mul(X, X);
    return X;
}

// --------------------------------------------------------------------------- DARE
// scipy's `np.spacing(x)` = nextafter(x, inf) - x.
inline double spacing(double x) { return std::nextafter(x, INFINITY) - x; }

// Port of scipy `_solve_discrete_are(a, b, q, r, e=None, s=None, balanced=True)`.
// A: (m,m), B: (m,n), Q: (m,m), R: (n,n). Writes the stabilizing solution into X (m,m).
// Returns false wherever scipy raises (LinAlgError or the ordqz ValueError) or LAPACK fails.
inline bool solve_discrete_are(const Mat& A, const Mat& B, const Mat& Q, const Mat& R, Mat& X) {
    const int m = A.r, n = B.c, N = 2 * m + n;
    if (!all_finite(A) || !all_finite(B) || !all_finite(Q) || !all_finite(R)) return false;

    // ---- the (H, J) matrix pencil -------------------------------------------------------------
    Mat H(N, N), J(N, N);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i) {
            H(i, j) = A(i, j);
            H(m + i, j) = -Q(i, j);
            J(m + i, m + j) = A(j, i);          // a.conj().T
        }
    for (int i = 0; i < m; ++i) { H(m + i, m + i) = 1.0; J(i, i) = 1.0; }
    // The extended DARE pencil (scipy's layout): H[2m:, m:2m] is ZERO -- only J carries -b^H there.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            H(i, 2 * m + j) = B(i, j);
            J(2 * m + j, m + i) = -B(i, j);     // -b.conj().T
        }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) H(2 * m + i, 2 * m + j) = R(i, j);

    // ---- symplectic balancing (Benner) --------------------------------------------------------
    std::vector<double> sca(N, 1.0);
    {
        Mat M(N, N);
        for (int i = 0; i < N * N; ++i) M.d[i] = std::fabs(H.d[i]) + std::fabs(J.d[i]);
        for (int i = 0; i < N; ++i) M(i, i) = 0.0;
        int ilo = 0, ihi = 0, info = 0;
        dgebal_("S", &N, M.data(), &N, &ilo, &ihi, sca.data(), &info, 1);
        if (info != 0) return false;
        // np.allclose(sca, 1) -- gebal only emits exact powers of two, so this is "no scaling".
        bool unit = true;
        for (double v : sca)
            if (std::fabs(v - 1.0) > 1e-8 + 1e-5) { unit = false; break; }
        if (!unit) {
            std::vector<double> lg(N);
            for (int i = 0; i < N; ++i) lg[i] = std::log2(sca[i]);
            for (int i = 0; i < m; ++i) {
                // np.round == round-half-to-EVEN (banker's rounding), which std::nearbyint gives
                // under the default FE_TONEAREST rounding mode, std::round does not.
                const double s = std::nearbyint((lg[m + i] - lg[i]) / 2.0);
                sca[i] = std::exp2(s);
                sca[m + i] = std::exp2(-s);
            }
            for (int i = 2 * m; i < N; ++i) sca[i] = std::exp2(lg[i]);
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i) {
                    const double w = sca[i] / sca[j];
                    H(i, j) *= w;
                    J(i, j) *= w;
                }
        }
    }

    // ---- deflate the pencil by the R column ---------------------------------------------------
    Mat Qfull(N, N);                            // full Q of qr(H[:, -n:])
    {
        std::vector<double> tau(n), work(4 * N + 64);
        int lwork = (int)work.size(), info = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < N; ++i) Qfull(i, j) = H(i, 2 * m + j);
        dgeqrf_(&N, &n, Qfull.data(), &N, tau.data(), work.data(), &lwork, &info);
        if (info != 0) return false;
        dorgqr_(&N, &N, &n, Qfull.data(), &N, tau.data(), work.data(), &lwork, &info);
        if (info != 0) return false;
    }
    Mat q2(N, 2 * m);                           // q_of_qr[:, n:]
    for (int j = 0; j < 2 * m; ++j)
        for (int i = 0; i < N; ++i) q2(i, j) = Qfull(i, n + j);
    Mat Hs(N, 2 * m), Js(N, 2 * m);
    for (int j = 0; j < 2 * m; ++j)
        for (int i = 0; i < N; ++i) { Hs(i, j) = H(i, j); Js(i, j) = J(i, j); }
    Mat HH = mul(q2, Hs, true, false);          // (2m, 2m)
    Mat JJ = mul(q2, Js, true, false);
    if (!all_finite(HH) || !all_finite(JJ)) return false;

    // ---- ordqz(HH, JJ, sort='iuc') ------------------------------------------------------------
    const int nn = 2 * m;
    std::vector<double> alphar(nn), alphai(nn), beta(nn), work(8 * nn + 64);
    std::vector<int> bwork(nn);
    Mat VSL(nn, nn), VSR(nn, nn);
    int sdim = 0, lwork = (int)work.size(), info = 0;
    dgges_("V", "V", "N", nullptr, &nn, HH.data(), &nn, JJ.data(), &nn, &sdim, alphar.data(),
           alphai.data(), beta.data(), VSL.data(), &nn, VSR.data(), &nn, work.data(), &lwork,
           bwork.data(), &info, 1, 1, 1);
    if (info != 0) return false;

    std::vector<int> select(nn, 0);             // scipy `_iuc`: |alpha/beta| < 1, beta != 0
    for (int i = 0; i < nn; ++i)
        select[i] = (beta[i] != 0.0 &&
                     std::abs(std::complex<double>(alphar[i], alphai[i]) / beta[i]) < 1.0) ? 1 : 0;

    {
        const int ijob = 0, wantq = 1, wantz = 1, liwork = 1;
        int mout = 0, tinfo = 0;
        double pl = 0.0, pr = 0.0, dif[2] = {0.0, 0.0};
        std::vector<double> twork(4 * nn + 16);
        std::vector<int> tiwork(1);
        const int tlwork = (int)twork.size();
        dtgsen_(&ijob, &wantq, &wantz, select.data(), &nn, HH.data(), &nn, JJ.data(), &nn,
                alphar.data(), alphai.data(), beta.data(), VSL.data(), &nn, VSR.data(), &nn, &mout,
                &pl, &pr, dif, twork.data(), &tlwork, tiwork.data(), &liwork, &tinfo);
        // tinfo == 1 is exactly the "Reordering of (A, B) failed ... very ill-conditioned"
        // ValueError of scipy. Here it means "undesignable".
        if (tinfo != 0) return false;
    }
    const Mat& u = VSR;

    // ---- stable subspace -> X = u10 @ inv(u00) ------------------------------------------------
    Mat u00(m, m), u10(m, m);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i) { u00(i, j) = u(i, j); u10(i, j) = u(m + i, j); }

    {   // scipy: lu(u00) then `if 1/cond(uu) < np.spacing(1.): raise`
        Mat lu = u00;
        std::vector<int> ipiv(m);
        int linfo = 0;
        dgetrf_(&m, &m, lu.data(), &m, ipiv.data(), &linfo);
        if (linfo < 0) return false;
        Mat U(m, m);                            // the upper-triangular factor
        for (int j = 0; j < m; ++j)
            for (int i = 0; i <= j; ++i) U(i, j) = lu(i, j);
        std::vector<double> s(m), svwork(8 * m * m + 256), dummy(1, 0.0);
        std::vector<int> siwork(8 * m);
        int slwork = (int)svwork.size(), sinfo = 0;
        const int ld1 = 1;
        // JOBZ='N' -> U/VT are not referenced, but hand LAPACK real storage rather than nullptr.
        dgesdd_("N", &m, &m, U.data(), &m, s.data(), dummy.data(), &ld1, dummy.data(), &ld1,
                svwork.data(), &slwork, siwork.data(), &sinfo, 1);
        if (sinfo != 0) return false;
        const double rcond = (s[0] == 0.0) ? 0.0 : s[m - 1] / s[0];   // 1 / cond_2(U)
        if (!(rcond >= spacing(1.0))) return false;
    }

    // X u00 = u10  <=>  u00^T X^T = u10^T
    Mat lhs = transpose(u00), rhs = transpose(u10);
    if (!solve(lhs, rhs)) return false;
    X = transpose(rhs);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i) X(i, j) *= sca[i] * sca[j];

    // ---- symmetry check (scipy: "eigenvalues too close to the unit circle") -------------------
    Mat usym = mul(u00, u10, true, false);
    const double nsym = norm1(usym);
    Mat skew(m, m);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i) skew(i, j) = usym(i, j) - usym(j, i);
    if (norm1(skew) > std::max(spacing(1000.0), 0.1 * nsym)) return false;

    for (int j = 0; j < m; ++j)
        for (int i = 0; i < j; ++i) {
            const double v = 0.5 * (X(i, j) + X(j, i));
            X(i, j) = X(j, i) = v;
        }
    return all_finite(X);
}

}  // namespace la
}  // namespace esbo
