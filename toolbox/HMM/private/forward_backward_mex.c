/*
 * Forward-backward recursion for the variational posterior over states
 * FORMAT [gamma,Xi,logZ] = forward_backward_mex(lpi,lA,lb,xirange)
 *
 * C implementation of the recursion in spm_hmm_forward_backward.m:
 * B = exp(lb - max(lb,[],2)), A = exp(lA), a scaled forward pass, a
 * backward pass, and Xi summed over time.
 *
 * lpi    - [K x 1] log initial-state potentials
 * lA     - [K x K] log transition potentials (not renormalised)
 * lb     - [T x K] log emission potentials
 * xirange - [first last]: Xi sums the transitions into samples first to
 *          last (1-based; empty if last < first)                [[2 T]]
 *
 * gamma  - [T x K] marginal posteriors
 * Xi     - [K x K] summed pairwise posteriors
 * logZ   - log normaliser
 *
 * Chetan Gohil
 * Copyright (C) 2026 NDCN, University of Oxford
 */

#include <math.h>
#include "mex.h"

static void check_input(const mxArray *a, const char *name)
{
    if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a))
        mexErrMsgIdAndTxt("spm_hmm_forward_backward:invalidInput",
            "%s must be a real, full double array.", name);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const double *lpi, *lA, *lb;
    double *A, *B, *alpha, *beta, *c, *tmp, *gamma, *Xi;
    double sum_mx = 0.0, logZ = 0.0, s, mx;
    mwSize T, K, t, i, j, nbad = 0, x0, x1;

    if (nrhs < 3 || nrhs > 4) mexErrMsgIdAndTxt("spm_hmm_forward_backward:invalidInput",
        "Three or four inputs are required.");
    if (nlhs > 3) mexErrMsgIdAndTxt("spm_hmm_forward_backward:invalidInput",
        "At most three outputs are returned.");
    check_input(prhs[0], "lpi");
    check_input(prhs[1], "lA");
    check_input(prhs[2], "lb");
    T = mxGetM(prhs[2]);
    K = mxGetN(prhs[2]);
    if (mxGetNumberOfElements(prhs[0]) != K || mxGetM(prhs[1]) != K || mxGetN(prhs[1]) != K)
        mexErrMsgIdAndTxt("spm_hmm_forward_backward:invalidInput",
            "lpi must have K elements and lA must be K x K, with K = size(lb,2).");
    lpi = mxGetPr(prhs[0]);
    lA  = mxGetPr(prhs[1]);
    lb  = mxGetPr(prhs[2]);
    x0 = 1; x1 = T > 0 ? T-1 : 0;          /* 0-based targets of transitions */
    if (nrhs == 4) {
        const double *r;
        check_input(prhs[3], "xirange");
        if (mxGetNumberOfElements(prhs[3]) != 2)
            mexErrMsgIdAndTxt("spm_hmm_forward_backward:invalidInput",
                "xirange must have two elements.");
        r = mxGetPr(prhs[3]);
        if (!(r[0] >= 1) || !(r[0] <= (double)T + 1) || !(r[1] >= 0) || !(r[1] <= (double)T)
                || r[0] != floor(r[0]) || r[1] != floor(r[1]))
            mexErrMsgIdAndTxt("spm_hmm_forward_backward:invalidInput",
                "xirange must be [first last] with 1 <= first <= T+1 and 0 <= last <= T.");
        x0 = (mwSize)r[0] - 1; if (x0 < 1) x0 = 1;    /* first >= 1, so no wrap */
        if (r[1] < 1) { x0 = 1; x1 = 0; }            /* empty range */
        else x1 = (mwSize)r[1] - 1;
    }

    plhs[0] = mxCreateDoubleMatrix(T, K, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(K, K, mxREAL);
    gamma   = mxGetPr(plhs[0]);
    Xi      = mxGetPr(plhs[1]);
    if (T == 0 || K == 0) {
        plhs[2] = mxCreateDoubleScalar(0.0);
        return;
    }

    /* Work arrays are stored with time as the slow index: x[t*K + k] */
    A     = (double *)mxMalloc(K*K*sizeof(double));   /* A[i*K + j]: i -> j */
    B     = (double *)mxMalloc(T*K*sizeof(double));
    alpha = (double *)mxMalloc(T*K*sizeof(double));
    beta  = (double *)mxMalloc(T*K*sizeof(double));
    c     = (double *)mxMalloc(T*sizeof(double));
    tmp   = (double *)mxMalloc(K*sizeof(double));

    for (i = 0; i < K; i++)
        for (j = 0; j < K; j++)
            A[i*K + j] = exp(lA[i + j*K]);

    /* Emission potentials, max-subtracted per sample */
    for (t = 0; t < T; t++)
        for (j = 0; j < K; j++)
            if (mxIsNaN(lb[t + j*T])) {
                mxFree(A); mxFree(B); mxFree(alpha); mxFree(beta); mxFree(c); mxFree(tmp);
                mexErrMsgIdAndTxt("spm_hmm_forward_backward:invalidInput",
                    "lb contains NaN (set missing samples to zero).");
            }
    for (t = 0; t < T; t++) {
        mx = lb[t];
        for (j = 1; j < K; j++) if (lb[t + j*T] > mx) mx = lb[t + j*T];
        if (!mxIsFinite(mx)) { nbad++; continue; }
        sum_mx += mx;
        for (j = 0; j < K; j++) B[t*K + j] = exp(lb[t + j*T] - mx);
    }
    if (nbad > 0) {
        mxFree(A); mxFree(B); mxFree(alpha); mxFree(beta); mxFree(c); mxFree(tmp);
        mexErrMsgIdAndTxt("spm_hmm_forward_backward:numerical",
            "Every state has zero likelihood at %d sample(s); check the scale of the data.",
            (int)nbad);
    }

    /* Forward pass */
    s = 0.0;
    for (j = 0; j < K; j++) { alpha[j] = exp(lpi[j])*B[j]; s += alpha[j]; }
    c[0] = s;
    for (j = 0; j < K; j++) alpha[j] /= s;
    for (t = 1; t < T; t++) {
        const double *ap = alpha + (t-1)*K, *Bt = B + t*K;
        double *a = alpha + t*K;
        for (j = 0; j < K; j++) a[j] = 0.0;
        for (i = 0; i < K; i++) {
            const double x = ap[i], *Ai = A + i*K;
            for (j = 0; j < K; j++) a[j] += x*Ai[j];
        }
        s = 0.0;
        for (j = 0; j < K; j++) { a[j] *= Bt[j]; s += a[j]; }
        c[t] = s;
        for (j = 0; j < K; j++) a[j] /= s;
    }
    for (t = 0; t < T; t++) {
        if (!(c[t] > 0.0) || !mxIsFinite(c[t])) {
            mxFree(A); mxFree(B); mxFree(alpha); mxFree(beta); mxFree(c); mxFree(tmp);
            mexErrMsgIdAndTxt("spm_hmm_forward_backward:numerical",
                "Zero probability path (underflow in transitions).");
        }
    }

    /* Backward pass, with Xi(i,j) += alpha(t-1,i) A(i,j) B(t,j) beta(t,j) / c(t)
     * for the transitions into samples x0 to x1 */
    for (j = 0; j < K; j++) beta[(T-1)*K + j] = 1.0;
    for (t = T-1; t > 0; t--) {
        const double *bn = beta + t*K, *Bt = B + t*K, *ap = alpha + (t-1)*K;
        double *b = beta + (t-1)*K;
        for (j = 0; j < K; j++) tmp[j] = Bt[j]*bn[j]/c[t];
        if (t >= x0 && t <= x1) {
            for (i = 0; i < K; i++) {
                const double *Ai = A + i*K, x = ap[i];
                double acc = 0.0, v;
                for (j = 0; j < K; j++) {
                    v = Ai[j]*tmp[j];
                    acc += v;
                    Xi[i + j*K] += x*v;
                }
                b[i] = acc;
            }
        } else {
            for (i = 0; i < K; i++) {
                const double *Ai = A + i*K;
                double acc = 0.0;
                for (j = 0; j < K; j++) acc += Ai[j]*tmp[j];
                b[i] = acc;
            }
        }
    }

    /* Posteriors and log normaliser */
    for (t = 0; t < T; t++) {
        logZ += log(c[t]);
        for (j = 0; j < K; j++) gamma[t + j*T] = alpha[t*K + j]*beta[t*K + j];
    }
    plhs[2] = mxCreateDoubleScalar(logZ + sum_mx);

    mxFree(A); mxFree(B); mxFree(alpha); mxFree(beta); mxFree(c); mxFree(tmp);
}
