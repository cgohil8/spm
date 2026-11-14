function [gamma,path,missing] = spm_hmm_decode(Y,HMM,segments)
% State probabilities and most probable state sequences for fitted model
% FORMAT [gamma,path,missing] = spm_hmm_decode(Y,HMM,segments)
%
% Y         - data: [T x D] matrix, cell array of [T_i x D] segments, or a
%             data reader (see spm_hmm_meeg_reader). A matrix is split into
%             segments of lengths HMM.options.T if the model was fitted that
%             way (and they add up to its rows); otherwise it is one segment
% HMM       - fitted model (see spm_hmm)
% segments  - indices of the segments to decode                     [all]
%
% gamma     - cell array of [T_i x K] state probabilities q(s_t = k)
% path      - cell array of [T_i x 1] Viterbi paths: the mode of q(s_1:T)
% missing   - cell array of [T_i x 1] logical, true for missing samples
%             (the path and gamma continue through them, informed only
%             by the transition model)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


T = [];
if isnumeric(Y) && isfield(HMM,'options') && isfield(HMM.options,'T') && ...
        ~isempty(HMM.options.T) && sum(HMM.options.T) == size(Y,1)
    T = HMM.options.T;
end
R = data_reader(Y,T);
if nargin < 3 || isempty(segments), segments = 1:R.nseg; end
q     = HMM.q;
block = 1e5;

gamma   = cell(numel(segments),1);
path    = cell(numel(segments),1);
missing = cell(numel(segments),1);
for j = 1:numel(segments)
    i  = segments(j);
    T  = R.T(i);
    lb = zeros(T,q.K);
    m  = false(T,1);
    for lo = 1:block:T
        hi = min(lo+block-1,T);
        y  = R.read(i,lo,hi);
        mi = any(isnan(y),2);
        l  = spm_hmm_expected_loglik(y,q);
        l(mi,:) = 0;
        lb(lo:hi,:) = l;
        m(lo:hi)    = mi;
    end
    gamma{j}   = spm_hmm_forward_backward(q.lpi,q.lA,lb);
    missing{j} = m;
    if nargout > 1
        path{j} = spm_hmm_viterbi(q.lpi,q.lA,lb);
    end
end
