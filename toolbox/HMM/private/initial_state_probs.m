function gamma = initial_state_probs(R,S,prior,stream)
% Initial state probabilities for one initialisation
%
% 'random' follows HMM-MAR: a Markov state path is sampled for each segment
% with transition matrix P = (d-1)*I + 1 (rows normalised), d = S.initdiag
% (default min(median(T_i)/10, 500)), a uniformly drawn first state, and a
% small amount of noise added to the one-hot path before renormalising.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


K = prior.K;
if iscell(S.init)
    gamma = S.init(:);
    if numel(gamma) ~= R.nseg
        error('spm_hmm:invalidInput','S.init must have one cell per segment.');
    end
    for i = 1:R.nseg
        if ~isequal(size(gamma{i}),[R.T(i) K]) || any(gamma{i}(:) < 0) || ...
                any(~isfinite(gamma{i}(:))) || any(sum(gamma{i},2) <= 0)
            error('spm_hmm:invalidInput',['S.init{%d} must be a finite, non-negative ' ...
                '[T_i x K] matrix with positive row sums.'],i);
        end
        gamma{i} = gamma{i}./sum(gamma{i},2);
    end
    return
end

switch lower(S.init)
    case 'random'
        T = R.T;
        d = S.initdiag;
        if isempty(d), d = min(median(T)/10,500); end
        d = max(d,1);
        P = (d-1)*eye(K) + ones(K);
        P = cumsum(P./sum(P,2),2);
        P(:,end) = 1;
        gamma = cell(R.nseg,1);
        for i = 1:R.nseg
            u = rand(stream,T(i),1);
            s = zeros(T(i),1);
            s(1) = randi(stream,K);
            for t = 2:T(i)
                s(t) = find(u(t) < P(s(t-1),:),1);
            end
            g = full(sparse(1:T(i),s,1,T(i),K)) + 1e-4*rand(stream,T(i),K);
            gamma{i} = g./sum(g,2);
        end
    otherwise
        error('spm_hmm:invalidInput','Unknown initialisation ''%s''.',S.init);
end
