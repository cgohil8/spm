function [q,Fhist,Finit,rho] = fit_stochastic(R,S,prior,stream)
% Stochastic variational inference for the HMM
% FORMAT [q,Fhist,Finit,rho] = fit_stochastic(R,S,prior,stream)
%
% R is a data reader (see data_reader).
%
% 1. Initialise: batch VB (S.ninit restarts, S.initcyc cycles) on a random
%    subset of chunks treated as separate segments, with the full-data
%    prior. Its statistics are rescaled to the size of the full dataset.
%    (With S.init a cell array of state probabilities, the posterior is
%    initialised from them on all the data instead.)
% 2. For each mini-batch B of chunks (a random partition every epoch):
%       t_hat = (M/|B|) * statistics owned by the chunks in B
%       eta   = (1 - rho_i)*eta + rho_i*(eta_0 + t_hat)
%       rho_i = (i + delay)^(-forget)
%    M is the number of chunks; uniform sampling makes t_hat unbiased.
% See HMM_maths.md, sections 8 and 11.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


o      = S.svi;
chunks = spm_hmm_make_chunks(R.T,o.chunklength,o.buffer);
M      = size(chunks,1);

%-Initialise from user-supplied state probabilities (all data)
%--------------------------------------------------------------------------
if iscell(S.init)
    gamma = initial_state_probs(R,S,prior,stream);
    q     = spm_hmm_update_params(sufficient_stats_from_probs(R,gamma,prior),prior);
    Finit = NaN;
    if S.verbose, fprintf('spm_hmm: initialised from S.init\n'); end
else

%-Initialise on a subset of chunks
%--------------------------------------------------------------------------
idx  = randperm(stream,M,min(M,o.initchunks));
Ysub = cell(numel(idx),1);
for j = 1:numel(idx)
    c = chunks(idx(j),:);
    Ysub{j} = R.read(c(1),c(2),c(3));
end
Si          = S;
Si.inference = 'batch';
Si.T        = [];
Si.keepgamma = false;
Si.maxcyc   = S.initcyc;
Si.verbose  = false;
Si.seed     = randi(stream,2^31-1);
if isempty(Si.initdiag)                % HMM-MAR default from segment lengths
    Si.initdiag = min(median(R.T)/10,500);
end
Si.prior    = struct('alpha0',prior.alpha0,'beta0',prior.beta0,'m0',prior.m0,...
    'kappa0',prior.kappa0,'nu0',prior.nu0,'W0',prior.W0);
Hsub  = spm_hmm(Ysub,Si);
Finit = Hsub.Finit;
st    = spm_hmm_infer_states(Ysub,Hsub.q,prior);
% every subset chunk starts a "segment"; the data have R.nseg starts
st.g1 = st.g1*R.nseg/M;
q     = spm_hmm_update_params(st,prior,[],1,M/numel(idx));
if S.verbose
    fprintf('spm_hmm: initialised on %d of %d chunks, F(subset) = %.6g\n',...
        numel(idx),M,max(Finit));
end
end

%-Stochastic updates
%--------------------------------------------------------------------------
nb    = min(o.batchsize,M);
nbat  = ceil(M/nb);
rho   = zeros(1,o.nepochs*nbat);
Fhist = [];
i     = 0;
for ep = 1:o.nepochs
    perm = randperm(stream,M);
    for b = 1:nbat
        j  = perm((b-1)*nb+1:min(b*nb,M));
        i  = i + 1;
        rho(i) = (i + o.delay)^(-o.forget);
        st = spm_hmm_infer_states_chunked(R,chunks(j,:),q,prior);
        q  = spm_hmm_update_params(st,prior,q,rho(i),M/numel(j));
    end
    if o.Fevery > 0 && mod(ep,o.Fevery) == 0 && ep < o.nepochs
        st = spm_hmm_infer_states(R,q,prior);
        Fhist(end+1) = st.logZ - spm_hmm_kl_divergence(q,prior); %#ok<AGROW>
        if S.verbose
            fprintf('spm_hmm: epoch %d/%d, rho = %.3g, F = %.8g\n',ep,o.nepochs,rho(i),Fhist(end));
        end
    elseif S.verbose
        fprintf('spm_hmm: epoch %d/%d, rho = %.3g\n',ep,o.nepochs,rho(i));
    end
end
rho = rho(1:i);
