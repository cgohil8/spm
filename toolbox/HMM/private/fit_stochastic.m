function [q,Fhist,Finit,rho] = fit_stochastic(R,S,prior,stream)
% Stochastic variational inference for the HMM
% FORMAT [q,Fhist,Finit,rho] = fit_stochastic(R,S,prior,stream)
%
% R is a data reader (see data_reader).
%
% 1. Initialise: batch VB (S.ninit restarts, S.initcyc cycles) on a random
%    subset of chunks treated as separate segments, with the full-data
%    prior (chunks with no observed samples are skipped). Its statistics
%    are rescaled to the size of the full dataset; the transition counts
%    and initial-state counts to their known totals.
%    (With S.init a cell array of state probabilities, the posterior is
%    initialised from them on all the data instead.)
% 2. For each mini-batch B of chunks (a random partition every epoch):
%       t_hat = (M/|B|) * statistics owned by the chunks in B
%       eta   = (1 - rho_i)*eta + rho_i*(eta_0 + t_hat)
%       rho_i = (i + delay)^(-forget)
%    M is the number of chunks.
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

%-Initialise on a random subset of chunks with observed samples
%--------------------------------------------------------------------------
perm  = randperm(stream,M);
ntarg = min(M,o.initchunks);
Ysub  = cell(ntarg,1);
m     = 0;                             % chunks kept
k     = 0;                             % chunks examined
while m < ntarg && k < M
    k = k + 1;
    c = chunks(perm(k),:);
    y = R.read(c(1),c(2),c(3));
    if all(any(isnan(y),2)), continue; end
    m = m + 1;
    Ysub{m} = y;
end
Ysub = Ysub(1:m);
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
scale = M/k;                           % the k chunks examined are a random sample
% every subset chunk starts a "segment" (the data have R.nseg starts), and
% the transitions into the chunk starts are missing (the data have
% sum(R.T) - R.nseg transitions): rescale both to their totals
st.g1 = st.g1*R.nseg/(scale*sum(st.g1));
nx    = sum(st.Xi(:));
if nx > 0, st.Xi = st.Xi*(sum(R.T) - R.nseg)/(scale*nx); end
q     = spm_hmm_update_params(st,prior,[],1,scale);
if S.verbose
    fprintf('spm_hmm: initialised on %d of %d chunks, F(subset) = %.6g\n',...
        m,M,max(Finit));
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
        st = spm_hmm_infer_states(R,q,prior,parallel_workers(S.parallel,R,'segments'));
        Fhist(end+1) = st.logZ - spm_hmm_kl_divergence(q,prior); %#ok<AGROW>
        if S.verbose
            fprintf('spm_hmm: epoch %d/%d, rho = %.3g, F = %.8g\n',ep,o.nepochs,rho(i),Fhist(end));
        end
    elseif S.verbose
        fprintf('spm_hmm: epoch %d/%d, rho = %.3g\n',ep,o.nepochs,rho(i));
    end
end
rho = rho(1:i);
