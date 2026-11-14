% Hidden Markov model toolbox
%
% Variational Bayesian hidden Markov model with multivariate Gaussian
% observations (Normal-Wishart priors; optional zero-mean states).
% Maths and design: toolbox/HMM/doc/HMM_maths.md
%
%   spm_hmm                  - fit the model (main function)
%   spm_eeg_hmm              - fit the model to M/EEG datasets
%   spm_hmm_meeg_reader      - read M/EEG datasets from disk as segments
%   spm_hmm_tde_pca          - time-delay embedding + PCA (as a reader)
%   spm_eeg_hmm_envelope     - amplitude envelope datasets (AE-HMM)
%   tbx_cfg_hmm              - batch interface (SPM > Tools > HMM)
%   spm_hmm_decode           - state probabilities and Viterbi paths
%   spm_hmm_viterbi          - most probable state sequence (mode of q)
%   spm_hmm_summary_stats    - occupancy, lifetimes, intervals, switching
%   spm_hmm_set_priors       - prior distributions
%   spm_hmm_infer_states     - E-step: q(states) and sufficient statistics
%   spm_hmm_infer_states_chunked - E-step on buffered chunks (stochastic)
%   spm_hmm_make_chunks      - split segments into buffered chunks
%   spm_hmm_update_params    - M-step: q(parameters) (batch or stochastic)
%   spm_hmm_forward_backward - forward-backward recursion
%   spm_hmm_expected_loglik  - expected log-likelihood of each sample/state
%   spm_hmm_kl_divergence    - KL[q(parameters)||p(parameters)]
%   spm_hmm_free_energy      - variational free energy (general form)
%   spm_hmm_simulate         - simulate data from an HMM
%__________________________________________________________________________

% Copyright (C) 2026 NDCN, University of Oxford
