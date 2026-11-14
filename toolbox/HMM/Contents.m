% Hidden Markov models of M/EEG data with osl-dynamics
%
% Installation
%   spm_osld_install              - install osl-dynamics in a Python environment
%   spm_osld_python               - the Python with osl-dynamics in use
%
% M/EEG data
%   spm_eeg_hmm_parcellate        - parcel time courses from a DAiSS LCMV beamformer
%   spm_eeg_hmm_orthogonalise     - symmetric orthogonalisation (leakage correction)
%   spm_eeg_hmm_sign_flip         - align the signs of channels across sessions
%   spm_eeg_hmm_envelope          - amplitude envelopes
%   spm_eeg_hmm                   - fit an HMM
%   spm_eeg_hmm_spectra           - state multitaper spectra
%
% Post-hoc analysis
%   spm_hmm_spectral_components   - spectral components of coherence (NMF)
%   spm_hmm_power_maps            - power maps from state spectra
%   spm_hmm_coherence_networks    - coherence networks from state spectra
%   spm_hmm_threshold_network     - keep the strongest edges of networks
%   spm_hmm_parcel_maps           - parcel values as a NIfTI map; parcel centres
%   spm_hmm_parcel_weights        - weights that combine sources into parcels
%
% Plots
%   spm_hmm_plot_time_course      - state time courses
%   spm_hmm_plot_summary_stats    - summary statistics of the state time courses
%   spm_hmm_plot_spectra          - state power spectra
%   spm_hmm_plot_power_map        - parcel maps on the cortical surface
%   spm_hmm_plot_network          - networks on glass-brain views
%
% Overview: toolbox/HMM/README.md
% Example pipelines: toolbox/HMM/examples (README.md)
%__________________________________________________________________________

% Copyright (C) 2026 NDCN, University of Oxford
