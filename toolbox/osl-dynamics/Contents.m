% osl-dynamics: dynamic network models of M/EEG data
%
% Installation
%   spm_osld_install        - install osl-dynamics in a Python environment
%   spm_osld_python         - the Python with osl-dynamics in use
%
% Data preparation
%   spm_osld_parcellate          - parcel time courses from a DAiSS LCMV beamformer
%   spm_osld_orthogonalise       - symmetric orthogonalisation (leakage correction)
%   spm_osld_sign_flip           - align the signs of channels across sessions
%
% Models
%   spm_osld_hmm                 - fit a hidden Markov model
%
% Post-hoc analysis
%   spm_osld_spectra             - state multitaper spectra
%   spm_osld_spectral_components - spectral components of coherence (NMF)
%   spm_osld_power_maps          - power maps from state spectra
%   spm_osld_coherence_networks  - coherence networks from state spectra
%   spm_osld_threshold_network   - keep the strongest edges of networks
%   spm_osld_parcel_maps         - parcel values as a NIfTI map; parcel centres
%   spm_osld_parcel_weights      - weights that combine sources into parcels
%
% Plots
%   spm_osld_plot_time_course    - state time courses
%   spm_osld_plot_summary_stats  - summary statistics of the state time courses
%   spm_osld_plot_spectra        - state power spectra
%   spm_osld_plot_power_map      - parcel maps on the cortical surface
%   spm_osld_plot_network        - networks on glass-brain views
%
% Batch interface (SPM > Tools > osl-dynamics)
%   tbx_cfg_osld                 - leakage correction, sign flipping, HMM
%
% Overview: toolbox/osl-dynamics/README.md
% Example TDE-HMM pipeline: toolbox/osl-dynamics/examples (README.md)
%__________________________________________________________________________

% Copyright (C) 2026 NDCN, University of Oxford
