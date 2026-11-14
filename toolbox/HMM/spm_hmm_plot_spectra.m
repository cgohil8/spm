function F = spm_hmm_plot_spectra(spec,S)
% Plot state power spectra
% FORMAT F = spm_hmm_plot_spectra(spec,S)
%
% spec            - spectra from spm_hmm_multitaper / spm_eeg_hmm_spectra
%                   (.f and .psd [G x K x D x F]: sessions, states,
%                   channels, frequencies)
% S               - options:
%   .states         states to plot                                   [1:K]
%   .channels       channels (indices) to include                    [1:D]
%   .freq           [fmin fmax] frequency range shown       [whole range]
%   .subtractmean   subtract the (weighted) mean over all states, to show
%                   each state relative to the average            [false]
%   .meanweights    [K x 1] weights for that mean, e.g. the fractional
%                   occupancies                                [uniform]
%   .title          figure title                                       ['']
%
% F               - figure handle
%
% One line per state: the PSD averaged over channels, then over sessions,
% with the standard error over sessions shaded (G > 1).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
psd = spec.psd;
sz  = size(psd); sz(end+1:4) = 1;
[G,K,D] = deal(sz(1),sz(2),sz(3));
def = struct('states',1:K,'channels',1:D,'freq',[],...
    'subtractmean',false,'meanweights',[],'title','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}) || (isempty(S.(fn{i})) && ~isempty(def.(fn{i})))
        S.(fn{i}) = def.(fn{i});
    end
end
f = spec.f(:)';
if isempty(S.freq), S.freq = [f(1) f(end)]; end

if S.subtractmean && K > 1
    w = S.meanweights; if isempty(w), w = ones(K,1); end
    w = reshape(w/sum(w),1,K);
    psd = psd - sum(psd.*w,2);
end
psd = psd(:,:,S.channels,:);
col = state_colours(K);

F  = figure('Color','w','Name','State spectra');
ax = axes(F); hold(ax,'on');
P  = reshape(mean(psd,3),[G K numel(f)]);           % channel mean
m  = reshape(mean(P,1),[K numel(f)]);
h  = gobjects(1,numel(S.states));
for j = 1:numel(S.states)
    k = S.states(j);
    if G > 1
        se = reshape(std(P(:,k,:),0,1),1,[])/sqrt(G);
        fill(ax,[f fliplr(f)],[m(k,:) - se fliplr(m(k,:) + se)],col(k,:),...
            'FaceAlpha',0.2,'EdgeColor','none','Tag','error');
    end
    h(j) = plot(ax,f,m(k,:),'Color',col(k,:),'LineWidth',1.5,'Tag','psd',...
        'DisplayName',sprintf('State %d',k));
end
legend(ax,h,'Location','northeast','Box','off');
if S.freq(2) > S.freq(1), xlim(ax,S.freq); end
xlabel(ax,'Frequency (Hz)');
if S.subtractmean, ylabel(ax,'PSD relative to mean (a.u.)'); else, ylabel(ax,'PSD (a.u.)'); end
box(ax,'off');
if ~isempty(S.title), sgtitle(F,S.title); end

