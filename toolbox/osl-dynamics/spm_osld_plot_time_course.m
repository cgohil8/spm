function F = spm_osld_plot_time_course(states,S)
% Plot state time courses (state probabilities or Viterbi paths)
% FORMAT F = spm_osld_plot_time_course(states,S)
%
% states          - one of:
%                   a fitted model from spm_osld_hmm (uses .path, .missing
%                   and .meeg.fsample);
%                   a cell array of [T_i x K] state probabilities;
%                   a cell array of [T_i x 1] state sequences (Viterbi paths)
% S               - options:
%   .segments       segments to plot, one panel each                     [1]
%   .samples        [first last] samples of each segment to plot  [1 to 2000]
%   .fs             sampling frequency in Hz, for a time axis in seconds
%                   [the model's, else none (sample numbers)]
%   .K              number of states (for state sequences)  [largest state]
%   .missing        cell array of [T_i x 1] logical, true for missing
%                   samples, which are left blank   [the model's, else none]
%   .title          figure title                                       ['']
%
% F               - figure handle
%
% As osl_dynamics.utils.plotting.plot_alpha: the state probabilities of
% each sample are stacked, with a colour bar labelled with the state
% numbers. A state sequence is shown as one-hot probabilities.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
def = struct('segments',1,'samples',[1 2000],'fs',[],'K',[],'missing',{{}},'title','');

%-State probabilities from the input
%--------------------------------------------------------------------------
if isstruct(states)
    H = states;
    if isfield(H,'path') && ~isempty(H.path)
        states = cellfun(@double,H.path,'UniformOutput',false);
        if isfield(H,'missing'), def.missing = H.missing; end
        if isfield(H,'meeg'), def.fs = H.meeg.fsample; end
        def.K = H.K;
    else
        error('spm_osld:invalidInput','The model has no Viterbi paths (.path).');
    end
end
if ~iscell(states), states = {states}; end
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}) || isempty(S.(fn{i})), S.(fn{i}) = def.(fn{i}); end
end
ispath = @(p) isrow(p) && numel(p) > 1 && all(p >= 1 & p == round(p));
for i = 1:numel(states)                                 % paths as columns
    if ispath(double(states{i})), states{i} = states{i}(:); end
end
if size(states{1},2) == 1                               % state sequences
    K = S.K;
    if isempty(K), K = max(cellfun(@max,states)); end
    states = cellfun(@(p) full(sparse(1:numel(p),double(p),1,numel(p),K)),...
        states,'UniformOutput',false);
end
K   = size(states{1},2);
col = state_colours(K);

%-One panel per segment
%--------------------------------------------------------------------------
F  = figure('Color','w','Name','State time courses');
ns = numel(S.segments);
axs = gobjects(ns,1);
for j = 1:ns
    i  = S.segments(j);
    g  = states{i};
    t1 = min(S.samples(2),size(g,1));
    idx = S.samples(1):t1;
    if isempty(idx)
        error('spm_osld:invalidInput','Segment %d has no samples in S.samples.',i);
    end
    g  = g(idx,:);
    if ~isempty(S.missing), g(S.missing{i}(idx),:) = NaN; end
    if isempty(S.fs), t = idx; else, t = (idx - 1)/S.fs; end
    ax = subplot(ns,1,j); axs(j) = ax;
    h  = area(ax,t,g,'EdgeColor','none');
    for k = 1:K, set(h(k),'FaceColor',col(k,:),'Tag','state'); end
    if t(end) > t(1), xlim(ax,[t(1) t(end)]); end
    ylim(ax,[0 1]);
    if ns > 4                                           % short labels
        set(ax,'YTick',[]); ylabel(ax,sprintf('%d',i),'Rotation',0,...
            'HorizontalAlignment','right','VerticalAlignment','middle');
    else
        ylabel(ax,sprintf('Segment %d',i));
    end
    box(ax,'off');
end
set(axs(1:end-1),'XTickLabel',[]);
if isempty(S.fs), xlabel(ax,'Sample'); else, xlabel(ax,'Time (s)'); end

%-Colour bar labelled with the state numbers
%--------------------------------------------------------------------------
colormap(ax,col);
cb = colorbar(ax,'eastoutside');
caxis(ax,[0 K]); %#ok<CAXIS> clim needs R2022a
set(cb,'Ticks',(1:K) - 0.5,'TickLabels',arrayfun(@num2str,1:K,'UniformOutput',false));
cb.Label.String = 'State';
w = get(ax,'Position');                                 % equal panel widths
for a = axs', p = get(a,'Position'); set(a,'Position',[p(1) p(2) w(3) p(4)]); end
top = get(axs(1),'Position');                           % bar spans all panels
c   = get(cb,'Position');
set(cb,'Position',[c(1) w(2) c(3) top(2) + top(4) - w(2)]);
if ~isempty(S.title), sgtitle(F,S.title); end
