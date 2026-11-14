function plot_violins(ax,X,col)
% Violin plots of the columns of X (NaN ignored), as seaborn's violinplot
% with osl-dynamics' settings (inner='quart', cut=0)
%
% ax   - axes
% X    - [n x K] values; column k is drawn at x = k
% col  - [K x 3] colours
%
% Each column's density is a Gaussian kernel density estimate with Scott's
% bandwidth (scipy.stats.gaussian_kde: std(x)*n^(-1/5)), evaluated on 100
% points between the smallest and largest value. The densities of all
% violins are divided by the largest of them (density_norm='area', so all
% violins have the same area) and drawn with half-width 0.4. Dashed lines
% mark the quartiles (numpy percentiles), across the violin. A column with
% one distinct value is drawn as a short line.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


K  = size(X,2);
hw = 0.4;
v  = cell(1,K); y = cell(1,K); d = cell(1,K);
for k = 1:K
    x = X(~isnan(X(:,k)),k);
    v{k} = x;
    if numel(unique(x)) < 2, continue; end
    bw   = std(x)*numel(x)^(-1/5);
    y{k} = linspace(min(x),max(x),100)';
    d{k} = mean(exp(-0.5*((y{k} - x')/bw).^2),2)/(bw*sqrt(2*pi));
end
dmax = max([0; cat(1,d{:})]);
hold(ax,'on');
for k = 1:K
    x = v{k};
    if isempty(x), continue; end
    if isempty(d{k})                                    % one distinct value
        plot(ax,k + [-hw hw]/2,[x(1) x(1)],'-','Color',col(k,:),'LineWidth',2,'Tag','violin');
        continue
    end
    s = d{k}/dmax*hw;
    fill(ax,[k - s; flipud(k + s)],[y{k}; flipud(y{k})],col(k,:),...
        'EdgeColor',0.25*[1 1 1],'Tag','violin');
    q  = percentile_linear(x,[25 50 75]);
    sq = interp1(y{k},s,q);
    ls = {':','--',':'};
    for j = 1:3
        plot(ax,k + [-sq(j) sq(j)],[q(j) q(j)],ls{j},'Color',0.25*[1 1 1],...
            'LineWidth',1,'Tag','quartile','UserData',q(j));
    end
end
xlim(ax,[0.5 K + 0.5]);
set(ax,'XTick',1:K);
