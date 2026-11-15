function c = state_colours(K)
% Colours for K states: matplotlib's 'tab10' (as osl-dynamics), repeated
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


tab10 = [ 31 119 180; 255 127  14;  44 160  44; 214  39  40; 148 103 189;
         140  86  75; 227 119 194; 127 127 127; 188 189  34;  23 190 207]/255;
c = tab10(mod(0:K-1,10) + 1,:);
