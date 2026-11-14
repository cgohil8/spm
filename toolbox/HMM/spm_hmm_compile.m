function spm_hmm_compile(outdir)
% Compile the C forward-backward recursion of the HMM toolbox
% FORMAT spm_hmm_compile
% FORMAT spm_hmm_compile(outdir)
%
% outdir  - folder for the compiled file              [toolbox/HMM/private]
%
% Builds private/forward_backward_mex.c with mex, then checks that the
% compiled recursion agrees with the MATLAB code. A C compiler must be set
% up for MATLAB first (mex -setup C; on a Mac this needs Xcode).
%
% Compiling is optional: without the compiled file the toolbox uses the
% MATLAB recursion, with the same results. The compiled file makes each
% E-step about 1.2-1.7 times faster with few channels (e.g. D <= 40) and
% gains little with many (see HMM_maths.md, section 12). It is not built
% with the other SPM MEX files, so it is compiled here, once per platform.
% From a shell, 'make' in toolbox/HMM/private does the same build.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


here = fileparts(mfilename('fullpath'));
src  = fullfile(here,'private','forward_backward_mex.c');
if nargin < 1 || isempty(outdir), outdir = fileparts(src); end

cc = mex.getCompilerConfigurations('C','Selected');
if isempty(cc)
    error('spm_hmm:noCompiler',['No C compiler is set up for MATLAB. Run ' ...
        '''mex -setup C'' first (on a Mac this needs Xcode).']);
end
fprintf('spm_hmm_compile: compiling %s with %s\n',src,cc.Name);
mex('-O','-largeArrayDims','-outdir',outdir,src);
clear('forward_backward_mex','c_version_available');

%-Check the compiled recursion against the MATLAB code
%--------------------------------------------------------------------------
if ~strcmp(outdir,fileparts(src))
    fprintf('spm_hmm_compile: compiled to %s (not checked)\n',outdir);
    return
end
rs  = RandStream('mt19937ar','Seed',0);
K   = 4;
lb  = 3*randn(rs,500,K);
lA  = log(0.9*eye(K) + 0.1/K) - 0.01;
lpi = log(ones(K,1)/K);
[g0,X0,z0] = spm_hmm_forward_backward(lpi,lA,lb,'matlab');
[g1,X1,z1] = spm_hmm_forward_backward(lpi,lA,lb,'c');
err = max([max(abs(g1(:) - g0(:))), max(abs(X1(:) - X0(:)))/max(X0(:)), abs(z1 - z0)/abs(z0)]);
if err > 1e-10
    error('spm_hmm:compileCheck',['The compiled recursion differs from the ' ...
        'MATLAB code by %g; delete it from %s.'],err,outdir);
end
fprintf('spm_hmm_compile: done; agrees with the MATLAB recursion to %.1g\n',err);
