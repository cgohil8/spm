function p = osld_paths(dir)
% Paths of an osl-dynamics installation made by spm_osld_install
% FORMAT p = osld_paths(dir)
%
% dir      - installation folder                     [~/.spm/osl-dynamics]
%
% p        - .home, .settings (settings file), .default_dir, .dir, .env
%            (Python environment), .python (its executable), .bin, .uv,
%            .python_install (uv-managed Pythons), .cache, .module (folder
%            of spm_osld.py)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ispc, home = getenv('USERPROFILE'); else, home = getenv('HOME'); end
if isempty(home)
    error('spm_osld:noHome','Cannot find the home folder (HOME or USERPROFILE).');
end
p.home        = home;
p.settings    = fullfile(home,'.spm','spm_osld.json');
p.default_dir = fullfile(home,'.spm','osl-dynamics');
if nargin < 1 || isempty(dir), dir = p.default_dir; end
p.dir = char(dir);
p.env = fullfile(p.dir,'env');
if ispc
    p.python = fullfile(p.env,'Scripts','python.exe');
    p.uv     = fullfile(p.dir,'bin','uv.exe');
else
    p.python = fullfile(p.env,'bin','python');
    p.uv     = fullfile(p.dir,'bin','uv');
end
p.bin            = fullfile(p.dir,'bin');
p.python_install = fullfile(p.dir,'python');
p.cache          = fullfile(p.dir,'cache');
p.module         = fullfile(fileparts(fileparts(mfilename('fullpath'))),'python');
