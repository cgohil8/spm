function s = osld_settings(s)
% Read the toolbox settings file, or merge fields into it
% FORMAT s = osld_settings        - read (empty struct if there is none)
% FORMAT s = osld_settings(s)     - set the fields of s and write the file
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


file = osld_paths().settings;
old  = struct();
if exist(file,'file')
    try
        old = jsondecode(fileread(file));
    catch
        warning('spm_osld:settings','Cannot read %s; it will be replaced.',file);
    end
end
if nargin < 1
    s = old;
    return
end
f = fieldnames(s);
for i = 1:numel(f), old.(f{i}) = s.(f{i}); end
s = old;
if ~exist(fileparts(file),'dir'), mkdir(fileparts(file)); end
fid = fopen(file,'w');
if fid < 0, error('spm_osld:io','Cannot write %s.',file); end
fwrite(fid,jsonencode(s),'char');
fclose(fid);
