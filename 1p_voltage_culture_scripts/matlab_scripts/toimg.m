function out = toimg(mat, nr, nc);

% function out = toimg(mat, nr, nc);
% converts a matrix of size [nr*nc, nt] to a movie of size [nr, nc, nt];
%
% modified from code by Vicente Parot

%
% 2013-2017 Vicente Parot
% Cohen Lab - Harvard University
% 
if nargin < 3
    assert(numel(nr)==2,'single size argument must be [nr nc]');
    nc = nr(2);
    nr = nr(1);
end
out = reshape(mat, nr, nc, []);