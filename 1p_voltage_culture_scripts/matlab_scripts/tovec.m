function out = tovec(mov);

% function out = tovec(mov);
% converts a movie of size [nr, nc, nt] to a matrix of size [nr*nc, nt];
%
% modified from code by Vicente Parot

[nr, nc, ~] = size(mov);
out = reshape(mov,nr*nc,[]);