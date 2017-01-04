function [X, fX, i] = mini(X,f,varargin)
si=size(X);
fX=si;
i=0;
feval(f, X, varargin{:}); 
end