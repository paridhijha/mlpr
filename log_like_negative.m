function [NLp, dNLp_dw] = log_like_negative(ww, xx, yy)
% this fn returns the negative log likelihood
yy = (yy==1)*2 - 1;
sigmas = 1./(1 + exp(-yy.*(xx*ww)));
NLp = -sum(log(sigmas));
if nargout > 1% additionally returns the derivate w
    dNLp_dw = (((-1)*(1-sigmas).*yy)' * xx)';
end

