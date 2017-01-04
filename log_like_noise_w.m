function [Lp, dLp_dw] = log_like_noise_w(ww, xx, yy, e)
% this fn returns the log likelihood for a model with noise e
% and the gradient w
yy = (yy==1)*2 - 1;
sigmas = 1./(1 + exp(-yy.*(xx*ww))); % Nx1
Lp = sum(log((1-e)*sigmas+(e/2)));
if nargout > 1
    dLp_dw = (((((1-e)./((1-e)*sigmas+e/2)).*sigmas.*(1-sigmas)).*yy)' * xx)';
end

