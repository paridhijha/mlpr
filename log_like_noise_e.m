function [Lp, dLp_de] = log_like_noise_e(e, xx, yy, ww)
yy = (yy==1)*2 - 1;
% this fn returns the log likelihood for a model with noise e
% and the gradient e
sigmas = 1./(1 + exp(-yy.*(xx*ww))); % Nx1
Lp = sum(log((1-e)*sigmas+(e/2)));
if nargout > 1
    dLp_de = sum((-1.*sigmas + (1/2)).* (1./((1-e)*sigmas+(e/2))));
end
