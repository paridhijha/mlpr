function [Lp, dLp_de] = lr_loglike_noise_constrained_e(a, xx, yy, ww)
% modified so it produces the negative log likelihood
% and the relevant gradient

%LR_LOGLIKE log-likelihood and gradients of logistic regression
%
%     [Lp, dLp_dw] = lr_loglike(ww, xx, yy);
%
% Inputs:
%          ww Dx1 logistic regression weights
%          xx NxD training data, N feature vectors of length D
%          yy Nx1 labels in {+1,-1} or {1,0}
%
% Outputs:
%          Lp 1x1 negative log-probability of data, the log-likelihood of ww
%      dLp_dw Dx1 gradients: partial derivatives of Lp wrt ww

% Iain Murray, October 2014, August 2015

% Ensure labels are in {+1,-1}:
yy = (yy==1)*2 - 1;
epsilon = 1./(1 + exp(-a));

sigmas = 1./(1 + exp(-yy.*(xx*ww))); % Nx1
Lp = sum(log((1-epsilon)*sigmas+(epsilon/2)));

log_derivative = (1./((1-epsilon)*sigmas+(epsilon/2)));
if nargout > 1
    dLp_de = sum(epsilon*(1-epsilon)*(-1.*sigmas + 0.5).* log_derivative);
end

% WARNING: The sigmas can numerically saturate to 1 for large weights. (Or zero,
% if the weights are a terrible explanation of the data). The gradient signal
% (1-sigmas) will then evaluate to zero. We could stave off problems for a
% little longer using Lp=-sum(log1p(exp(...))), and computing (1-sigmas) as
% 1./(1 + exp(yy.*(xx*ww))); However, it doesn't matter how careful we are,
% eventually the gradient *will* underflow for big enough weights. Netlab clips
% the input to the sigmoid functions to prevent total saturation, which could be
% considered. However, the way the computation is arranged here, nothing too bad
% happens (? -- at least we don't get NaNs!). The *real* problem is that models
% with sigmoids that can saturate so much are unrealistically confident. If we
% get numerical problems, our attention should really be on changing the models,
% so we can't get physically unrealistic probabilities in the first place.

