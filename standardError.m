function SE = standardError(x)
% Standard error
% Sqrt( Sum( (mean - x)^2 ) / (N-1) ) / Sqrt(N)
expectation = mean(x);
standardDeviation =  sqrt(sum((expectation-x).^2)/(length(x)-1));
SE = standardDeviation/sqrt(length(x));
end