function [params] = split_params(vector)
% slice_sample returns a matrix, this fn
% extracts the 3 sampled parameters
[rows,cols]=size(vector);
%initial case while starting slice_sample
if rows == 1
    e=vector(1);
    log_lambda=vector(2);
    weights=vector(3:end);
    params={e,log_lambda,weights'};
    return
else
    e=vector(1,:);
    log_lambda=vector(2,:);
    weights=vector(3:end,:);
    params={e,log_lambda,weights}; % the returned result of slice_sample
end