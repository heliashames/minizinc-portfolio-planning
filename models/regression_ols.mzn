% regression_ols.mzn
% Linear regression  y ~ X w  by least squares.
%
%   min_w  || y - X w ||^2
%
% The optimum is characterised by the normal equations (X'X) w = X'y, which are
% LINEAR in w, so we state them as constraints and let the solver find w.
% Column 1 of X is the intercept.

int: n;                                   % number of training weeks
int: k;                                   % number of columns (intercept + features)
array[1..n, 1..k] of float: X;
array[1..n] of float: y;                  % next-week return in %

array[1..k] of var -1000.0..1000.0: w;    % coefficients

constraint forall(j in 1..k)(
    sum(l in 1..k)( (sum(i in 1..n)(X[i,j] * X[i,l])) * w[l] ) = sum(i in 1..n)(X[i,j] * y[i])
);

solve satisfy;
