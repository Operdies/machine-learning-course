function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
regTheta = theta;
regTheta(1) = 0;
h0x = X*theta;
% You need to return the following variables correctly 
regCost = (lambda/(2*m))*sum(regTheta.^2);
regGradientCost = (lambda/m)*regTheta;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = (1/(2*m))*sum((h0x-y).^2) + regCost;
grad = (1/m)*(h0x'-y')*X + regGradientCost';


% =========================================================================

grad = grad(:);

end
