function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for %
%regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad
% Initialize some useful values
m = length(y);
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================


J=(sum((theta'*X'-y').^2)/(2*m))+((lambda/(2*m))*sum(theta(2:end).^2));
grad0=(1/m)*((theta'*X'-y')*X)';
gradj=(1/m)*((theta'*X'-y')*X)+(lambda/m)*theta';
grad=[grad0(1),gradj(2:end)];








% =========================================================================

grad = grad(:);

end
