function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================

C=[0.01 0.03 0.1 0.3 1 3 10];
sigma=[0.01 0.03 0.1 0.3 1 3 10];

for i=1:length(C)
for j=1:length(sigma)

model = svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j))); 
predictions = svmPredict(model, Xval);
errors(i,j) = mean(double(predictions ~= yval));

end;
end;


[minval,ind]=min(errors(:))
[I,J] = ind2sub([size(errors,1) size(errors,2)],ind)
C=C(I)
sigma=sigma(J)





% =========================================================================

end
