function [ar_param, prediction, residual] = var_model(A,b)
ar_param = pinv(A'*A)*A'*b;
prediction = A * ar_param;
residual = b - prediction;