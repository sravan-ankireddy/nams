function [e_iter,e_final,wn] = Linear_Lasso_LMS(x,y,step,rho,iter,pre_cursor,post_cursor)
% ZA LMS
% REF:
% [1] Regularised LMS
% [2] Sparse LMS for system Identification

M1 = pre_cursor;
M2 = post_cursor;

% Initialization
x = [zeros(M1,1);x;zeros(M2,1)];
wn = zeros(M1 + M2 + 1, 1);
xn = zeros(M1 + M2 + 1, iter);
yn = zeros(iter,1);

e_iter  = zeros(iter,1);
e_final = zeros(iter,1);

for i = 1:iter
    xn(:,i) = x( i: i + M1 + M2,1);   % Length of the Filter = M1+ M2 + 1
    yn(i,1) = y(i,1);
    e_iter(i) = yn(i) - conj(wn')*xn(:,i);
    wn = wn + step*e_iter(i).*conj(xn(:,i)) - rho*sign(wn);
end

for i = 1:iter
    e_final(i,1) = yn(i) - conj(wn')*xn(:,i);
end
end
