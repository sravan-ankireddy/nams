function [e_iter,e_final,wn1,wn2,e_iter1] = Linear_LMS2_Load(x1,x2,y,step,iter,post_cursor, pre_cursor,w1,w2)

M1 = pre_cursor;
M2 = post_cursor;

x1 = [zeros(M1,1);x1;zeros(M2,1)];
x2 = [zeros(M1,1);x2;zeros(M2,1)];
wn = [w1;w2];
xn = zeros((M1 + M2 + 1)*2, iter);


e_iter1 = zeros(iter,1);
e_iter  = zeros(iter,1);
e_final = zeros(iter,1);

for i = 1:iter
    xn(:,i) = [x1(i : i + M1 + M2,1);x2(i : i + M1+M2,1)];
    
    e_iter1(i) = y(i) - conj(wn(1:M1+M2+1)')*xn(1:M1+M2+1,i);
    e_iter(i) = y(i) - conj(wn')*xn(:,i);
    wn = wn + step*e_iter(i).*conj(xn(:,i));
end
wn1 = wn(1:M1+M2+1,1);
wn2 = wn(M1+M2+2:end,1);
for i = 1:iter
    e_final(i,1) = y(i) - conj(wn')*xn(:,i);
end