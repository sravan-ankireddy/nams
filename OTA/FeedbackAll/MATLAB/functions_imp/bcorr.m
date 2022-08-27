function [Start_point] = bcorr(X, Y)
[P1,~]=xcorr(X,Y);
b = find(abs(P1)>1000);
b = b(end);
b = b - length(Y);
a = length(X) - b;
Start_point = a;



end

