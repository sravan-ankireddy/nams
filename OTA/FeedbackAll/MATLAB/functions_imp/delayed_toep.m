function Y = delayed_toep(X,m,n)
first_row = [X(n:-1:1).' zeros(1,m)];
first_col = [X(n:1:end);zeros(n-1,1)];
Y = toeplitz(first_col,first_row);
end