function [df, dt, x] = read_VNA_compex_v2(filename)


h = load(filename);
x = h(:,1);

df = h(:,2)+i*h(:,3);
dt = ifft(df);