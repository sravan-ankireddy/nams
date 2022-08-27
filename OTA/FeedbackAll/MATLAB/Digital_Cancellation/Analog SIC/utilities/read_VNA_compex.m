function [df, dt, x] = read_VNA_compex(filename_real,filename_imag)


h_real = load(filename_real);
h_imag = load(filename_imag);
x = h_real(:,1);

temp = h_real(:,2) + 1i*h_imag(:,2);
df = temp;
dt = ifft(temp);
