function [taps x]= read_VNA(filename)

h_struct = load(filename);

x = h_struct(:,1);


N_taps = size(h_struct,2)-2;
taps = cell(1,N_taps);
offset = 2;
for idx = 1:N_taps
    taps{idx} = h_struct(:,idx+offset);
end

