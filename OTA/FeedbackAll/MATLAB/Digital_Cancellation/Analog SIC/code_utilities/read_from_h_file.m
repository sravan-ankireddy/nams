
function dout = read_from_h_file(params)

filename  = params.filename;
iscomplex = params.iscomplex; 
QF        = params.QF;
convert_to_frac     = params.convert_to_frac;

dtemp = load(filename);

if iscomplex == 1
    dout = zeros(1,length(dtemp)/2);
    dout = dtemp(1:2:end) + i*dtemp(2:2:end);
else
    dout = dtemp;    
end

if convert_to_frac == 1 % convert to fractional 
    dout = dout*2^-QF; 
end

dout = dout.';
    