% normalize to +- 1*scale and convert to signed fixed point format 
% QI: number of integer bits
% QF: number of fractional bits

function write_to_h_file(params,din)

filename = params.filename;
flag_int = params.convert_to_int;

if flag_int == 1    
    scale    = params.scale;
    QI       = params.QI;
    QF       = params.QF;
    
    dwr = normalize(din)*scale;
    dwr = q_format(dwr,QI,QF)*2^QF;
    dwr = int32(dwr);
    format_str1 = '%d,\n';
    format_str2 = '%d';

else % floating point 

    dwr = din;  
    format_str1 = '%10.16f,\n';
    format_str2 = '%10.16f';
end

if(isreal(dwr))
    dinIQ = dwr; 
else
    dinIQ = zeros(1,2*length(dwr));
    dinIQ(1:2:end) = real(dwr);
    dinIQ(2:2:end) = imag(dwr);
end

fid = fopen(filename,'w');
for idx = 1 : length(dinIQ)-1
   fprintf(fid,format_str1,dinIQ(idx));
end
fprintf(fid,format_str2,dinIQ(end));
fclose(fid);
