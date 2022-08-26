% convert complex into I,Q format

function write_to_h_file_complex(din,filename)

dinIQ = zeros(1,2*length(din));
dinIQ(1:2:end) = real(din);
dinIQ(2:2:end) = imag(din);


dinIQ = int32(dinIQ);
fid = fopen(filename,'w');
for idx = 1 : length(dinIQ)-1
   fprintf(fid,'%d,\n',dinIQ(idx));
end
fprintf(fid,'%d',dinIQ(end));
fclose(fid);