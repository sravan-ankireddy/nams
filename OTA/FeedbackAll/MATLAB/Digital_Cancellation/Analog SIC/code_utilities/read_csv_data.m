function data_out = read_csv_data(python_file,rd_filename,col_name1,col_name2)

if nargin == 3
    cmd = sprintf('python %s %s %s',python_file,rd_filename,col_name1);
else
    cmd = sprintf('python %s %s %s %s',python_file,rd_filename,col_name1,col_name2);   
end
system(cmd);

fid = fopen('temp.txt', 'r');
data = fscanf(fid,'%d');

if nargin == 3
  data_out = data;
else
  data_out = data(1:2:end)+i*data(2:2:end);
end

fclose(fid);
