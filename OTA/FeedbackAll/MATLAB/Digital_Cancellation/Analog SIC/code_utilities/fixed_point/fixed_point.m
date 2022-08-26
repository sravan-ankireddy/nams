% fixed point class
% assume all data in the same matrix have the same fixed point format

classdef fixed_point
    properties
        data
        QI
        QF
    end
    methods        
        function obj = fixed_point(x,QI,QF)
            if nargin ~= 0
                if(isa(x,'fixed_point'))
                    obj.QI = QI;
                    obj.QF = QF;
                    obj.data = quantize_r4(x.data,QI,QF,1,0); % flag_signed=1 ,flag_floor=0
                else
                    obj.QI = QI;
                    obj.QF = QF;
                    obj.data = quantize_r4(x,QI,QF,1,0); % flag_signed=1 ,flag_floor=0
                end                
            end
        end
        
        function y = plus(a,b)
            y      = fixed_point;
            y.QI   = max(a.QI,b.QI)+1;
            y.QF   = max(a.QF,b.QF);
            y.data = a.data + b.data;
        end
        
        function y = minus(a,b)
            y      = fixed_point;            
            y.QI   = max(a.QI,b.QI)+1;
            y.QF   = max(a.QF,b.QF);
            y.data = a.data - b.data;
        end
        
        function y = conj(a)
            y      = fixed_point;            
            y.QI   = a.QI;
            y.QF   = a.QF;
            y.data = conj(a.data);
        end
        
        function y = abs_square(a)
            y = fixed_point;            
            y = a.*conj(a);
        end
        
        function y = mtimes(a,b)
            y    = fixed_point;
            y.QI = a.QI + b.QI;
            y.QF = a.QF + b.QF;
            y.data = a.data * b.data;
        end
        
        function y = times(a,b)
            y    = fixed_point;            
            y.QI = a.QI + b.QI;
            y.QF = a.QF + b.QF;
            y.data = a.data.* b.data;
        end

        function y = transpose(a)
            y      = fixed_point;
            y.QI   = a.QI ;
            y.QF   = a.QF ;
            y.data = a.data.';
        end

        function y = ctranspose(a)
            y      = fixed_point;
            y.QI   = a.QI ;
            y.QF   = a.QF ;
            y.data = a.data';
        end
        
        % not sure about the best fixed point format 
        function y = rdivide(a,b)
            y      = fixed_point;
            y.QI   = a.QI ;
            y.QF   = a.QF - b.QF;
            y.data = a.data./b.data;
        end
        
        function y = power(a,N)
            y    = fixed_point;            
            if N == 0
                y.QI = a.QI;
                y.QF = a.QF;
            else
                y.QI = a.QI*N;
                y.QF = a.QF*N;
            end
            y.data = a.data.^N;
        end
        
        % change fixed point format
        function y = transform(a,QI,QF)
            y         = fixed_point;
            y.QI      = QI;
            y.QF      = QF;
            shift     = QF - a.QF ;
            mask      = 2^(QI + QF)-1; % create all ones mask
            data_int  = a.data*2^a.QF; % convert to integer 
            data_temp = floor(data_int.*2^shift);
            
               y.data = double( bitand(int64(real(data_temp)), mask) )*2^-QF + ...
                        1i*double( bitand(int64(imag(data_temp)), mask) )*2^-QF ; %convert back to decimal point representation                       
        end
        
        function y = to_integer(x)
            y = [x.data].*2.^[x.QF];
        end
        
        function y = to_digit(x)
            for n=1:length(x)
                integer = x(n).data * 2^x(n).QF;
                Num_bit = x(n).QI+x(n).QF;
                Num_level = 2^(Num_bit);
                y(n,:) = find_dig1(mod( real( integer ),Num_level),2,Num_bit) + i*find_dig1(mod( imag( integer ) ,Num_level),2,Num_bit);
            end
        end
        
    end
end