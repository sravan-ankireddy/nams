function [ w ] = calculate_window( params )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    N   = params.NFFT;
    Ncp = params.NCP;
    Nrp = params.NRP;

    alpha = Nrp/(N + Ncp);
    w = zeros(1,N + Ncp + Nrp);

    phi = pi/(alpha*(N+Ncp));

    for i1 = 0:(N+Ncp-Nrp)/2-1
        w((N+Ncp+Nrp)/2+i1+1) = 1;
    end

    for i1 = (N+Ncp-Nrp)/2:(N+Ncp+Nrp)/2-1
        w((N+Ncp+Nrp)/2+i1+1) = 0.5*(1-sin(phi*(i1-(N+Ncp)/2+0.5)));
    end

    for i1 = 0:(N+Ncp+Nrp)/2-1
        w((N+Ncp+Nrp)/2-i1) = w((N+Ncp+Nrp)/2+i1+1);
    end
    
end

