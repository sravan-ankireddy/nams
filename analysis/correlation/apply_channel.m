function received_codewords = apply_channel(codewords, sig, noise, channel)
    if (channel == "rayleigh_fast")
%         rng(round(sum(abs(noise))),'twister')
        fading_h = sqrt((randn(size(codewords)).^2 +  randn(size(codewords)).^2)/sqrt(3.14/2.0));
        received_codewords = fading_h.*codewords + noise;
    elseif (channel == "AWGN")
        received_codewords = codewords + noise;
    elseif (channel == "bursty")
        sig_bursty = 3*sig;
        noise_bursty = sig_bursty * rand(size(codewords));
        p = 0.1;
        bin = randn(size(codewords));
        bin(bin<=p) = 1;
        bin(bin~=1) = 0;
        noise_bursty = bin.*noise_bursty;
        received_codewords = codewords + noise + noise_bursty;
    end
end