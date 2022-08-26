function C = ActiveEncoder(B, Z1, Z2, I)

    if I == 1
        C = 1 - 2 * B;
    elseif I == 2
        C = -(1.457 + 0.9832 * Z1) .* (B - 1) - (1.464 - 0.9905 * Z1) .* B;
    elseif I == 3
        C = -(0.1762 + 0.1 * Z1 + 2.526 * max(0, 0.2328 * Z1 + 0.9178 * Z2 + 1.044)) .* (B - 1) + (0.1762 + 0.1 * Z1 + 2.526 * min(0, -0.2328 * Z1 + 0.9178 * Z2 - 1.044)) .* B;
    end
    C = (C - mean(C))/std(C); 

end
