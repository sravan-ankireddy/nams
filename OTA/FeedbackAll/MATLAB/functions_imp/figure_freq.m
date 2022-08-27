function figure_freq(X,Y,Z,W)
figure;
hold on;
title("OFDM Cancellation");
ylabel("Magnitude in dB")
xlabel("Subcarrriers")
plot(X);
plot(Y);
plot(Z);
plot(W);
legend("Received","Cancelled L","Cancelled NL")
end