function figure_time(X,Y)
figure;
hold on;
title("Integrated Power Cancellation");
ylabel("Cancellation in dB")
xlabel("Samples")
plot(X)
plot(Y)
legend("Cancelled L","Cancelled NL")
end