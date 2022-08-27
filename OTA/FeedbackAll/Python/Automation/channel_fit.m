function channel_fit(dir)

    clc;
    close all;

    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));
    run('Parameters_feedback.m');

    % Channel
    H = open('Channel_Files/Channel_Output.mat');
    H = H.Channel;

    H = reshape(H, total_carriers, []);
    Channel = zeros(no_of_subcarriers, 2);

    k = 1;
    l = 1;
    % h1 = figure;
    % h2 = figure;
    for i = subcarrier_locations

        if ~(any(pilot_carriers(:) == i))
            C = abs(H(l, :)).';
            %         figure(h1)
            %         scatter(C,i*ones(length(C),1));
            %         hold on;
            %         grid on;

            pd = fitdist(C, 'Rician');
            %               HH = random(pd, 10000, 1);
            %               figure;
            %               hold on;
            %               histogram(HH,50,'Normalization', 'pdf');
            %               histogram(C,50,'Normalization', 'pdf');

            %         figure(h2)
            %         scatter(HH,i*ones(length(HH),1));
            %         hold on;
            %         grid on;
            %
            mean = pd.s;
            sig = pd.sigma;
            Channel(k, 1) = mean^2 / sig^2;
            Channel(k, 2) = sqrt(mean^2 + sig^2);
            k = k + 1;
        end

        l = l + 1;
    end

    % Noise
    N = open('Channel_Files/Noise_Output.mat');
    N = N.Noise;
    N = real(N.');
    Noise = zeros(no_of_subcarriers, 2);

    for i = 1:no_of_subcarriers
        C = real(N(i, :)).';
        pd = fitdist(C, 'Normal');
        Noise(i, 1) = pd.mean;
        Noise(i, 2) = (pd.sigma)^2;
    end

    save(strcat('Channel_Files/', dir, '_Channel.mat'), 'Channel');
    save(strcat('Channel_Files/', dir, '_Noise.mat'), 'Noise');
end
