function frame_capture()
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));
    run('Parameters_feedback.m');

    frame_capture = zeros(no_of_frames, 1);
    save('frame_capture.mat', 'frame_capture');
end
