function update_plot(h,y) 

for idx = 1:length(h)
    set(h{idx},'YData',y{idx});
end

pause(0.01)