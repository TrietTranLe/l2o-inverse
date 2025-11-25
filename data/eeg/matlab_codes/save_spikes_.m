function save_spikes_(spike_data, savefile_path, previous_iter_spike_num)
% Save the spike data into seperate files
% INPUTS: spike_data: time * num_spikes * channel; extracted spike data
%         savefile_path: string
    for iii = 1:size(spike_data,2)
        % The raw data
        data = squeeze(spike_data(:,iii,:));
        save(strcat(savefile_path, int2str(iii+previous_iter_spike_num), '.mat'), 'data', '-v7')
    end
end  