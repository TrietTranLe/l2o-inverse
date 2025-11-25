function scaled_nmm = rescale_nmm_channel(nmm, region_id, spike_time, alpha_value)
% Re-scaling NMM channels in source channels
%
% INPUTS:
%     - nmm        : NMM data with single activation, time * channel
%     - spike_time : spike peak time
%     - region_id  : source regions, start at 1
%     - alpha_value: scaling factor
% OUTPUTS:
%     - scaled_nmm : scaled NMM in the source region; time * channel

    nmm_rm = nmm - mean(nmm, 1);
    for i=1:length(spike_time)
        sig = nmm_rm(spike_time(i)-249:spike_time(i)+250, region_id);      % one second data around the peak

        thre = 0.1;
        small_ind = find(abs(sig)<thre);                                   % the index that the values are close to 0
        small_ind((small_ind>450) | (small_ind < 50)) = [];
        start_ind = find((small_ind-250)<0);                               % spike start time
        % test 1
        while isempty(start_ind)
            thre = thre+0.05;
            small_ind = find(abs(sig)<thre);
            small_ind((small_ind>450) | (small_ind < 50)) = [];
            start_ind = find((small_ind-250)<0);
        end
        start_sig = small_ind(start_ind(end));

        % test 2
        [~, min_ind] = min(sig(301:400));
        min_ind = min_ind + 301;
        end_ind = find((small_ind-min_ind)>0);
        while isempty(end_ind)
            thre = thre+0.05;
            small_ind = find(abs(sig)<thre);
            small_ind((small_ind>450) | (small_ind < 50)) = [];
            end_ind = find((small_ind-min_ind)>0);
        end
        end_sig = small_ind(end_ind(1));                                   % spike end time

        sig(start_sig:end_sig) = sig(start_sig:end_sig) * alpha_value;     % scale the signal
        nmm_rm(spike_time(i)-249:spike_time(i)+250, region_id) = sig;
    end
    scaled_nmm = nmm_rm + mean(nmm, 1);


end