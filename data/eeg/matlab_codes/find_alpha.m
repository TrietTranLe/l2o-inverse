function [alpha] = find_alpha(nmm, fwd, region_id, time_spike, target_SNR)
% Find the scaling factor for the NMM channels.
%
% INPUTS:
%     - nmm        : NMM data with single activation, time * channel
%     - fwd        : leadfield matrix, num_electrode * num_regions
%     - region_id  : source regions, start at 1, region_id(1) is the center
%     - time_spike : spike peak time
%     - target_SNR : set snr between signal and the background activity.
% OUTPUTS:
%     - spike_time : the spike peak time in the (downsampled) NMM data
%     - spike_chan : the spike channel for each spike
%     - alpha      : the scaling factor for one patch source

    spike_ind = repmat(time_spike, [200, 1]) + (-99:100)';
    spike_ind = min(max(spike_ind(:),0), size(nmm,1));                     % make sure the index is not out of range
%     spike_ind = max(0, time_spike-100): max(time_spike+100,size(nmm,1));   % make sure the index is not out of range                     
    spike_shape = nmm(:,region_id(1)); %/max(nmm(:,region_id(1)));
    nmm(:, region_id) = repmat(spike_shape,1,length(region_id));
    % calculate the scaling factor
    [Ps, Pn, ~] = calcualate_SNR(nmm, fwd, region_id, spike_ind);
    alpha = sqrt(10^(target_SNR/10)*Pn/Ps);
end

function [Ps, Pn, cur_snr] = calcualate_SNR(nmm, fwd, region_id, spike_ind)
% Caculate SNR at sensor space.
%
% INPUTS:
%     - nmm        : NMM data with single activation, time * channel
%     - fwd        : leadfield matrix, num_electrode * num_regions
%     - region_id  : source regions, start at 1, region_id(1) is the center
%     - spike_ind  : index to calculate the spike snr
% OUTPUTS:
%     - Ps         : signal power
%     - Pn         : noise power
%     - cur_snr    : current SNR in dB

    sig_eeg = (fwd(:, region_id)*nmm(:, region_id)')';   % time * channel
    sig_eeg_rm = sig_eeg - mean(sig_eeg, 1);
    dd = 1:size(nmm,2);
    dd(region_id) = [];
    noise_eeg = (fwd(:,dd)*nmm(:,dd)')';
    noise_eeg_rm = noise_eeg - mean(noise_eeg, 1);

    Ps = norm(sig_eeg_rm(spike_ind,:),'fro')^2/length(spike_ind);
    Pn = norm(noise_eeg_rm(spike_ind,:),'fro')^2/length(spike_ind);
    cur_snr = 10*log10(Ps/Pn);
end