function [spike_time, spike_chan] = find_spike_time(nmm)
% Process raw tvb output to find the spike peak time.
%
% INPUTS:
%     - nmm        : (Downsampled) raw tvb output, time * channel
% OUTPUTS:
%     - spike_time : the spike peak time in the (downsampled) NMM data
%     - spike_chan : the spike channel for each spike

    spikes_nmm = nmm;
    spikes_nmm(nmm < 8) = 0;                                               % find the spiking activity stronger than the background
    local_max = islocalmax(spikes_nmm);                                    % find the peak
    [spike_time, spike_chan] = find(local_max);
    [spike_time, sort_ind] = sort(spike_time);
    spike_chan = spike_chan(sort_ind);                                     % sort the activity based on time
    use_ind = (spike_time-249 > 0) & ...                                   % ignore the spikes at the beginning or end of the signal
        (spike_time+500 < size(nmm, 1) & ...
        [1 diff(spike_time)'>100]');                                       % ignore peaks close together for now (will have signals with close peaks in multi-source condition)
    spike_time = spike_time(use_ind)';
    spike_chan = spike_chan(use_ind)';

end