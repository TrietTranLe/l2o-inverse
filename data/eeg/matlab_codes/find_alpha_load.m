function alpha = find_alpha_load(region_id, nmm_idx, fwd, target_SNR, clip_folder)
% Re-scaling NMM channels in source channels
%
% INPUTS:
%     - region_id  : source regions, start at 1
%     - nmm_idx    : load nmm data
%     - fwd        : leadfield matrix, num_electrode * num_regions
%     - target_SNR : set snr between signal and the background activity.
% OUTPUTS:
%     - alpha      : the scaling factor for one patch source

%%% solution temporaire pour palier au fait que pour certaines données je
%%% n'ai pas de spikes détectées
% % for the raw_nmm simulation (first simulation, july 23)
%     shitty_results = [...
%         387;62;903;914;937;910;7;674;355;419;...
%         210;913;15;938;441;420;559;417;934;414;...
%         325;916;954;411;921;356;640;358;546;915;...
%         909;949;390;917;418;603;373;434;628;413;...
%         935;920;922;500;908;409;421;23;415;912;993];
%   % for the raw_nmm_long simulation (august 23)
%     shitty_results = [...
%         7; 325; 356; 356; 387; 411; 415; 417; 418;...
%         418; 419; 910; 915; 915; 917; 917; 920; 921;...
%         922; 923; 923; 936; 938; 940; 949; 949; 993];
%     
%     shitty_results = [
%         387; 910; 7; 419; 936; 938; 417; 325;...
%         411; 921; 356; 923; 915; 949; 917; 418; 940;...
%         920; 922; 415; 993 ];
    
    shitty_results = [];
    good_results = setdiff( 0:993, shitty_results ); 
%     if any(shitty_results==region_id(1)-1)
%         dum = randsample(good_results, 1);
%         spike = load(strcat('../source/', clip_folder,'/a' ,int2str(dum), '/nmm_', int2str(nmm_idx+1), '.mat'));
%     else
%         spike = load(strcat('../source/', clip_folder,'/a' ,int2str(region_id(1)-1), '/nmm_', int2str(nmm_idx+1), '.mat'));
%     end
    if any(shitty_results==region_id(1)-1)
        dum = randsample(good_results, 1);
        spike = load(strcat( clip_folder,'/a' ,int2str(dum), '/nmm_', int2str(nmm_idx+1), '.mat'));
    else
        spike = load(strcat( clip_folder,'/a' ,int2str(region_id(1)-1), '/nmm_', int2str(nmm_idx+1), '.mat'));
    end

spike_shape = spike.data(:,region_id(1))/max(spike.data(:,region_id(1)));
[~, peak_time] = max(spike_shape);
spike.data(:, region_id) = repmat(spike_shape,1,length(region_id));
[Ps, Pn, ~] = calculate_SNR(spike.data, fwd, region_id, max(peak_time-50,1):min(peak_time+50,500));

alpha = sqrt(10.^(target_SNR./10).*Pn./Ps);
end