% based on the process_raw_nmm function from deepSIF article
% with small modification so that everything works...
% july 2023

clear; close all; 

cd ~/Documents/deepsif/DeepSIF-main/forward/

%%% PARAMETERS %%%
n_regions = 994; % number of regions of the head model
tvb_data_folder = "raw_nmm_finland"; % which tvb data to use (folder)
filename = "spikes_finland"; % name of the folder in which to store the results
%savefile_path = '../source/';
savefile_path = '../source/';

%headmodel = load(['../anatomy/' 'leadfield_75_20k.mat']);
headmodel = load('/home/reynaudsarah/Documents/deepsif/DeepSIF-main/anatomy/LF_fsav_994.mat'); %leadfield from mne python
fwd = headmodel.G;

iter_list = 0:0;   % the iter during NMM generation (iter_m in the NMM generation)
iter_T = 0:5; % iter over 10sec time window simulated

plt = false; % create figs to visualize data
reg_reshape = false; %true; % reshape regions of nmm data (998 -> 994 regions) if it was not done already in the generate data script
%%
% create folder to save figs
%if ~ exist("./figs", 'dir')
%    mkdir('./figs');
%end
%if ~ exist( strcat( './figs/',filename ), 'dir' )
%    mkdir( strcat( './figs/',filename ) );
%end
%
%if ~ exist(strcat( './figs/',filename, '/original' ), 'dir')
%    mkdir( strcat( './figs/',filename, '/original' ) );
%end
%if ~exist( strcat( './figs/',filename, '/spikes' ), 'dir' )
%    mkdir( strcat( './figs/',filename, '/spikes' ) );
%end
%if ~ exist( strcat( './figs/',filename, '/rescaled_nmm' ), 'dir' )
%    mkdir( strcat( './figs/',filename, '/rescaled_nmm' ) );
%end
%% 
% -------------------------------------------------------------------------
previous_iter_spike_num = zeros(1, n_regions);
no_spikes=[];
for i_iter = 1:length(iter_list)
    iter = iter_list(i_iter);
    if isempty(dir(strcat(savefile_path, 'nmm_', filename, '/clip_info/iter', int2str(iter))))
       mkdir(strcat(savefile_path, 'nmm_', filename, '/clip_info/iter', int2str(iter)));
    end

    % ------- Resume running if the process was interupted ----------------
    done = dir(strcat(savefile_path, 'nmm_', filename, '/clip_info/iter', int2str(iter), '/iter_', int2str(iter), '_i_*'));
    finished_regions = zeros(1, length(done));
    for i = 1:length(done)
        finished_regions(i) = str2num(done(i).name(10:end-3));
    end
    remaining_regions = setdiff(1:n_regions, finished_regions+1);
    if isempty(remaining_regions)
        continue;
    end

    % -------- start the main progress -----------------------------------%
    for i = 1:n_regions %length(remaining_regions)
        %i = ii; %remaining_regions(ii);
        
        % creat folders to save nmm files
        if isempty(dir(strcat(savefile_path, 'nmm_', filename, '/a', int2str(i-1))))
            mkdir(strcat(savefile_path, 'nmm_', filename, '/a', int2str(i-1)))
        end
        
        for it = 1:length(iter_T)
            T = iter_T(it);
            fn = strcat(savefile_path, tvb_data_folder, '/a', int2str(i-1), '/mean_iter_', int2str(iter), '_a_iter_', int2str(i-1));
            raw_data = load(strcat(fn, '_',int2str(T), '.mat'));
            nmm = raw_data.data; %raw_data.all_data;
            %nmm = downsample(nmm, 4);        
            
%             if reg_reshape 
%                 % since i forgot to uncomment this part in the python code
%                 nmm(:, 7+1)     = nmm(:, 994+1);
%                 nmm(:, 325+1)   = nmm(:, 997+1);
%                 nmm(:, 921+1)   = nmm(:, 996+1);
%                 nmm(:, 949+1)   = nmm(:, 995+1);
%                 nmm = nmm(500:end, 1:994);
%             end
            nmm = nmm( 100:end , : );
        
            [spike_time, spike_chan] = find_spike_time(nmm);                   % Process raw tvb output to find the spike peak time
            
            if plt
                figure()
                for isp = 1:size(nmm,2)
                    plot(nmm(:,isp)); hold on 
                end
                if ~ isempty(spike_time)
                    for isp = 1:numel(spike_time)
                        xline(spike_time(isp)); hold on
                    end
                end
                hold off
                title(strcat('nmm data',int2str(i))); 
                saveas(gcf, strcat('./figs/', filename,...
                    '/original/original_', int2str(i), '_iterM_', int2str(iter),...
                    '_iterT_,', int2str(T), '.png'))
                close();
            end
        
            % ----------- select the spikes we want to extract ---------------%
            rule1 =  (spike_chan == i);                                        % there is spike in the source region
            start_time = floor(spike_time(rule1)/500) * 500 + 1;
            if ~ isempty(start_time) % there is no source in other region in the clip
                clear_ind = repmat(start_time, [900, 1]) + (-200:699)';            % 900 * num_spike
                rule2 = (sum(ismember(clear_ind, spike_time(~rule1)), 1) == 0);    % there are no other spikes in the clip
                spike_time = spike_time(rule1);
                spike_time = spike_time(rule2);
                if ~ isempty(spike_time)
                    % ----------- Optional :  Scale the NMM here----------------------%
                    alpha_value = find_alpha(nmm, fwd, i, spike_time, 15);
                    nmm = rescale_nmm_channel(nmm, i, spike_time, alpha_value);       
                    % ------------Save Spike NMM Data --------------------------------%
                    start_time = floor(spike_time/500) * 500 + 1;
                    spike_ind = repmat(start_time, [500, 1]) + (0:499)';       
            %         start_time = floor((spike_time+200)/500) * 500 + 1 - 200;        % start time can be changed
            %         start_time = max(start_time, 101);
            %         spike_ind = repmat(start_time, [500, 1]) + (0:499)';

                    nmm_data = reshape(nmm(spike_ind,:), 500, [], size(nmm,2));        % size: time * num_spike * channel
                    save_spikes_(nmm_data, strcat(savefile_path, 'nmm_', filename,'/a', int2str(i-1), '/nmm_'), previous_iter_spike_num(i));
                    previous_iter_spike_num(i) = previous_iter_spike_num(i) + length(spike_time);
                    % Save something in clip info, so that we can make sure we finish this process
                    save_struct = struct();
                    save_struct.num_spike = previous_iter_spike_num(i);
                    save_struct.spike_time = spike_time;
                    save(strcat(savefile_path, 'nmm_', filename, '/clip_info/iter', int2str(iter),...
                        '/iter_', int2str(iter), '_i_', int2str(i-1), '.mat'), 'save_struct')
                    
                    disp(strcat('iter_', int2str(iter), '_i_', int2str(i-1), 'is done\n') );

                    if plt
                        figure()
                        for is = 1:numel(spike_time)
                            plot(nmm_data(:,is,i)); hold on 
                        end
                        hold off
                        title(strcat('spikes for nmm data',int2str(i))); 
                        saveas(gcf,strcat('./figs/', filename,...
                            '/spikes/spikes_region_',int2str(i), '_iterM_', int2str(iter),...
                            '_iterT_,', int2str(T), '.png'))
                        close()

                        figure()
                        for is = 1:size(nmm,2)
                            plot(nmm(:,is)); hold on 
                        end
                        hold off
                        title(strcat('nmm data scaled',int2str(i))); 
                        saveas(gcf,strcat('./figs/', filename, '/rescaled_nmm/rescaled_region_',int2str(i), ...
                             '_iterM_', int2str(iter), '_iterT_,', int2str(T), '.png'))
                        close()
                    end

                
                end %end if
            else
                no_spikes = [no_spikes, i-1];
                disp(strcat('iter_', int2str(iter), '_i_', int2str(i-1),'has no spikes... ?\n'));
            end % end if
        end % end iter T
    end % END REGION
end % END ITER
    
  
% %% == helper function to save spikes == %% 
%function save_spikes_(spike_data, savefile_path, previous_iter_spike_num)
%% Save the spike data into seperate files
%% INPUTS: spike_data: time * num_spikes * channel; extracted spike data
%%         savefile_path: string
%    for iii = 1:size(spike_data,2)
%        % The raw data
%        data = squeeze(spike_data(:,iii,:));
%        save(strcat(savefile_path, int2str(iii+previous_iter_spike_num), '.mat'), 'data', '-v7')
%    end
%end  




