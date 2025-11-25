% % based on the generate_sythetic_data.m script from deepSIF article
% with small modification so that everything works...
% july 2023

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
% !!! CAREFUL in the find_alpha_load function for now there is a
% "shitty_results" list which gathers signals for which spikes were not
% detected during my first tests. To remove in the future / add a warning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd ~/Documents/deepsif/DeepSIF-main/forward/

clear; close all;

home_dir = char(py.os.path.expanduser('~') );


dataset_name = '_dl_nmm_spikes_nov23_train_huge_';
clip_folder = "nmm_spikes_nov23_train"; % which spike data to use
clip_folder = strcat(home_dir, '/Documents/Data/simulation/constrained/standard_1020/fsav_994/simu/', clip_folder);


train = 1;
n_sources = 2;

% load anatomy data
load('../anatomy/fs_cortex_20k_inflated.mat')
load('../anatomy/fs_cortex_20k.mat')
load('../anatomy/fs_cortex_20k_region_mapping.mat');

% when load mat in python, python cannot read nan properly, so use a magic number to represent nan when saving
NAN_NUMBER = 15213; 
MAX_SIZE = 70;
if train
    nper = 1; % Number of nmm spike samples 
    %n_data = 40; % useless?
    n_iter = 50; % The number of variations in each source center
    ds_type = 'train';
else
    nper = 10;
    n_data = 1;
    n_iter = 3;
    ds_type = 'test';
end

%% ========================================================================
%=============== Generate Source Patch ====================================
%% ======== Region Growing Get Candidate Source Regions ===================
selected_region_all = cell(994, 1);                                          
for i=1:994
    % get source direction
    selected_region_all{i} = [];
    region_id =i;
    all_nb = cell(1,4);
    all_nb{1} = find_nb_rg(nbs, region_id, region_id);                     % first layer regions
    all_nb{2} = find_nb_rg(nbs, all_nb{1}, [region_id, all_nb{1}]);        % second layer regions
    v0 = get_direction(centre(region_id, :),centre(all_nb{1}, :));         % direction between the center region and first layer neighbors
    angs = zeros(size(v0,1),1);
    for k=1:size(v0,1)                                                     
        CosTheta = max(min(dot(v0(1,:),v0(k,:)),1),-1);
        angs(k) = real(acosd(CosTheta));
    end
    [~,ind] = sort(angs);
    ind = ind(1:ceil(length(angs)/2));                                     % directions to grow the region
    % second layer neighbours
    for iter = 1:5
    all_rg = cell(1,4);
    for k=1:length(ind)
        ii = ind(k); 
        all_rg(1:2) = all_nb(1:2); 
        v = get_direction(centre(region_id, :),centre(all_rg{1}(ii), :));
        all_rg{2} = all_rg{2}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{2}, :),1,0));
        [add_rg, rm_rg] = smooth_region(nbs, [region_id,all_nb{1},all_rg{2}]);
        final_r = setdiff([region_id, all_nb{1}, all_rg{2} add_rg],rm_rg, 'stable')-1;
        selected_region_all{i} = [selected_region_all{i};final_r NAN_NUMBER*ones(1,MAX_SIZE-length(final_r))];
    end
    end
    % third layer neighbours
    for iter = 1:5
    all_rg = cell(1,4);
    for k=1:length(ind)
        ii = ind(k); 
        all_rg(1:2) = all_nb(1:2); 
        v = get_direction(centre(region_id, :),centre(all_rg{1}(ii), :));
        all_rg{2} = all_rg{2}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{2}, :),1,0.1));
        all_rg{3} = find_nb_rg(nbs, all_rg{2}, [region_id, all_rg{1}, all_rg{2}]);  
        all_rg{3} = all_rg{3}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{3}, :),1,-0.15));
        [add_rg, rm_rg] = smooth_region(nbs, [region_id,all_nb{1},all_rg{2},all_rg{3}]);
        final_r = setdiff([region_id, all_nb{1}, all_rg{2} all_rg{3} add_rg],rm_rg, 'stable')-1;
        selected_region_all{i} = [selected_region_all{i};final_r NAN_NUMBER*ones(1,MAX_SIZE-length(final_r))];
    end
    end
%     % fourth neighbours
%     for iter = 1:5
%     all_rg = cell(1,4);
%     for k=1:length(ind)
%         ii = ind(k); 
%         all_rg(1:2) = all_nb(1:2); 
%         v = get_direction(centre(region_id, :),centre(all_rg{1}(ii), :));
%         all_rg{2} = all_rg{2}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{2}, :),1,0.2));
%         all_rg{3} = find_nb_rg(nbs, all_rg{2}, [region_id, all_rg{1}, all_rg{2}]);  
%         all_rg{3} = all_rg{3}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{3}, :),1,-0.1));
%         all_rg{4} = find_nb_rg(nbs, all_rg{3}, [region_id, all_rg{1}, all_rg{2},  all_rg{3}]); 
%         all_rg{4} = all_rg{4}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{4}, :),1,-0.35));
%         [add_rg, rm_rg] = smooth_region(nbs, [region_id,all_nb{1},all_nb{2},all_rg{3},all_rg{4}]);
%         final_r = setdiff([region_id, all_nb{1}, all_nb{2}, all_rg{3},all_rg{4}, add_rg],rm_rg, 'stable')-1;
%         if length(final_r) < 71
%             selected_region_all{i} = [selected_region_all{i};final_r NAN_NUMBER*ones(1,MAX_SIZE-length(final_r))];
%         end
%     end
%     end
end

%% ======== Get Region Center for Each Sample =============================
selected_region = NAN_NUMBER*ones(994*n_iter, n_sources, MAX_SIZE); 
n_iter_list = nan(n_iter*(n_sources-1), 994);
for i = 1:n_iter
    for k=1:(n_sources-1)
        n_iter_list(i+(k-1)*n_iter,:) = randperm(994);
    end
end
n_iter_list(n_iter+1,:) = 1:994; % pas comprendre

%% ======== Build Source Patch ============================================
for kk = 1:n_iter 
    for ii  = 1:994
        idx = 994*(kk-1) + ii; % indice de l'échantillon courant
        tr = selected_region_all{ii};
        if kk <= size(tr, 1) && train % on s'assure qu'on prend toutes les selected regions, si on génère plus d'example par 
            % région qu'il n'y a de selected regions, on complète en
            % sélectionnant aléatoirement
            selected_region(idx,1,:) = tr(kk,:);
        else
            selected_region(idx,1,:) = tr(randi([1,size(tr,1)],1,1),:);
        end
        for k=2:n_sources % choix d'une deuxième source active (le comment 
            % je ne comprends pas hyper bien)
            tr = selected_region_all{n_iter_list(kk+n_iter*(k-2),ii)};
            selected_region(idx,k,:) = tr(randi([1,size(tr,1)],1,1),:);
        end
    end
end
selected_region_raw = selected_region;
selected_region = reshape(permute(selected_region_raw, [3,2,1]), MAX_SIZE*n_sources, 994, n_iter);
selected_region = permute(selected_region,[1,3,2]);
selected_region = reshape(repmat(selected_region, 4, 1, 1), MAX_SIZE, n_sources, []);  % 4 SNR levels
selected_region = permute(selected_region,[3,2,1]);

%% SAVE
save([ds_type '_sample_' dataset_name '.mat'], 'selected_region')


%% ========================================================================
%=============== Generate Other Parameters=================================
%% NMM Signal Waveform
random_samples = randi([0,nper-1], 994*n_iter*4, n_sources); %randi(imax, s1,...,sn)  % the waveform index for each source
nmm_idx = (selected_region(:,:,1)+1)*nper + random_samples + 1; % ça vraiment je ne comprends pas du tout.
save([ds_type '_sample_' dataset_name '.mat'],'nmm_idx', 'random_samples',  '-append')

%% SNR
current_snr = reshape(repmat(5:5:20,n_iter*994,1)',[],1); 
save([ds_type '_sample_' dataset_name '.mat'],'current_snr', '-append')
%% Scaling Factor
%load('../anatomy/leadfield_75_20k.mat');
load(['/home/reynaudsarah/Documents/deepsif/DeepSIF-main/anatomy/LF_fsav_994.mat']);
fwd = G;
gt = load([ds_type '_sample_' dataset_name '.mat']);
scale_ratio = [];
n_source = size(gt.selected_region, 2);
parfor i=1:size(gt.selected_region, 1)
    for k=1:n_source
        a = gt.selected_region(i,k,:);
        a = a(:);
        a(a>1000) = [];
        if train
            scale_ratio(i,k,:) = find_alpha_load(a+1, random_samples(i, k), fwd, 10:2:20, clip_folder);
            
        else
            scale_ratio(i,k,:) = find_alpha_load(a+1, random_samples(i, k), fwd, [10,15], clip_folder);
        end
    end
end
save([ds_type '_sample_' dataset_name '.mat'], 'scale_ratio', '-append')

%% Change Source Magnitude 
clear mag_change
all_dis = load('../anatomy/dis_matrix_fs_20k.mat').raw_dis_matrix; %ajouté à la main
point_05 = [40, 60];  % 45,35        % Magnitude falls to half of the centre region
point_05 = randi(point_05);
sigma = 0.8493*point_05;
mag_change = [];
parfor i=1:size(gt.selected_region,1)
    for k=1:n_sources
        rg = gt.selected_region(i,k,:);
        rg(rg>1000) = [];
        dis2centre = all_dis(rg(1)+1,rg+1);
        mag_change(i,k,:) = [exp(-dis2centre.^2/(2*sigma^2)) NAN_NUMBER*ones(1,size(gt.selected_region,3)-length(rg))];
    end
end
save([ds_type '_sample_' dataset_name '.mat'], 'mag_change', '-append')









