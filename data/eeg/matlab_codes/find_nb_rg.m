function rg = find_nb_rg(nbs, centre_rg, prev_layers)
% Find the neighbouring regions of the centre region
% INPUTS:
%     - nbs           : neighbour regions for each cortical region; 1*994
%     - centre_rg     : centre regions 
%     - prev_layers   : regions in inner layers 
% OUTPUTS:
%     - rg            : neighbouring regions
rg = unique(cell2mat(nbs(centre_rg)));
rg(ismember(rg, prev_layers)) = [];
end