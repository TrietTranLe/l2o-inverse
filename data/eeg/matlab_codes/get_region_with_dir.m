function [selected_rg] = get_region_with_dir(v, region_centre, nb_points, ratio, bias)
% Select region given the region growing direction
% INPUTS:
%     - v             : region growing direction
%     - region_centre : centre region in 3D 
%     - nb_points     : neighbour region in 3D 
%     - ratio, bias   : adjust the probability of selecting neighbour
%                       regions (numbers decided by trial and error)
% OUTPUTS:
%     - selected_rg   : selected neighbouring regions

    v2 = get_direction(region_centre, nb_points);                          % direction between center region and neighbour regions
    dir_range = abs(v2*v');                                                % dot product between region growing direction and all neighbouring directions
    dir_range = ratio*((dir_range-min(dir_range))/(max(dir_range) - min(dir_range))) + bias;  % the probability of selecting neighbour regions
%     dir_range = 0.5                                                      % Equal probability for all directions
    selected_rg = rand(length(dir_range),1) < dir_range;
end