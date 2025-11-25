function [add_rg, rm_rg] = smooth_region(nbs, current_regions)
% Clean up the current selected regions; since we randomly select the
% neighbouring regions, there could be "holes" in the source patch. We add
% the regions where all its neighbours are in the current source patch; and
% remove the regions where no neighbours is in current source patch;
% INPUTS:
%     - nbs             : neighbour regions for each cortical region; 1*994
%     - current_regions : selected regions 
% OUTPUTS:
%     - selected_rg   : selected neighbouring regions
    add_rg = [];
    rm_rg = [];
    all_final_nb = find_nb_rg(nbs, current_regions, []); 
    all_final_nb = setdiff(all_final_nb, current_regions);
    for i=1:length(all_final_nb)
        current_rg = all_final_nb(i);
        if length(intersect(current_regions, nbs{current_rg})) > length(nbs{current_rg})-2
            add_rg = [add_rg current_rg];
        end
    end
    for i=1:length(current_regions)
        current_rg = current_regions(i);
        if length(intersect([current_regions,add_rg], nbs{current_rg})) == 1
            rm_rg = [rm_rg current_rg];
        end
    end
end