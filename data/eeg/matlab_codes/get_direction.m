function v = get_direction(a,b)
% Calculate direction between two point
% INPUTS:
%     - a,b        : points in 3D; size 1*3
% OUTPUTS:
%     - v          : direction between two points; size 1*3
v = b-a;
v = v./sqrt(sum(v.^2,2)); % change from original code (see lign bellow) 
%v = v./mynorm(v,2);
end