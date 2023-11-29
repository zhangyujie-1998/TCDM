function sim = calculate_sim_prediction(pc_r_predict_coor, pc_r_predict_color, pc_d_predict_coor, pc_d_predict_color, K, T)
[indx,dist_r] = knnsearch(pc_r_predict_coor, pc_r_predict_coor, 'K', K , 'Distance','euclidean');
indx = indx(:,2:end);
indx_T = indx';
nei_color_r = pc_r_predict_color(indx_T(:),:);
center_color_r = reshape(repmat(pc_r_predict_color(:)',K-1, 1),[],3);
color_diff_r = ((abs(nei_color_r - center_color_r)*[1;2;1])+1);

color_diff_r_reshape = reshape(color_diff_r, K-1, [])';

energy_diff_r = color_diff_r_reshape .* dist_r(:,2:end);

nei_color_d = pc_d_predict_color(indx_T(:),:);
center_color_d =  reshape(repmat(pc_d_predict_color(:)',K-1, 1),[],3);
color_diff_d = ((abs(nei_color_d - center_color_d)*[1;2;1])+1);
color_diff_d_reshape = reshape(color_diff_d, K-1, [])';

pc_d_predict_x = pc_d_predict_coor(:,1);
pc_d_predict_y = pc_d_predict_coor(:,2);
pc_d_predict_z = pc_d_predict_coor(:,3);

nei_x_d = pc_d_predict_x(indx);
nei_y_d = pc_d_predict_y(indx);
nei_z_d = pc_d_predict_z(indx);
x_diff_d = nei_x_d - repmat(pc_d_predict_x, 1, K-1);
y_diff_d = nei_y_d - repmat(pc_d_predict_y, 1, K-1);
z_diff_d = nei_z_d - repmat(pc_d_predict_z, 1, K-1);

dist_d = sqrt(x_diff_d.^2 + y_diff_d.^2 + z_diff_d.^2);
energy_diff_d = color_diff_d_reshape.* dist_d;
sim = similarity_cov(energy_diff_r,energy_diff_d, T);
