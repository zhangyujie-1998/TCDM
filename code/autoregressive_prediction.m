function [pc_r_predict_color, residual_color, pc_r_predict_coor, residual_coor] = autoregressive_prediction(pc_r, pc_d, center_coor, K)

trans_r = repmat(center_coor,size(pc_r,1),1);
trans_d = repmat(center_coor,size(pc_d,1),1);
pc_r_coor = pc_r(:,1:3) - trans_r;
pc_d_coor = pc_d(:,1:3) - trans_d;
pc_r_color = single(pc_r(:,4:6));
pc_d_color = single(pc_d(:,4:6));

[idx_r, dist_r] = knnsearch(pc_d_coor, pc_r_coor, 'K', K, 'Distance', 'euclidean');
pc_count = size(pc_r,1);

nei_r_idx = idx_r(:,2:end)';
nei_r_dist = dist_r(:,2:end);

sigma = mean(dist_r.^2, 2)+0.001;
sigma_rep = repmat(sigma,1,K-1);
weight_r = 1./(1+exp(-nei_r_dist./sigma_rep));
weight_r_sum  = sum(weight_r,2);
weight_r_rep = repmat(weight_r_sum,1,K-1);
weight_r_norm = weight_r./weight_r_rep;
weight_r_kron = kron(weight_r_norm, [1,1,1]);

nei_color  = pc_d_color(nei_r_idx(:),:);
nei_color_T = nei_color';
nei_color_v = nei_color_T(:);
nei_color_reshape = (reshape(nei_color_v, (K-1)*3, []))';

nei_coor  = pc_d_coor(nei_r_idx(:),:);
nei_coor_T = nei_coor';
nei_coor_v = nei_coor_T(:);
nei_coor_reshape = (reshape(nei_coor_v, (K-1)*3, []))';

A_color = nei_color_reshape .* weight_r_kron;
b_color = pc_r_color;
A_coor = nei_coor_reshape .* weight_r_kron;
b_coor = pc_r_coor;

[~, pc_r_predict_color, residual_color] = var_model(A_color, b_color);
[~, pc_r_predict_coor, residual_coor] = var_model(A_coor, b_coor);



