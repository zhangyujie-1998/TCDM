%% Date 2023.11.19
%% Author: Yujie Zhang (Any question, please contact: yujie19981026@sjtu.edu.cn)
%% Affiliation: Shanghai Jiao Tong University
%% If you use our code, please cite our paper: TCDM: Transformational Complexity Based Distortion Metric for Perceptual Point Cloud Quality Assessment
%% ***************************************************************************************************************
function results = TCDM(pt_r, pt_d, param)

if ~isfield(param, 'L'), param.L = 400; end
if ~isfield(param, 'K'), param.K = 20; end
if ~isfield(param, 'T'), param.k = 0.000001; end
if ~isfield(param, 'alpha'), param.alpha = 0.3; end
if ~isfield(param, 'sampling'), param.sampling = 'FPS'; end

if strcmp(param.sampling, 'RS') % random sampling
    seed= RS_sampling(pt_r, param.L);
elseif strcmp(param.sampling, 'FPS') % farther point sampling
    seed= FPS_sampling(pt_r, param.L);
elseif strcmp(param.sampling, 'FastFPS') % FPS after RS
    seed= FastFPS_sampling(pt_r, param.L);
end


pc_r_coor0 = pt_r.Location;
pc_d_coor0 = pt_d.Location;
pc_r_color0 = single(pt_r.Color);
pc_d_color0 = single(pt_d.Color);
pc_r0 = [pc_r_coor0, pc_r_color0];
pc_d0 = [pc_d_coor0, pc_d_color0];

[idx_cluster_r, ~] = knnsearch(seed, pc_r_coor0, 'K', 1, 'Distance','euclidean');
[idx_cluster_d, ~] = knnsearch(seed, pc_d_coor0, 'K', 1, 'Distance','euclidean');


K = param.K + 1;

for i = 1:param.L
    
   center_coor = seed(i,:);
   indx_r = find(idx_cluster_r == i);
   indx_d = find(idx_cluster_d == i);
  
   pc_r = pc_r0(indx_r,:);
   pc_d = pc_d0(indx_d,:);

   if (size(pc_d, 1) >= K) & (size(pc_r, 1) >= K)
       
       [pc_d_predict_color, pc_d_residual_color, pc_d_predict_coor, pc_d_residual_coor] = autoregressive_prediction(pc_r, pc_d, center_coor, K);
       [pc_r_predict_color, pc_r_residual_color, pc_r_predict_coor, pc_r_residual_coor] = autoregressive_prediction(pc_r, pc_r, center_coor, K);
    
       %% Complexity-based feature
       F1_O_local(i) = calculate_sim_complexity(pc_r_residual_coor, pc_d_residual_coor, param.T);
       F1_I_local(i) = calculate_sim_complexity(pc_r_residual_color, pc_d_residual_color, param.T);
      
       %% Prediction-based feature
       F2_local(i) = calculate_sim_prediction(pc_r_predict_coor, pc_r_predict_color, pc_d_predict_coor, pc_d_predict_color, K, param.T);
   
   else
       F1_O_local(i) = 0;
       F1_I_local(i) = 0;
       F2_local(i) = 0;

   end

end

F1 = mean(F1_O_local) * mean(F1_I_local);
F2 = mean(F2_local);
results = param.alpha*F1 + (1-param.alpha)*F2;


