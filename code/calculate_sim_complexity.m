function sim = calculate_sim_complexity(pc_r_residual,pc_d_residual, T)
N_l = size(pc_r_residual,1);
cov_r = (pc_r_residual'*pc_r_residual/N_l);
complexity_self = det(cov_r);
cov_d = (pc_d_residual'*pc_d_residual/N_l);
complexity_trans = det(cov_d);
sim = similarity(complexity_self, complexity_trans, T);
      