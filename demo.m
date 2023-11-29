%% Date 2023.11.19
%% Author: Yujie Zhang (Any question, please contact: yujie19981026@sjtu.edu.cn)
%% Affiliation: Shanghai Jiao Tong University
%% If you find our code is useful, please cite our paper: TCDM: Transformational Complexity Based Distortion Metric for Perceptual Point Cloud Quality Assessment
%% ***************************************************************************************************************
clear

addpath('./code/');
addpath('./data/');

%% reference point cloud
pc_r = pcread("reference.ply");

%% distorted point cloud
pc_d = pcread("distorted.ply");

%% Parameter setting
param.L = 400;
param.K = 20;
param.T = 0.000001;
param.alpha = 0.3;
param.sampling = 'FastFPS';

%% Running
TCDM_score = TCDM(pc_r, pc_d, param);
fprintf('TCDM score is: %d\n',TCDM_score);
