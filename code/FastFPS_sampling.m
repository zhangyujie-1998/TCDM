function seed = FastFPS_sampling(pc, L)
pc_coor = pc.Location;
count = pc.Count;
indx = datasample(1:count, 10000, 'Replace', false);

pc_coor_rs = pc_coor(indx,:);
idx = randperm(10000,1);
seed = pc_coor_rs(idx,:);
pc_coor_rs(idx,:) = [];       
for i=1:L-1   
    [~,dist] = knnsearch(seed,pc_coor_rs,'K',1);
    [~,maxIndex] = max(dist);
    seed = [seed;pc_coor_rs(maxIndex,:)];
    pc_coor_rs(maxIndex,:) = [];
end


