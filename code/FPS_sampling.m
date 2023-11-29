function seed = FPS_sampling(pc, L)
pc_coor=pc.Location;    
count = pc.Count;

idx = randperm(count,1);
seed = pc_coor(idx,:);

pc_coor(idx,:) = [];       
for i=1:L-1   
    [~,dist] = knnsearch(seed,pc_coor,'K',1);
    [~,maxIndex] = max(dist);
    seed = [seed;pc_coor(maxIndex,:)];
    pc_coor(maxIndex,:) = [];
end


