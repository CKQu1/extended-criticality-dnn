% calculate the statistical property of SGD walk
function SGD_analysis_step_level(varargin)
% read in data
d = dir('/project/THAP/Anomalous-diffusion-dynamics-of-SGD/trained_nets/resnet*');
steps_in_part = 1e3;
% Loop number for PBS array job
loop_num = 0;

for ii = 1:length(d)
    sub_loss_w_dir = dir(fullfile(d(ii).folder,d(ii).name,'model*sub_loss_w.mat'));
    % For PBS array job
    loop_num = loop_num + 1;
    if nargin ~= 0
        PBS_ARRAYID = varargin{1};
        if loop_num ~=  PBS_ARRAYID
            continue;
        end
    end
    part_num = 1;
    index = 1;
    for file = 1:length(sub_loss_w_dir)
        R = load(fullfile(d(ii).folder,d(ii).name,sprintf('model_%d_sub_loss_w.mat',file)));
        % organise the weights
        all_weights_tmp = cellfun(@(x) reshape(x,1,[]) ,R.sub_weights,'UniformOutput',false);
        all_sub_weights{index} = cell2mat(all_weights_tmp);
        all_sub_loss{index} = R.sub_loss;
        grads{index} = R.grads;
        estimated_full_batch_grad{index} = R.estimated_full_batch_grad;
        gradient_noise_norm{index} = R.gradient_noise_norm;
        % use trajectory length to split the data
        if index*numel(R.sub_loss) > steps_in_part || file == length(sub_loss_w_dir)
            MSD_feed = organize_data(all_sub_weights, all_sub_loss, d(ii), sub_loss_w_dir(1), part_num);
            clear all_sub_weights
            data_analysis(MSD_feed, all_sub_loss, d(ii), sub_loss_w_dir(1), part_num,grads,estimated_full_batch_grad,gradient_noise_norm)
            clear all_sub_loss all_test_sub_loss grads estimated_full_batch_grad gradient_noise_norm
            part_num = part_num + 1;
            index = 1; % start again
        end
        index = index + 1;
    end    
end
end


function MSD_feed = organize_data(all_sub_weights, all_sub_loss, d, GN_dir, part_num)
fprintf('Organizing part: %d\n',part_num)
% organize data to feed MSD calculation
empty_flag = cellfun(@isempty,all_sub_weights);
if sum(empty_flag) > 0 && sum(empty_flag) < 2
    all_sub_weights = all_sub_weights(~empty_flag);
    all_sub_loss = all_sub_loss(~empty_flag);
elseif sum(empty_flag) > 1
    warning('Data %d has empty component and it is not the first one',part_num);
    MSD_feed = nan;
    save(fullfile(GN_dir(1).folder,[d.name(1:end-24),'_data_part_',num2str(part_num),'.mat']),'MSD_feed','-v7.3')
    return
end
tiny_step_num = size(all_sub_loss{1},2);
t = 1:(tiny_step_num*length(all_sub_loss));
MSD_feed = [cell2mat(all_sub_weights'),t',reshape(cell2mat(all_sub_loss),[],1)];
save(fullfile(GN_dir(1).folder,[d.name(1:end-24),'_data_part_',num2str(part_num),'.mat']),'MSD_feed','-v7.3')
end

function data_analysis(MSD_feed, all_sub_loss, d, GN_dir, part_num,grads,estimated_full_batch_grad,gradient_noise_norm)
fprintf('Calculating MSD part: %d',part_num)
% calculate MSD
[MSD,Mean_contourlength,tau,MSD_forcontour,MSL_contourlength,contour_length,MSL_distance,distance,Displacement_all,Contour_length_all] = get_contour_lenth_MSD_loss(MSD_feed);
[MSD_noaverage,tau_noaverage] = get_no_average_MSD(MSD_feed(:,1:end-1));
clear MSD_feed
MSD = gather(MSD);
Mean_contourlength = gather(Mean_contourlength);
tau = gather(tau);
MSD_forcontour = gather(MSD_forcontour);
MSL_contourlength = gather(MSL_contourlength);
contour_length = gather(contour_length);
MSL_distance = gather(MSL_distance);
distance = gather(distance);
Displacement_all = gather(Displacement_all);
Contour_length_all = gather(Contour_length_all);
disp('MSD and contour: done!')

% loss dynamics
loss = cell2mat(all_sub_loss);
loss = reshape(loss',[],1);
% get the step difference of training loss
delta_train_loss = diff(loss);

% fit gradient distribution
F_grads = fitdist(double(grads(randperm(numel(grads),5000))),'stable');
F_full_batch_grad = fitdist(double(estimated_full_batch_grad(randperm(numel(estimated_full_batch_grad),5000))),'stable');
F_gradient_noise_norm = fitdist(double(gradient_noise_norm(randperm(numel(gradient_noise_norm),5000))),'stable');
    
% transversed excursion, randomly get 100 fragments
[end_end_dis2,transversed_excursion] = zeros(100,1);
fragment_length = 50;
for rd = 1:100
    fragment_start = randperm(size(MSD_feed,1) - fragment_length,1);
    fragment_end = fragment_start + fragment_length;
    temp_tran_excu = zeros(fragment_length,1);
    end_end_dis2(rd) = sum((MSD_feed(fragment_start,1:end-2) - MSD_feed(fragment_end,1:end-2)).^2);
    for jj = 1:fragment_length
        CA = MSD_feed(fragment_start + jj,1:end-2) - MSD_feed(fragment_start,1:end-2);
        BA = MSD_feed(fragment_end,1:end-2) - MSD_feed(fragment_start,1:end-2);
        % distance from point C to line AB
        temp_tran_excu(jj) = norm(CA-dot(CA,BA)/dot(BA,BA)*BA);
    end
    transversed_excursion(rd) = max(temp_tran_excu)^2;
end

%save
add_source = split(GN_dir(1).folder,'/');
add_source{3} = 'cortical';
add_new = join(add_source,'/');
save(fullfile(add_new{1},[d.name(1:end-24),'_data_part_',num2str(part_num),'22.mat']),'loss','delta_train_loss',...
    'MSD','Mean_contourlength','tau','MSD_forcontour','MSL_contourlength','contour_length','MSL_distance',...
    'distance','Displacement_all','Contour_length_all','MSD_noaverage','tau_noaverage',...
    'F_grads','F_full_batch_grad','F_gradient_noise_norm','end_end_dis2','transversed_excursion','-v7.3')
end

