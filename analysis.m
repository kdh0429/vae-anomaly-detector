clc
clear all
%%
num_data_type = 3; % i-dyna, qdot, q_desired - q, qdot_desired - qdot, theta-q, ee_acc
num_time_step = 5;
num_input = num_time_step*(6*(num_data_type-1)); %num_time_step*(6*(num_data_type-1) + 3*1);
label_idx = num_input+1;
%%
TestingDataAnomalyCollision_tmp = load('data/TestingDataAnomalyCollision.csv');
TestingDataAnomalyCollision = TestingDataAnomalyCollision_tmp(2:size(TestingDataAnomalyCollision_tmp,1),:);
clear TestingDataAnomalyCollision_tmp

TestingDataAnomalyFree_tmp = load('data/TestingDataAnomalyFree.csv');
TestingDataAnomalyFree = TestingDataAnomalyFree_tmp(2:size(TestingDataAnomalyFree_tmp,1),:);
clear TestingDataAnomalyFree_tmp

trainreconloss = load('training_recon_loss.csv');

testcolreconloss = load('test_collision_recon_loss.csv');
testcolreconresult = load('test_collision_recon_result.csv');
testcolprediction = load('test_collision_prediction.csv');

testfreereconloss = load('test_free_recon_loss.csv');
testfreereconresult = load('test_free_recon_result.csv');
testfreeprediction = load('test_free_prediction.csv');

%%
num_collision = sum(TestingDataAnomalyCollision(:,label_idx));
num_free = size(TestingDataAnomalyCollision,1) - num_collision;
col_recon_loss = zeros(num_collision,1);
col_reconresult = zeros(num_collision,num_input);
free_recon_loss = testfreereconloss;
free_reconresult = TestingDataAnomalyFree(:,1:num_input) - testfreereconresult;

col_idx = 1;
for i=1:size(TestingDataAnomalyCollision,1)
    if TestingDataAnomalyCollision(i,label_idx) == 1
        col_reconresult(col_idx,:) = TestingDataAnomalyCollision(i,1:num_input) - testcolreconresult(i,1:num_input);
        col_recon_loss(col_idx) = testcolreconloss(i);
        col_idx = col_idx+1;
    end
end

%%
col_recon_tau = zeros(num_collision, num_input/2);
free_recon_tau = zeros(size(TestingDataAnomalyFree,1), num_input/2);

for i=1:2:num_input
    free_recon_tau(:,(i+1)/2) = free_reconresult(:,i);
end
for i=1:2:num_input
    col_recon_tau(:,(i+1)/2) = col_reconresult(:,i);
end

free_recon_tau_sum = sum(free_recon_tau,2);
col_recon_tau_sum = sum(col_recon_tau,2);

free_recon_tau_sum = sum(abs(free_recon_tau),2);
col_recon_tau_sum = sum(abs(col_recon_tau),2);


trainreconloss = free_recon_tau_sum;

col_recon_loss = col_recon_tau_sum;
free_recon_loss = free_recon_tau_sum;


testfreereconloss = free_recon_tau_sum;
col_recon_tau = zeros(size(TestingDataAnomalyCollision,1), num_input/2);
col_reconresult= TestingDataAnomalyCollision(:,1:num_input) - testcolreconresult(:,1:num_input);
for i=1:2:num_input
    col_recon_tau(:,(i+1)/2) = col_reconresult(:,i);
end
col_recon_tau_sum = sum(abs(col_recon_tau),2);

testcolreconloss = col_recon_tau_sum;
%%
threshold_percentile = 0.999;
max_k_idx = round(size(trainreconloss,1)*(1-threshold_percentile));
max_k_arr = maxk(trainreconloss,max_k_idx);
threshold = max_k_arr(max_k_idx)

testcolprediction_user = zeros(size(testcolprediction));
for i=1:size(testcolreconloss,1)
    if testcolreconloss(i) > threshold
        testcolprediction_user(i) = 1;
    end
end

testfreeprediction_user = zeros(size(testfreeprediction));
for i=1:size(testfreereconloss,1)
    if testfreereconloss(i) > threshold
        testfreeprediction_user(i) = 1;
    end
end

%%
subplot(1,3,1)
histogram(trainreconloss)
hold on
histogram(free_recon_loss)
histogram(col_recon_loss)

subplot(1,3,2)
plot(TestingDataAnomalyCollision(:,label_idx))
hold on
plot(testcolprediction_user)

subplot(1,3,3)
plot(TestingDataAnomalyFree(:,label_idx))
hold on
plot(testfreeprediction_user)
