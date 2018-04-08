clear
%%%%%经过归一化处理
load('YALE.mat');
X = fea';%X=d*n
[d, n] = size(X);

label = gnd;
c = length(unique(label)); % classes

k = 5;
perr = 1e-4;
lambda_1 = [0.001 0.01 0.1 1 10 100 1000];%1 and 2 0.1
lambda_2 = [0.001 0.01 0.1 1 10 100 1000];

percentage=0.4;
newlabel=label;
permutation=randperm(n);
index=permutation(1:floor((1-percentage)*n));
newlabel(index)=0;

pair = pair_gen(newlabel);
[Aw_1, Aw_2] = graph_construct_new(X, pair, k); %Aw_1--Gaussian; Aw_2--Side information

iterNum = 10;
rOUR_Gau_acc = zeros(7, 7); rOUR_Gau_nmi = zeros(7, 7);


%Our method
for i = 1:7
    for j = 1:7
        for iter = 1:iterNum
            clc
            [Y_New1, S1, Obj_term1,C1] = Our_clustering(X, c, Aw_1, lambda_1(i), lambda_2(j), perr, 150);
            result_1(iter,:) = ClusteringMeasure(label,Y_New1);
        end
        
        rOUR_Gau_acc(i,j)=mean(result_1(:,1)); rOUR_Gau_nmi(i,j)=mean(result_1(:,2));     
    end
end



save('iter10_YALE_our.mat', 'rOUR_Gau_acc', 'rOUR_Gau_nmi');

