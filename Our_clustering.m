function [Y_New, S, Obj_term, C] = Our_clustering( X, c, A, lambda_1, lambda_2, perr, ITER_MAX)
% X: every column denotes a sample
% A: side information
%

[dim, n] = size(X);

Y = orth(rand(n, c));
d = rand(n, c);
C = zeros(dim, c);
%%parameter
sigma=1;%0.1
 gamma=1;%0.1

interval = 1;
count = 0;
Obj = 1e+4;
% o = [];
Obj_term = [];
disp('start iteration')

 iter=1;
%  while(abs(interval)>perr)
  while(iter<ITER_MAX)
%     if count>0
%         o = [o;Y];
%     end
    
    %%update C
    for j = 1:c
        d_1 = 0;
        d_2 = 0;
        for i = 1:n
            d_1 = d(i,j)*Y(i,j)*X(:,i)+d_1;
            d_2 = d(i,j)*Y(i,j)+d_2;
        end
        C(:,j) = d_1/d_2;
    end
    
    %%update B and F
    for i = 1:n
        for j = 1:n
            B(i,j) = norm((Y(i,:) - Y(j,:)), 2)^2;
        end
    end
 
    F = A - lambda_1*B/(2*lambda_2);
    
    %%update S and L_S
    for i = 1:n
        [S(:,i), ft] = EProjSimplex_new(F(:,i), 1);%k=1
    end
    
    P_S = spdiags(sum(S,2),0,n,n);
    L_S = P_S - (S + S')/2;
    L_S = (L_S + L_S')/2;
    
    %%update d_ij and E and Term_1
    Term_1 = 0;
    for i = 1:n
        for j = 1:c
            diff = norm(X(:,i)-C(:,j),2);
            s_1 = (diff + 2*sigma)*(1+sigma);
            s_2 = 2*(diff + sigma)^2;
            d(i,j) = s_1/s_2;
            E(i,j) = d(i,j) * (diff^2);
            Term_1 = d(i,j)*Y(i,j)*(diff^2)+Term_1;
        end
    end
    
    %%update Y
   Y = Y*diag(sqrt(1./(diag(Y'*Y)+eps)));
        Y= Y.*(2*gamma*Y + eps)./(E + 4*lambda_1*L_S*Y + 2*gamma*Y*Y'*Y + eps);
     Y = Y*diag(sqrt(1./(diag(Y'*Y)+eps)));
%     B_Y = -E/2;
%     Y = GPI(2*lambda_1*L_S,B_Y);
    
    
    
    interval = norm(Obj-Y, 'fro');
    Obj = Y;
    count = count+1;
    disp(['循环次数是：', num2str(count)])
    
    %%objective value   
    Term_2 = 2*lambda_1*trace(Y'*L_S*Y);
    Term_3 = lambda_2*(norm(S-A, 'fro'))^2;
    Term = Term_1 + Term_2 + Term_3;
    Obj_term = [Obj_term; Term];

    iter = iter +1;
  end

Y_New = kmeans(Y, c);