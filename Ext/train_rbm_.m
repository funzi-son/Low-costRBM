function model = train_rbm_(conf,dat,vis)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training RBM                                                       %  
% conf: training setting                                             %
% -*-sontran2012-*-                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

assert(~isempty(dat),'[KBRBM] Data is empty'); 
%% initialization
sz      = size(dat,1);
visNum  = size(dat,2);
hidNum  = conf.hidNum;
if conf.bNum == 0
    bNum = ceil(sz/conf.sNum);
else
    bNum = conf.bNum;
end
sNum  = conf.sNum;
lr    = conf.params(1);
if exist('conf.N','var')    
    N     = conf.N;                                                                     % Number of epoch training with lr_1                     
else
    N = 1;
end

model.W = 0.01*randn(visNum,hidNum);
DW    = zeros(size(model.W));
model.visB  = zeros(1,visNum);
DVB   = zeros(1,visNum);
model.hidB  = zeros(1,hidNum);
DHB   = zeros(1,hidNum);


%% Reconstruction error & evaluation error & early stopping
mse    = 0;
%%%%%%%%%%
%% ==================== Start training =========================== %%
mse_plot = [];
for i=1:conf.eNum
    inx = randperm(size(dat,1));        
    mse = 0;
    spe = 0;
    for j=1:bNum
        iiii = inx((j-1)*conf.sNum+1:min(j*conf.sNum,sz));
        visP = dat(iiii,:);               
        sNum = size(visP,1);
       %up
       hidI = visP*model.W + repmat(model.hidB,sNum,1);
       hidP = logistic(hidI);
       hidPs =  1*(hidP >rand(sNum,hidNum));
       hidNs = hidPs;
       for k=1:conf.gNum
           % down
           visN  = logistic(hidNs*model.W' + repmat(model.visB,sNum,1));
           visNs = 1*(visN>rand(sNum,visNum));
%            if j==5 && k==1, save_images(visN,'',sNum,i,28,28); end
           % up
           hidN  = logistic(visNs*model.W + repmat(model.hidB,sNum,1));
           hidNs = 1*(hidN>rand(sNum,hidNum));
       end
       % Compute MSE for reconstruction       
       mse = mse + sum(sum((visP-visN).^2,1)/sNum,2);
       % Update W,visB,hidB
       diff = (visP'*hidP - visNs'*hidN)/sNum;
       DW  = lr*(diff - conf.params(4)*model.W) +  conf.params(3)*DW;
       model.W   = model.W + DW;
       DVB  = lr*sum(visP - visN,1)/sNum + conf.params(3)*DVB;
       model.visB = model.visB + DVB;
       DHB  = lr*sum(hidP - hidN,1)/sNum + conf.params(3)*DHB;
       model.hidB = model.hidB + DHB;
       
       % sparsity constraints
       if conf.lambda >0                      
           pppp = (conf.p - sum(hidP,1)/sNum);
           if sum(sum(isnan(pppp)))>=1 || sum(sum(isinf(pppp)))>=1
              find(isnan(pppp))
              find(isinf(pppp))              
               return;
           end
          % model.W    = model.W   + lr*conf.lambda*(repmat(pppp,visNum,1).*(visP'*((hidN.^2).*exp(-hidI))/sNum));
           model.hidB = model.hidB + lr*conf.lambda*(pppp.*(sum((hidN.^2).*exp(-hidI),1)/sNum));
           h = sum(sum(model.hidB));
           spe = spe+ sum(pppp,2)^2;
        end
    end
   
    fprintf('Epoch %d  : MSE = %.5f|SPE = %.5f\n',i,mse/bNum,spe/bNum);
    
end
    %Visualize
    if exist('vis','var') && vis
    [~,iixxx] = sort(sum(abs(model.W)),'descend');
    MN = min(min(model.W));
    MX = max(max(model.W));
    figure(1); 
    show_images((model.W(:,iixxx)'-MN)/(MX-MN),min(800,hidNum),conf.row,conf.col);
    %show_images(logistic(model.W(:,iixxx)'),min(800,sz),28,28);
    drawnow;
    end
end