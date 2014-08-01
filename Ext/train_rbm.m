function model = train_rbm(conf,data_file)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training RBM                                                       %  
% conf: training setting                                             %
% W: weights of connections                                          %
% -*-sontran2012-*-                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load data
vars = whos('-file', data_file);
A = load(data_file,vars(1).name);
data = A.(vars(1).name);
[model.W model.visB model.hidB] = train_rbm_(conf,data);
end