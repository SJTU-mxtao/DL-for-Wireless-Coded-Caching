%% 根据分类的结果进行LSTM预测' offline
clear
close all
tt='k_meanplusC7_Youtube_sample_t660.mat';
load(tt, 'Idx')
load(tt, 'cluster_number')
load(tt, 'window')
load(tt, 'kmeans_sample')
load(tt, 'cluster_sample_number')

num_cluster=cluster_number;
% window=10;
size_each_cluster=cluster_sample_number;
kmeans_sample_norm=zeros(max(size_each_cluster(1,:)),window+2,num_cluster);%一行代表一个样本，前window列是输入,window+1列是输出，window+2列是index，第三维是类别
max_norm=zeros(1,num_cluster);
for i=1:num_cluster
    iid=find(Idx==i);
    nm=length(iid);
    max_norm(1,i)=max(max(abs(kmeans_sample(iid,:))));
    kmeans_sample_norm(1:nm,1:window+1,i)=kmeans_sample(iid,:)./max_norm(1,i);
    kmeans_sample_norm(1:nm,window+2,i)=iid;
    
    
end

save Lstm_kmeans_C4W12T660_off_train.mat kmeans_sample_norm num_cluster size_each_cluster window max_norm






% s=size(cluster);
% index_cluster=ones(1,num_cluster);
% for i=1:s(1)-1
%     c=cluster(i,1)+1;
%     for w=1:window+1
%         for f=1:F
%             if ticc(i,f)==0
%                  kmeans_sample_norm(w,f,index_cluster(1,c),c)=ticc(i+w-1,f);
%             else
%                  kmeans_sample_norm(w,f,index_cluster(1,c),c)= ticc(i+w-1,f)/ticc(i,f);
%             end
%            lstm_sample_norm(w,:,index_cluster(1,c),c)=ticc(i+w-1,:)./ticc(i,:);
%         end
%     end
%     lstm_sample_norm(1:window+1,:,index_cluster(1,c),c)=ticc(i:window+i,:);
%     kmeans_sample_norm(window+2,1,index_cluster(1,c),c)=i+window;
%     lstm_sample_std(1,:,index_cluster(1,c),c)=ticc(i,:);
%     
%     index_cluster(1,c)=index_cluster(1,c)+1;
% end
% save(tt,'size_each_cluster','lstm_sample_norm','lstm_sample_std','-append');
