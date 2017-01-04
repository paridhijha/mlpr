load('imgregdata.mat');
% First scale all of the data to [0, 1] by dividing each pixel value by 63. 
xtr_sc=xtr/63;
ytr_sc=ytr/63;
xte_sc=xte/63;
yte_sc=yte/63;
%compute the standard deviation of each x(j, :) patch
xtr_sd=std(xtr_sc');
% 1(a)- Plot a histogram 
figure
hist(xtr_sd,64)
xlabel('Std deviation');
ylabel('No of patches');
title({'1(a) Histogram of the standard deviations of the patches in the training set.'});
rows = size(xtr_sc,1);
% as adviced in the assignment sheet.
threshold = 4/63;
for i = 1 : rows
    if (xtr_sd(i) <= threshold)
        patch_f=xtr_sc(i,:);
        break;
    end
end
%embed patch with 1's s, increase no of cols from 1032 to 1050 first
patch_embed=horzcat(patch_f,ones(1,18));
% now reshape into (35X30 piel size)
patch_reshaped=reshape(patch_embed,35,30);
figure
colormap gray;
imagesc(patch_reshaped',[0,1]);
title('1(c)-part1 Flat Image Patch');
for i = 1 : rows
    if (xtr_sd(i) > threshold)
        patch_nf=xtr_sc(i,:);
        break;
    end
end
%embed patch with 1's s, increase no of cols from 1032 to 1050 first
patch_embed=horzcat(patch_nf,ones(1,18));
% now reshape into (35X30 piel size)
patch_reshaped=reshape(patch_embed,35,30);
figure
colormap gray;
imagesc(patch_reshaped',[0,1]);
title('1(c)-part2 Non Flat Image Patch')
