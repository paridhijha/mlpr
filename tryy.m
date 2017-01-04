clear
        f=  @(x) exp( -x.^2/2).*(1+(sin(3*x)).^2).* (1+(cos(5*x).^2));
        N=10;
        burn=0;
        widths=100;
        step_out=false;
        INITIAL=0.001;
        x = slice_sample(N, burn, f, INITIAL, widths, step_out);
% %      % Plot a histogram of the sample
%       [binheight,bincenter] = hist(x,50);
%       h = bar(bincenter,binheight,'hist');
%       set(h,'facecolor',[0.8 .8 1]);
% %  woring
%      % Superimpose the f function scaled to have the same area
%      hold on 
%      xd = get(gca,'XLim');
%      xgrid = linspace(xd(1),xd(2),1000);
%      binwidth = (bincenter(2)-bincenter(1));
%      y = (N*binwidth/area) * f(xgrid);
%      plot(xgrid,y,'r','LineWidth',2)
%      hold off