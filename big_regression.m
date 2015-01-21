tic;
f = fopen('2008.csv');
%     C = textscan(f, '%d %d %d %d %d %d %d %d %s %d %s %d %d %d %d %d %*[^\n]', 100, 'Delimiter', ',', ...
%         'HeaderLines', 1, 'TreatAsEmpty', 'NA');
C = textscan(f, '%d %d %d %d %d %d %d %d %s %d %s %d %d %d %d %d %*[^\n]', 'Delimiter', ',', ...
    'HeaderLines', 1, 'TreatAsEmpty', 'NA');
%*[^\n]'
[~, ~, ~, DayOfWeek, DepTime, ~,~,~,~,~,~,~,~,~,~,DepDelay] = C{:};
fclose(f);
toc;
clear C;
save('restricted_data.mat');