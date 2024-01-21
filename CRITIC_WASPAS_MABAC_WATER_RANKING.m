df = readtable('water_potability.csv', 'VariableNamingRule', 'preserve');
% Extract features (X) and labels (y)
X = df(:, 1:end-1); % Exclude the last column ('Potability') as it contains labels
y = df.Potability;  % Extract labels

% Split the data into training and testing sets
rng(42); % Set random seed for reproducibility (equivalent to random_state=42)
split_ratio = 0.2; % Test size
cv = cvpartition(size(df, 1), 'HoldOut', split_ratio);

X_train = X(cv.training, :);
y_train = y(cv.training);

X_test = X(cv.test, :);
y_test = y(cv.test);

% Assuming X_train is a table or matrix

% Calculate basic statistics for each feature (column) in X_train
X_train = table2array(X_train);
X_test = table2array(X_test);
df1= table2array(df);
skewness_values = skewness(X_train);
% Calculate the median for each column in X_train_matrix
median_values = median(X_train, 'omitnan');
for j=1:width(df)-1
    nan_indices_train = isnan(df1(:, j));
    df1(nan_indices_train, j) = median_values(1,j);
end
water_numericData=df1(:, 1:end-1);
%Now convert into fuzzy numbers:
numerical_positions = [1, 2, 3, 4, 5, 6, 7, 8, 9]; % Adjust these positions as needed

% Create a figure
figure;

% Define the number of rows and columns for the subplots
num_rows = 3;
num_cols = 3;
bar_colors = {'b', 'g', 'r', 'c', 'm', 'y', 'k', [0.5, 0.5, 0.5], [0.8, 0.2, 0.6]};
% Loop through the numerical positions
for i = 1:numel(numerical_positions)
    % Create a subplot
    subplot(num_rows, num_cols, i);
    
    % Get the data for the corresponding position
    column_data = water_numericData(:, numerical_positions(i));
    
    % Plot the histogram as bars
    histogram(column_data, 'DisplayStyle', 'bar', 'EdgeColor', 'k', 'FaceColor', bar_colors{i}, 'Normalization', 'pdf');    
    % Add grid lines
    grid on;
    
    % Add a title with the position
    title(sprintf('Position %d', numerical_positions(i)));
    
    % Estimate and plot the KDE curve directly on the bars
    hold on;
    [f, xi] = ksdensity(column_data);
    plot(xi, f, 'LineWidth', 1.5, 'Color', 'r'); % Adjust LineWidth and Color as needed
    hold off;
end

% find the standard deviation of each criteria:
% Calculate the sum of squared differences from the mean
%Next here
%Find the standard deviation
for j=1:width(water_numericData)
    std_dev(j)=std(water_numericData(:,j));
    mean_data(j)=mean(water_numericData(:,j));
end
for i=1:length(water_numericData)
    for j=1:width(water_numericData)
        membership_values(i,j) = gaussmf(water_numericData(i,j), [std_dev(j), mean_data(j)]);
    end
end
%Know Apply Critic Method to membersip values
%Step2, normalize the score values
for j=1:width(membership_values)
    for i=1:length(membership_values)
        norm_mem_value(i,j)=(membership_values(i,j)-min(membership_values(:,j)))/((max(membership_values(:,j)))-min(membership_values(:,j)));
    end
end
%calculate the standard deviation 

% Calculate the sum of squared differences from the mean
% Calculate the standard deviation
for j=1:width(membership_values)
    std_deviation(j) = std(membership_values(:,j));
end
% disp(std_deviation);
%find the correlation of mem_value
r=corr(membership_values);
% Examine the information for each criterion
for j=1:width(membership_values)
    s2=0;
    for t=1:width(membership_values)
        s2=s2+(1-r(j,t));
    end
        c(j)=std_deviation(j).*s2;
end
%FInd the weight of each criteria
for j=1:width(norm_mem_value)
    w(j)=c(j)/sum(c);
end
%Then apply the Aggregation operator

%Apply MABAC method
for i=1:length(membership_values)
    for j=1:width(membership_values)
            Sc_mabac(i,j)=membership_values(i,j)/max(membership_values(i,:));
    end
end

for i=1:length(membership_values)
    for j=1:width(membership_values)
            Sc_mabac(i,j)=(membership_values(i,j)-min(membership_values(:,j)))/(max(membership_values(:,j))-min(membership_values(:,j)));
    end
end

for i=1:length(membership_values)
    for j=1:width(membership_values)
        Sc_edge_mab(i,j)=w(j)+Sc_mabac(i,j)*w(j);
    end
end
for j=1:width(membership_values)
    prod=1;
    for i=1:length(membership_values)
%         prod=prod*Sc_edge_mab(i,j);
        prod=prod*(Sc_edge_mab(i,j))^(1/length(membership_values));
    end
        g_edge_mab(j)=(prod);
end
for i=1:length(membership_values)
    for j=1:width(membership_values)
        q_mabac(i,j)=Sc_edge_mab(i,j)-g_edge_mab(j);
    end
end

for i=1:length(membership_values)
    s=0;
    for j=1:width(membership_values)
        phi_mabac(i)=sum(q_mabac(i,:));
    end
end
% bar(phi_mabac)
figure;
max_index_mabac = find(phi_mabac == max(phi_mabac));
bar(phi_mabac);
hold on;
bar(max_index_mabac, phi_mabac(max_index_mabac), 'r', 'BarWidth', 4); % Highlight the maximum value with a red bar
% Display the corresponding values on top of the red bar
text(max_index_mabac, phi_mabac(max_index_1), ['x = ', num2str(max_index_mabac)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8);
text(max_index_mabac, phi_mabac(max_index_1), ['y = ', num2str(phi_mabac(max_index_mabac))], 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontSize', 8);

% phi_mabac_sort=sort(phi_mabac,'descend');
[sortedData1, sortedIndices1] = sort(phi_mabac, 'descend');
%Apply the WASPAS Method
for i=1:length(membership_values)
    for j=1:width(membership_values)
%         if j==1 || j==2 || j==3 || j==4
%             Sc_waspas(i,j)=min(membership_values(i,:))/membership_values(i,j);
%         else
            Sc_waspas(i,j)=membership_values(i,j)/max(membership_values(i,:));
%         end
    end
end
% The Additive Relative Importance
for i=1:length(membership_values)
    s=0;
    for j=1:width(membership_values)
            s=s+ Sc_waspas(i,j)*w(j);
    end
       Q1_waspas(i)=s;
end
% The Multiplicative Relative Importance
for i=1:length(membership_values)
    prod=1;
    for j=1:width(membership_values)
            prod=prod*Sc_waspas(i,j)^w(j);
    end
       Q2_waspas(i)=prod;
end
for i=1:length(membership_values)
        Q_waspas(i)=(Q1_waspas(i)+Q2_waspas(i))/2;
end

%Apply by taking lambda
Q_waspas_lm=cell(1,4);
for lmda=[0.1 0.4 0.6 0.8]
    for i=1:length(membership_values)
        if lmda==0.1
            Q_waspas_lm{1}(i)=lmda*Q1_waspas(i)+(1-lmda)*Q2_waspas(i);
        elseif lmda==0.4
            Q_waspas_lm{2}(i)=lmda*Q1_waspas(i)+(1-lmda)*Q2_waspas(i);
        elseif lmda==0.6
            Q_waspas_lm{3}(i)=lmda*Q1_waspas(i)+(1-lmda)*Q2_waspas(i);
        else
            Q_waspas_lm{4}(i)=lmda*Q1_waspas(i)+(1-lmda)*Q2_waspas(i);
        end
    end
end
A=[1:length(membership_values)];
% new_waspas=cat(2,transpose(A),membership_values,transpose(Q_waspas_lm));
new_waspas=cat(2,transpose(A),membership_values,transpose(Q_waspas));
% Extract the last row (not the last column)
last_row = new_waspas(end, :);
% Create a sorting index vector in descending order of the last row values
[~, sorting_index] = sort(new_waspas(:,11), 'descend');
max_index = find(Q_waspas == max(Q_waspas));

bar(Q_waspas);
hold on;
bar(max_index, Q_waspas(max_index), 'r', 'BarWidth', 5); % Highlight the maximum value with a red bar
% Display the corresponding values on top of the red bar
text(max_index, Q_waspas(max_index), ['x = ', num2str(max_index)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8);
%text(max_index, Q_waspas(max_index), ['y = ', num2str(Q_waspas_lm(max_index))], 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontSize', 8);
hold off;
xlabel('Index');
ylabel('Ranking');
title('Bar Plot with Red Bar at Maximum Value')

% Use the sorting index to rearrange the rows of the matrix
new_waspas_sorted = new_waspas(sorting_index, :);
  new_waspas_water_data=cat(2,transpose(A),df1,transpose(Q_waspas));
  new_waspas_sorted_water_data = new_waspas_water_data(sorting_index, :);
% If you want to remove the last row after sorting:
%sorted_matrix_without_last_row = new_waspas_sorted(1:end-1, :);find(Q_waspas_lm==max(Q_waspas_lm))
% Plotthe graph of 
% bar(Q_waspas_lm{1})

% Create the figure and subplots
figure;

% Subplot 1
subplot(2, 2, 1);
max_index_1 = find(Q_waspas_lm{1} == max(Q_waspas_lm{1}));
bar(Q_waspas_lm{1});
hold on;
bar(max_index_1, Q_waspas_lm{1}(max_index_1), 'r', 'BarWidth', 4); % Highlight the maximum value with a red bar
% Display the corresponding values on top of the red bar
text(max_index_1, Q_waspas_lm{1}(max_index_1), ['x = ', num2str(max_index_1)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8);
text(max_index_1, Q_waspas_lm{1}(max_index_1), ['y = ', num2str(Q_waspas_lm{1}(max_index_1))], 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontSize', 8);
hold off;
%title(sprintf('Sigma = %.1f', sigma_values(1)));
ylabel('Ranking');
xlabel('γ');
%set(gca, 'XTickLabel', curlyvee_values);

% Subplot 2
subplot(2, 2, 2);
max_index_2 = find(Q_waspas_lm{2} == max(Q_waspas_lm{2}));
bar(Q_waspas_lm{2});
hold on;
bar(max_index_2, Q_waspas_lm{2}(max_index_2), 'r', 'BarWidth', 4); % Highlight the maximum value with a red bar
% Display the corresponding values on top of the red bar
text(max_index_2, Q_waspas_lm{2}(max_index_2), ['x = ', num2str(max_index_2)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8);
text(max_index_2, Q_waspas_lm{2}(max_index_2), ['y = ', num2str(Q_waspas_lm{2}(max_index_2))], 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontSize', 8);
hold off;
% Subplot 3
subplot(2, 2, 3);
max_index_3 = find(Q_waspas_lm{3} == max(Q_waspas_lm{3}));
bar(Q_waspas_lm{3});
hold on;
bar(max_index_3, Q_waspas_lm{3}(max_index_3), 'r', 'BarWidth', 4); % Highlight the maximum value with a red bar
% Display the corresponding values on top of the red bar
text(max_index_3, Q_waspas_lm{3}(max_index_3), ['x = ', num2str(max_index_3)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8);
text(max_index_3, Q_waspas_lm{3}(max_index_3), ['y = ', num2str(Q_waspas_lm{3}(max_index_3))], 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontSize', 8);
hold off;
 xlabel('γ');
 ylabel('Ranking');
% Subplot 4
subplot(2, 2, 4);
max_index_4 = find(Q_waspas_lm{4} == max(Q_waspas_lm{4}));
bar(Q_waspas_lm{4});
hold on;
bar(max_index_4, Q_waspas_lm{4}(max_index_4), 'r', 'BarWidth', 4); % Highlight the maximum value with a red bar
% Display the corresponding values on top of the red bar
text(max_index_4, Q_waspas_lm{4}(max_index_4), ['x = ', num2str(max_index_4)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8);
text(max_index_4, Q_waspas_lm{4}(max_index_4), ['y = ', num2str(Q_waspas_lm{4}(max_index_4))], 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', 'FontSize', 8);
hold off;
 xlabel('γ');
 ylabel('Ranking');