

data = readtable("heart_data.csv");
X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));

%We will split the data intro training and test sets

c = cvpartition(size(data,1), "Holdout", 0.2);
idx = c.test;
X_train = X(~idx, :);
y_train = y(~idx, :);

X_test = X(idx, :);
y_test = y(idx, :);

%Define Perceptron Properties

inputSize = size(X_train, 2);
hiddenSize = 5;
outputSize = 1;


%set the weights and biases
W1 = randn(inputSize, hiddenSize);
b1 = zeros(1, hiddenSize);
W2 = randn(hiddenSize, outputSize);
b2 = zeros(1, outputSize);

%Set the parameters
learning_rate = 0.01;
epochs = 1000;

%Train the neural network using Backpropagation

for epoch = 1:epochs
    % Forward Pass
    z1 = X_train * W1 + b1;
    a1 =  1 ./ (1 + exp(-z1));
    z2 = a1 * W2 + b2;
    output = 1 ./ (1 + exp(-z2));

    % Calculate Loss (assuming binary cross-entropy)
    loss = -sum(y_train .* log(output) + (1- y_train) .* log(1 - output)) / size (X_train, 1);

    % Backward Pass
    delta2 = output - y_train;
    delta1 = delta2 * W2' .* (a1 .* (1 - a1));

    % Update Weights and Biases
    W2 = W2 - learning_rate * a1' * delta2 / size(X_train, 1);
    b2 = b2 - learning_rate * sum(delta2) / size(X_train, 1);
    W1 = W1 - learning_rate * X_train' * delta1 / size(X_train, 1);
    b1 = b1 - learning_rate * sum(delta1) / size(X_train, 1);

    if mod(epoch, 100) == 0
        disp(['Epoch: ', num2str(epoch), ', Loss: ', num2str(loss)]);
    end
end

% Predict on the training set
z1_train = X_train * W1 + b1;
a1_train = 1 ./ (1 + exp(-z1_train));
z2_train = a1_train * W2 + b2;
predicted_output_train = 1 ./ (1 + exp(-z2_train));

% Round to obtain binary predictions (0 or 1)
predicted_labels_train = round(predicted_output_train);

% Calculate accuracy
accuracy_train = sum(predicted_labels_train == y_train) / length(y_train);
disp(['Training Accuracy: ', num2str(accuracy_train)]);

%Predict on the test set
z1_test = X_test * W1 + b1;
a1_test = 1 ./ (1 + exp(-z1_test));
z2_test = a1_test * W2 + b2;
predicted_output_test = 1 ./ (1 + exp(-z2_test));
predicted_labels_test = round(predicted_output_test);

% Calculate accuracy on the test set
accuracy_test = sum(predicted_labels_test == y_test) / length(y_test);
disp(['Test Accuracy: ', num2str(accuracy_test)]);

%Measure training time
training_time = toc;
disp(['Training Time: ', num2str(training_time), ' seconds']);

%We will now perform a Grid Search using various learning rates and hidden
%layer size values 

learning_rate_grid = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.5];
hiddenSize_grid = [1, 3, 5, 7, 8, 10];

accuracy_values = zeros(length(learning_rate_grid), length(hiddenSize_grid));
training_times = zeros(length(learning_rate_grid), length(hiddenSize_grid));

for i = 1:length(learning_rate_grid)
    for j = 1:length(hiddenSize_grid)

        for epoch = 1:epochs
            % Forward Pass
            z1 = X_train * W1 + b1;
            a1 =  1 ./ (1 + exp(-z1));
            z2 = a1 * W2 + b2;
            output = 1 ./ (1 + exp(-z2));

            % Calculate Loss (assuming binary cross-entropy)
            loss = -sum(y_train .* log(output) + (1- y_train) .* log(1 - output)) / size (X_train, 1);

            % Backward Pass
            delta2 = output - y_train;
            delta1 = delta2 * W2' .* (a1 .* (1 - a1));

            % Update Weights and Biases
            W2 = W2 - learning_rate * a1' * delta2 / size(X_train, 1);
            b2 = b2 - learning_rate * sum(delta2) / size(X_train, 1);
            W1 = W1 - learning_rate * X_train' * delta1 / size(X_train, 1);
            b1 = b1 - learning_rate * sum(delta1) / size(X_train, 1);

            if mod(epoch, 100) == 0
                disp(['Epoch: ', num2str(epoch), ', Loss: ', num2str(loss)]);
            end
        end
          
                %Predict on the test set
                z1_test = X_test * W1 + b1;
                a1_test = 1 ./ (1 + exp(-z1_test));
                z2_test = a1_test * W2 + b2;
                predicted_output_test = 1 ./ (1 + exp(-z2_test));
                predicted_labels_test = round(predicted_output_test);

                % Calculate accuracy on the test set
                accuracy_test = sum(predicted_labels_test == y_test) / length(y_test);
                disp(['Test Accuracy: ', num2str(accuracy_test)]);

                %Store the accuracy scores

                accuracy_values(i, j) = accuracy_test;

                %Measure training time
                training_time = toc;
                disp(['Training Time: ', num2str(training_time), ' seconds']);
                training_times(i, j) = training_time;


    end

end

%We will now plot the highest accuracy  and the lowest training time

[max_accuracy, idx_max_accuracy] = max(accuracy_values(:));
[min_training_time, idx_min_time] = min(training_times(:));

% Find the corresponding learning rate and hidden layer size
[learning_rate_idx, hiddenSize_idx] = ind2sub(size(accuracy_values), idx_max_accuracy);
best_learning_rate = learning_rate_grid(learning_rate_idx);
best_hiddenSize = hiddenSize_grid(hiddenSize_idx);

disp(['Best Accuracy: ', num2str(max_accuracy), ' at Learning Rate: ', num2str(best_learning_rate), ', Hidden Size: ', num2str(best_hiddenSize)]);
disp(['Lowest Training Time: ', num2str(min_training_time), ' at Learning Rate: ', num2str(learning_rate_grid(idx_min_time)), ', Hidden Size: ', num2str(hiddenSize_grid(idx_min_time))]);

figure;

% Plot Accuracy
subplot(1, 2, 1);
imagesc(accuracy_values);
title('Accuracy Grid Search');
xlabel('Hidden Size');
ylabel('Learning Rate');
colorbar;

hold on;
scatter(hiddenSize_idx, learning_rate_idx, 'r', 'filled');
hold off;

% Plot Training Time
subplot(1, 2, 2);
imagesc(training_times);
title('Training Time Grid Search');
xlabel('Hidden Size');
ylabel('Learning Rate');
colorbar;

% Mark the minimum training time
hold on;
scatter(idx_min_time,learning_rate_idx, 'r', 'filled');
hold off;

% Using the ReLu activation function %



