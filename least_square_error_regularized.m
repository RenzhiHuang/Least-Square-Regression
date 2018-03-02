d=8;
lambda = [0.1;1;10;100;500;1000]';
loglambda = log10(lambda);
subset = 10;
res_train = zeros(6,1);
res_test = zeros(6,1);
cv_ave = zeros(6,1);
for j=1:6
    %% training model
    %get the train data
    m_train = 721;
    m_test = 309;
    x_train = subs{1,subset}(1:m_train,1:d);
    x_train(:,d+1) = ones(m_train,1);
    y_train = subs{1,subset}(1:m_train,d+1);
    x_test = test(1:m_test,1:d);
    x_test(:,d+1) = ones(m_test,1);
    y_test = test(1:m_test,d+1);
    %calculate the weights and bias
    R = eye(d);
    R(:,d+1)=zeros(d,1);
    R(d+1,:)=zeros(d+1,1);
    w_b = (x_train'* x_train+lambda(j)*m_train*R)^-1*...
        x_train'*y_train;
    w = w_b(1:d);
    b = w_b(d+1);
    %calculate the y_hat
    y_hat_train = x_train*w_b;
    y_hat_test = x_test*w_b;
    res_train(j) = mean_squared_error(y_train,y_hat_train);
    res_test(j) = mean_squared_error(y_test,y_hat_test);
    
    %% cross validation
    cv_sum = 0;
    for i=1:5
        testsub = cv_data_all{1,i};
        trainsub1 = cv_data_all{1,mod(i,5)+1}';
        trainsub2 = cv_data_all{1,mod(i+1,5)+1}';
        trainsub3 = cv_data_all{1,mod(i+2,5)+1}';
        trainsub4 = cv_data_all{1,mod(i+3,5)+1}';
        cv_train = [trainsub1,trainsub2,trainsub3,trainsub4]';
        cv_test = testsub;
        cv_m_train = size(cv_train,1);
        cv_m_test = size(cv_test,1);
        cv_x_train = cv_train(1:cv_m_train,1:d);
        cv_x_train(:,d+1) = ones(cv_m_train,1);
        cv_y_train = cv_train(1:cv_m_train,d+1);
        cv_x_test = cv_test(1:cv_m_test,1:d);
        cv_x_test(:,d+1) = ones(cv_m_test,1);
        cv_y_test = cv_test(1:cv_m_test,d+1);
        %calculate the weights and bias
        R = eye(d);
        R(:,d+1)=zeros(d,1);
        R(d+1,:)=zeros(d+1,1);
        cv_w_b = (cv_x_train'* cv_x_train+lambda(j)*cv_m_train*R)^-1*...
            cv_x_train'*cv_y_train;
        cv_w = cv_w_b(1:d);
        cv_b = cv_w_b(d+1);
        %calculate the y_hat
        cv_y_hat_train = cv_x_train*cv_w_b;
        cv_y_hat_test = cv_x_test*cv_w_b;
        cv_res_train = mean_squared_error(cv_y_train,cv_y_hat_train);
        cv_res_test = mean_squared_error(cv_y_test,cv_y_hat_test);
        cv_sum = cv_sum + cv_res_test;
    end
    cv_ave(j) = cv_sum / 5;
end