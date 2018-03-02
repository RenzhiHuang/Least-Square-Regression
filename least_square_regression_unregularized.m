%get the data and augment the sample matrix
d=1;
subset = 10;
m_train =100;
m_test = 1000;
x_train = subs{1,subset}(1:m_train,1:d);
x_train(:,d+1) = ones(m_train,1);
y_train = subs{1,subset}(1:m_train,d+1);
x_test = test(1:m_test,1:d);
x_test(:,d+1) = ones(m_test,1);
y_test = test(1:m_test,d+1);
%calculate the weights and bias
w_b = (x_train'* x_train)^-1*x_train'*y_train;
%calculate the y_hat and error
y_hat_train = x_train*w_b;
y_hat_test = x_test*w_b;
res_train = mean_squared_error(y_train,y_hat_train);
res_test = mean_squared_error(y_test,y_hat_test);


