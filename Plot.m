% load the dataset
data = load('./logs2.txt');
steps = data(:,1);
loss = data(:,2);
train_acc = data(:,3);
test_acc = data(:,4);


plot(steps, train_acc);
hold on

plot(steps, test_acc);
hold on

legend('Minibatch accuracy', 'Test accuracy')