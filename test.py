import 



train_input, train_target = generator(1000)
test_input, test_target = generator(1000)


model = Sequential(Linear(), ReLu(), Linean(), ReLU())