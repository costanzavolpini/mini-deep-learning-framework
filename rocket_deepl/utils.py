import torch

def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    """
    Compute miss-predicted values.
    Input:
        model: model
        data_input: values
        data_target: values between 0 and 1
        mini_batch_size: size of the batch
    Output:
        nb_data_errors: number of miss-predicted values.
    """

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        
        #retrieve input and target in batches
        input = data_input.narrow(0, b, mini_batch_size)
        target = data_target.narrow(0, b, mini_batch_size)

        #retrive max index of the the target
        target_argmax = torch.argmax(target,1)

        #apply forward pass
        model(input, target.t())

        #change type
        estimated = model.predicted.type(torch.LongTensor)
        target_argmax = target_argmax.type(torch.LongTensor)

        #compare the difference between target and estimated
        diff = estimated-target_argmax

        #count the number of values that are different than zero
        #that would represent the number of missprediction

        diff = len(diff[diff!=0])

        nb_data_errors+= diff
        
    return nb_data_errors

def train_model(model, train_input, train_target, epochs=25, mini_batch_size = 1):
    """
    Function to train the model.
    Input:
        train_input: values to train
        train_target: targets for training (values between 0 and 1)
        epochs: #iterations
        mini_batch_size: size of batch
    """



    for e in range(0, epochs):

        model.zero_grad()
        for batch in range(0, train_input.size(0), mini_batch_size):

            input = train_input.narrow(0, batch, mini_batch_size)
            target = train_target.narrow(0, batch, mini_batch_size)

            model(input, target.t())
            model.zero_grad()
            loss = model.loss
            model.backward()
            model.step()

        accuracy = (1 - (compute_nb_errors(model, train_input, train_target, mini_batch_size))/train_input.size(0))
       
        model.plot_accuracy.append(accuracy)
        model.plot_loss.append(loss)

        print("epoch : {}, loss : {}".format(e,loss))