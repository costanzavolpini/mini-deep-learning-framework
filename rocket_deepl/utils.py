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
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

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


        print("epoch : {}, loss : {}".format(e,loss))


def cross_validation(k):
    #TODO: implement it
    pass