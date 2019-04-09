import torch

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

#reused code from correction
def train_model(model, train_input, train_target, epochs=25, mini_batch_size = 1):

    for e in range(0, epochs):
        for batch in range(0, train_input.size(0), mini_batch_size):
            model(train_target.narrow(0, batch, mini_batch_size))

            print("loss at epoch {} = {}".format(e,model.loss))
            
            model.zero_grad()
            model.backward()
            model.step()
            
            #TODO: output the loss (bonus accuracy viz)


            #TODO: 


def cross_validation(k):
    pass 