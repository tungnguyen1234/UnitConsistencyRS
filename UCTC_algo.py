import torch as t
import gc

device = t.device(f"cuda:0" if t.cuda.is_available() else "cpu")
epsilon = 1e-9


class TensorTC():
    def __init__(self, device, tensor, epsilon):
        self.device = device
        self.tensor = tensor.clone()
        self.epsilon = epsilon

    def TC(self):
        '''
        Desciption:
            This function runs the tensor latent invariant algorithm.
        Input:
            tensor: torch.tensor
                The tensor to retrive the latent variables from.
            epsilon: float
                The convergence number for the algorithm.
        Output:
            Returns the latent vectors and the convergent errors from the iterative steps.
        '''

        d1, d2 = self.tensor.shape
        # Create a mask of non-zero elements
        self.rho_sign = (self.tensor != 0)*1

        # Get the number of nonzeros inside each row
        self.sigma_first = self.rho_sign.sum(1)
        self.sigma_second = self.rho_sign.sum(0)
        tensor = self.tensor.clone().detach()

        # Initiate lantent variables
        self.latent_1 = t.zeros(d1).to(self.device)
        self.latent_2 = t.zeros(d2).to(self.device)

        # Iteration errors
        errors = []
        step = 1
        # print('Start the TC process:')


        while True:
            error = 0.0
            self.rho_first = - t.div(tensor.sum(1), self.sigma_first).nan_to_num(0.0) # d1
            tensor += self.rho_first[:, None] * self.rho_sign
            self.latent_1 -= self.rho_first
            error += (self.rho_first**2).sum()


            self.rho_second = - t.div(tensor.sum(0), self.sigma_second).nan_to_num(0.0) # d2
            tensor += self.rho_second[None, :] * self.rho_sign
            self.latent_2 -= self.rho_second
            error += (self.rho_second**2).sum()

            error = float(error)

            gc.collect()
            t.cuda.empty_cache()
            errors.append(float(error))
            # print(f'This is step {step} with error {float(error)}')
            step += 1
            if error < self.epsilon:
                break

        # return the latent variables and errors
        tensor_pred = self.latent_1[:, None] + tensor + self.latent_2[None, :]
        return tensor_pred
    
    def TC_components(self):
        '''
        Desciption:
            This function runs the tensor latent invariant algorithm.
        Input:
            tensor: torch.tensor
                The tensor to retrive the latent variables from.
            epsilon: float
                The convergence number for the algorithm.
        Output:
            Returns the latent vectors and the convergent errors from the iterative steps.
        '''

        d1, d2 = self.tensor.shape
        # Create a mask of non-zero elements
        self.rho_sign = (self.tensor != 0)*1

        # Get the number of nonzeros inside each row
        self.sigma_first = self.rho_sign.sum(1)
        self.sigma_second = self.rho_sign.sum(0)
        tensor = self.tensor.clone().detach()

        # Initiate lantent variables
        self.latent_1 = t.zeros(d1).to(self.device)
        self.latent_2 = t.zeros(d2).to(self.device)

        # Iteration errors
        errors = []
        step = 1
        # print('Start the TC process:')


        while True:
            error = 0.0
            self.rho_first = - t.div(tensor.sum(1), self.sigma_first).nan_to_num(0.0) # d1
            tensor += self.rho_first[:, None] * self.rho_sign
            self.latent_1 -= self.rho_first
            error += (self.rho_first**2).sum()


            self.rho_second = - t.div(tensor.sum(0), self.sigma_second).nan_to_num(0.0) # d2
            tensor += self.rho_second[None, :] * self.rho_sign
            self.latent_2 -= self.rho_second
            error += (self.rho_second**2).sum()

            error = float(error)

            gc.collect()
            t.cuda.empty_cache()
            errors.append(float(error))
            # print(f'This is step {step} with error {float(error)}')
            step += 1
            if error < self.epsilon:
                break

        # return the latent variables and errors
        return self.latent_1, tensor, self.latent_2


class TensorUC():
    def __init__(self, device, tensor, epsilon):
        self.device = device
        self.tensor = tensor.clone()
        self.epsilon = epsilon

    def UC(self):
        '''
        Desciption:
            This function runs the tensor latent invariant algorithm.
        Input:
            tensor: torch.tensor
                The tensor to retrive the latent variables from.
            epsilon: float
                The convergence number for the algorithm.
        Output:
            Returns the latent vectors and the convergent errors from the iterative steps.
        '''

        d1, d2 = self.tensor.shape
        # Create a mask of non-zero elements
        rho_sign = (self.tensor != 0)*1

        # Get the number of nonzeros inside each row
        sigma_first = rho_sign.sum(1)
        sigma_second = rho_sign.sum(0)

        # Take log spaceof tensor
        tensor_log = t.log(self.tensor)

        # After log, all 0 values will be -inf, so we set them to 0
        tensor_log[tensor_log == - float("Inf")] = 0.0

        # Initiate lantent variables
        latent_1 = t.zeros(d1).to(self.device)
        latent_2 = t.zeros(d2).to(self.device)

        # Starting the iterative steps
        step = 1

        # Iteration errors
        errors = []

        # print('Start the UC process:')
        step = 1
        while True:
            error = 0.0
            if step % 2 == 0:
                rho_second = - t.div(tensor_log.sum(0), sigma_second).nan_to_num(0.0) # d2
                tensor_log += rho_second[None, :] * rho_sign
                latent_2 -= rho_second
                error += (rho_second**2).sum()

                rho_first = - t.div(tensor_log.sum(1), sigma_first).nan_to_num(0.0) # d1
                tensor_log += rho_first[:, None] * rho_sign
                latent_1 -= rho_first
                error += (rho_first**2).sum()
            else:
                rho_first = - t.div(tensor_log.sum(1), sigma_first).nan_to_num(0.0) # d1
                tensor_log += rho_first[:, None] * rho_sign
                latent_1 -= rho_first
                error += (rho_first**2).sum()

                rho_second = - t.div(tensor_log.sum(0), sigma_second).nan_to_num(0.0) # d2
                tensor_log += rho_second[None, :] * rho_sign
                latent_2 -= rho_second
                error += (rho_second**2).sum()

            gc.collect()
            t.cuda.empty_cache()

            errors.append(float(error))

            # print(f'This is step {step} with error {float(error)}')
            step += 1
            if error < self.epsilon:
                break

        # return the latent variables and errors
        # latent_1, latent_2 = t.exp(latent_1), t.exp(latent_2)
        gc.collect()
        t.cuda.empty_cache()
        tensor_return = t.exp(latent_1[:, None] + tensor_log + latent_2[None, :])
        return tensor_return
    
    def UC_components(self):
        '''
        Desciption:
            This function runs the tensor latent invariant algorithm.
        Input:
            tensor: torch.tensor
                The tensor to retrive the latent variables from.
            epsilon: float
                The convergence number for the algorithm.
        Output:
            Returns the latent vectors and the convergent errors from the iterative steps.
        '''

        d1, d2 = self.tensor.shape
        # Create a mask of non-zero elements
        rho_sign = (self.tensor != 0)*1

        # Get the number of nonzeros inside each row
        sigma_first = rho_sign.sum(1)
        sigma_second = rho_sign.sum(0)

        # Take log spaceof tensor
        tensor_log = t.log(self.tensor)

        # After log, all 0 values will be -inf, so we set them to 0
        tensor_log[tensor_log == - float("Inf")] = 0.0

        # Initiate lantent variables
        latent_1 = t.zeros(d1).to(self.device)
        latent_2 = t.zeros(d2).to(self.device)

        # Starting the iterative steps
        step = 1

        # Iteration errors
        errors = []

        # print('Start the UC process:')
        step = 1
        while True:
            error = 0.0
            if step % 2 == 0:
                rho_second = - t.div(tensor_log.sum(0), sigma_second).nan_to_num(0.0) # d2
                tensor_log += rho_second[None, :] * rho_sign
                latent_2 -= rho_second
                error += (rho_second**2).sum()

                rho_first = - t.div(tensor_log.sum(1), sigma_first).nan_to_num(0.0) # d1
                tensor_log += rho_first[:, None] * rho_sign
                latent_1 -= rho_first
                error += (rho_first**2).sum()
            else:
                rho_first = - t.div(tensor_log.sum(1), sigma_first).nan_to_num(0.0) # d1
                tensor_log += rho_first[:, None] * rho_sign
                latent_1 -= rho_first
                error += (rho_first**2).sum()

                rho_second = - t.div(tensor_log.sum(0), sigma_second).nan_to_num(0.0) # d2
                tensor_log += rho_second[None, :] * rho_sign
                latent_2 -= rho_second
                error += (rho_second**2).sum()

            gc.collect()
            t.cuda.empty_cache()

            errors.append(float(error))

            # print(f'This is step {step} with error {float(error)}')
            step += 1
            if error < self.epsilon:
                break

        # return the latent variables and errors
        latent_1, latent_2 = t.exp(latent_1), t.exp(latent_2)
        tensor = t.exp(tensor_log)
        gc.collect()
        t.cuda.empty_cache()
        return latent_1, tensor, latent_2