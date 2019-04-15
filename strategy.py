import numpy as np
import matplotlib.pyplot as plt



class Strategy:
    '''
    In this class, we have methods
        training_method
        recursive_response
    and objects
        validator
    '''
    def __init__(self, training_method, recursive_response, validator):
        self.training_method = training_method
        self.recursive_response = recursive_response
        self.attempt_record = []
        self.validator = validator

    def do_strategy(self, phi_0=None, **kwargs):
        error_message = "Training must be initialized with \
        some training method or some initial predictor."
        assert self.training_method or phi_0, error_message
        if self.training_method:
            phi = self.training_method(**kwargs)
        else:
            phi = phi_0
        answer = self.validator.loss(phi)
        attempt = (phi, answer)
        self.attempt_record.append(attempt)
        while attempt[0]:
            attempt = self.recursive_response(self.attempt_record, self.validator,  **kwargs)
            self.attempt_record.append(attempt)
    def show_attempts(self):
        # error_message = "The strategy has not been executed, so there are no \
        # attempts to show."
        # assert len(self.attempt_record) > 0, error_message
        self.attempt_record = np.array(self.attempt_record)
        self.loss_plot_fig, self.loss_plot_ax = plt.subplots()
        self.loss_plot_ax.plot(self.attempt_record[:,1])
        self.loss_plot_ax.set(xlabel = "Attempt number",
                            ylabel = "Loss (approximate)"
                            )
        self.loss_plot_fig.show()


def adaptive_mean_estimation_1(attempt_record, validator, step_size, tolerance):
    '''
    Performs a linear search to find the minimum loss solution

    '''
    last_attempt = attempt_record[-1]
    if last_attempt[1] <= tolerance:
        return (None, None)
    phi_l = last_attempt[0] - step_size
    answer_l = validator.loss(phi_l)
    phi_r = last_attempt[0] + step_size
    answer_r = validator.loss(phi_r)
    if answer_r <= answer_l:
        return (phi_r, answer_r)
    else:
        return (phi_l, answer_l)
