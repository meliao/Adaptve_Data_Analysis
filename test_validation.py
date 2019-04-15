import pytest
import numpy as np
import strategy
import mock_validation_server

def loss_1(mean, guess):
    pass

def test_validation_1():
    '''
    Test to ensure the validation constructor works correctly
    '''
    my_vec = np.array([0,1,1,0])
    aa = mock_validation_server.Validation(loss_1, my_vec, 0)
    assert type(aa) == mock_validation_server.Validation

vec_0 = np.array([0,0,0,0])
vec_1 = np.array([1,1,1,1])
vec_2 = np.array([2,2,0,0])
validation_server_1 = mock_validation_server.Validation(mock_validation_server.mean_loss,
                                        vec_0, 0)
def test_validation_2():
    '''
    Test to ensure that in the multi-dimensional case, the loss function
    'mock_validation_server.mean_loss' works correctly
    '''
    assert validation_server_1.loss(vec_1) == 2
def test_validation_3():
    assert np.isclose(validation_server_1.loss(vec_2), 2 * np.sqrt(2))

def test_validation_4():
    lst = []
    for i in range(100):
        lst.append(validation_server_1.trunc_gauss())
        assert not np.any(lst)

validation_server_2 =  mock_validation_server.Validation(mock_validation_server.mean_loss,
                                        vec_0, 0.5)
def test_validation_5():
        lst = []
        for i in range(100):
            lst.append(validation_server_1.trunc_gauss())
        arr = np.array(lst)
        bad_values = np.where(arr > 0.5)
        assert not np.any(bad_values)
