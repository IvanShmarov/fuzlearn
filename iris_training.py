
import numpy as np
from src.fuzlearn import LGM


data = np.load('datasets\\iris\\iris.npz')


def meas_acc(machines, test_data):
    acc = 0
    clauses = sum(len(mach.clauses) for mach in machines)
    for data_ind, data_point in enumerate(test_data):
        flag_correct = True
        for mach_ind, mach in enumerate(machines):
            true_res = (data_ind // 10 == mach_ind)
            comp_res = mach.classify(data_point)
            if (true_res != comp_res):
                flag_correct = False
                break
        if (flag_correct):
            acc += 1
    return np.array((acc / 30, clauses))

   
def do_round(seed):
    np.random.seed(seed)
    training_list = []
    test_list = []
    for elem in data.values():
        np.random.shuffle(elem)
        training_list.append(elem[:40])
        test_list.append(elem[40:])
    training_data = np.concatenate(training_list)
    test_data = np.concatenate(test_list)    
    data_indices = np.arange(120)
    np.random.shuffle(data_indices)
    machines = [LGM(4, subs=3) for _ in range(3)]
    res_data = np.empty((121, 2))
    res_data[0, :] = meas_acc(machines, test_data)
    for sample_index in range(120):
        data_index = data_indices[sample_index]
        for mach_index, mach in enumerate(machines):
            mach.train(training_data[data_index], (data_index // 40 == mach_index))
        
        res_data[sample_index+1, :] = meas_acc(machines, test_data)
    
    return res_data


if __name__ == "__main__":
    from random import getrandbits
    num_tests = 1500
    seeds = [getrandbits(32) for _ in range(num_tests)]
    res = np.array([do_round(seed) for seed in seeds])
    
    assert res.shape == (num_tests, 121, 2)
    
    np.save('compdata\\iris\\iris', res)
    