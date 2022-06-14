
import numpy as np
from src.fuzlearn import LGM


data = np.load('datasets\\iris\\iris.npz')
setosa = data['setosa']
versi = data['versi']
virgi = data['virgi']


NUM_ROUNDS = 25


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

   
def do_round(seed, machines):
    np.random.seed(seed)
    training_list = []
    test_list = []
    for elem in (setosa, versi, virgi):
        np.random.shuffle(elem)
        training_list.append(elem[:40])
        test_list.append(elem[40:])
    
    training_data = np.concatenate(training_list)
    test_data = np.concatenate(test_list)    
    data_indices = np.arange(120)
    np.random.shuffle(data_indices)
    
    for sample_index in range(120):
        data_index = data_indices[sample_index]
        for mach_index, mach in enumerate(machines):
            mach.train(training_data[data_index], (data_index // 40 == mach_index))
    
    return meas_acc(machines, test_data)


def do_session(seed):
    np.random.seed(seed)
    
    machines = [LGM(4, subs=3) for _ in range(3)]
    
    res_data = np.empty((NUM_ROUNDS + 1, 2))
    res_data[0, :] = np.array([0,0])
    
    for round_ind in range(NUM_ROUNDS):
        res_data[round_ind + 1, :] = do_round(seed + round_ind, machines)  
    return res_data

if __name__ == "__main__":
    import winsound as ws
    from random import getrandbits
    num_sessions = 1500
    seeds = [getrandbits(32) for _ in range(num_sessions)]
    res = np.array([do_session(seed) for seed in seeds])
    
    assert res.shape == (num_sessions, NUM_ROUNDS + 1, 2)
    
    np.save('compdata\\iris\\iris_vs_epoch', res)
    ws.PlaySound('SystemAsterisk', ws.SND_ALIAS | ws.SND_ASYNC)
    