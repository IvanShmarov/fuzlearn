import numpy as np

class LGM:
    
    def __init__(self, size, subs=2):
        self.size = size
        self.subs = subs
        self.clauses = {}

    ## TODO add clause training
    
    def compcell(self, arr):
        return np.where(arr < 1.0, np.floor(arr * self.subs), self.subs - 1.0)
    
    def spawnclause(self, key, coords):
        clause = np.empty((2, self.size))
        clause[0, :] = coords / self.subs
        clause[1, :] = (coords + 1.0) / self.subs
        self.clauses[key] = clause
    
    def autres(self, clause, arr):
        return np.logical_and(
            np.greater_equal(arr, clause[0, :]),
            np.less_equal(arr, clause[1, :]),
            )

    def classify(self, arr):
        cellkey = self.compcell(arr).tobytes()
        clause = self.clauses.get(cellkey)
        if (clause is not None):
            return np.all(self.autres(clause, arr))
        else:
            return False
    
    def extendclause(self, clause, arr):
        lowers, uppers = clause
        lowers = np.where(arr < lowers, arr, lowers)
        uppers = np.where(arr > uppers, arr, uppers)

    def retractclause(self, clause, arr):
        index = np.argmin(np.abs(clause - arr))
        row = index // self.size
        col = index % self.size
        epsilon = 2.0e-7 if row == 0 else -2.0e-7
        clause[row, col] = arr[col] + epsilon
    
    def train(self, arr, realclass):
        coords = self.compcell(arr)
        cellkey = coords.tobytes()
        clause = self.clauses.get(cellkey)
        if (clause is not None):
            compclass = np.all(self.autres(clause, arr))
            if (compclass == realclass):
                return
            elif (realclass):
                self.extendclause(clause, arr)
            else:
                self.retractclause(clause, arr)
        elif (realclass):
            self.spawnclause(cellkey, coords)


