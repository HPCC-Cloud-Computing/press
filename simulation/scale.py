import math

class Scale(object):
    def __init__(self, nbox, max_workload, cost):
        """
        nbox: number of boxes
        workload: workload of a box
        cost: cost of a box
        """
        self.nbox = nbox
        self.max_workload = max_workload
        self.cost = cost

    def scale(self, data, interval=1):
        """
        Input: 
            data: array
            interval
        Output:
            A tupple of:
            - number of boxes system used
            - cost for run
            - overload time
            - number of request lost

        """
        box = []
        overload_time = 0
        lost_request = 0
        cost = 0
        for d in data:
            b = math.ceil(d / self.max_workload)
            if b > self.nbox:
                box.append(self.nbox)
                overload_time += interval
                lost_request += d - self.nbox * self.max_workload
                cost += self.nbox * self.cost
            else:
                box.append(b)
                cost += b * self.cost

        return (box, cost, overload_time, lost_request)