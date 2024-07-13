import numpy as np

class evSal:
    def __init__(self, lmbda=5, tau=5, height=None, width=None, events=None):
        self.lmbda = lmbda
        self.tau = tau
        self.height = height
        self.width = width
        self.salMaps = {}
        self.events = events
        for i in range(self.tau):
            for j in range(self.lmbda):
                self.salMaps[i,j] = np.zeros((self.height, self.width)) #multi-spatiotemporal consipicuity maps

        self.salMapTop = np.zeros((self.height, self.width))
        self.timeKeys = None
        self.event_dict = {}

        print('grouping events by time...')
        for event in self.events:
            if event['t'] in self.event_dict:
                self.event_dict[event['t']].append(event)
            else:
                self.event_dict[event['t']] = [event]
        
        self.timeKeys = list(self.event_dict.keys()) # list of timestamps in ascending order
        self.timeKeys = sorted(self.timeKeys)
        print(len(self.timeKeys))

    def retrieveUsefulTimes(self, time, u=None):
        t_u = 10*(2**u)
        index = self.timeKeys.index(time)
        useful_times = []

        for i in range(index):
            # print(f'self.timeKeys[index-{i}] ', self.timeKeys[index-i])
            if ( self.timeKeys[index-i] < (self.timeKeys[index] - t_u) ):
                break
            else:
                useful_times.append(self.timeKeys[index-i])

        return useful_times
    
    def spatioTemporalAccumulations(self, trig_event):
        #simple serial implementation. Parallelization will accelerate the process
        for u in range(self.tau):
            useful_times = self.retrieveUsefulTimes(trig_event['t'], u=u)
            # print('useful times:', useful_times)
            for t in useful_times:
                for v in range(self.lmbda):
                    r_v = 1*(2**v)
                    for sub_event in self.event_dict[t]:
                        if ((np.abs(trig_event['x'].astype(np.int32) - sub_event['x'].astype(np.int32)) <= r_v)  and \
                            (np.abs(trig_event['y'].astype(np.int32) - sub_event['y'].astype(np.int32)) <= r_v) ):
                            self.salMaps[u,v][trig_event['x'], trig_event['y']] += 1.0/((1 + 2*r_v)**2)
                    self.salMaps[u,v] = self.salMaps[u,v] #/((1 + 2*r_v)**2)


    def computeSalMap(self, trig_event):
        # self.groupTime()
        self.spatioTemporalAccumulations(trig_event=trig_event)
        for u in range(self.tau):
            for v in range(self.lmbda):
                self.salMapTop += self.salMaps[u,v]

    def normalizeSalMap(self):
        self.salMapTop = self.salMapTop / np.max(self.salMapTop)

    def clearSalMap(self):
        self.salMapTop = np.zeros((self.height, self.width))
        for i in range(self.tau):
            for j in range(self.lmbda):
                self.salMaps[i,j] = np.zeros((self.height, self.width))
