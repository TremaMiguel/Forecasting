from multiprocessing import Process,Queue,Pool,cpu_count
import time 

class parallel_process():

    def __init__(self, function, func_args, elements:list):
        '''
          Perform tasks in parallell with Pools or Queues.
         
          Input:
            :param function: The function that you would like to implement in parallel
            :param func_args: The arguments of the function
            :param elements: List containing the elements to which you would like to implement in parallel the function
        
        Additional Remarks:
          1. For the function provided you should return the results with q.put_nowait(final) or q.put(block = True)
          2. Currently, it implement pool.apply_async() because the call returns immediately instead of waiting for the result.
        '''
        self.function = function
        self.func_args = func_args
        self.elements = elements
       
    def pp_Queue(self):
        
        start = time.time()  
        
        # Start Queue and Process
        queues = [Queue() for i in range(self.num)]
        processes = [Process(target=self.function,
                             args = (queues[i], e, **self.func_args)) 
                             for e in self.elements)]
        
        # Run processes
        for p in processes:
            p.start()
        
        # Join processes:
        for p in processes:
            p.join()
        
        # Get results
        res = [q.get(block = False) for q in queues if q.empty() != True]
        end = time.time()
        
        return (res, end-start)
        
    def pp_Pool(self):
        
        start = time.time()
        
        # Start n processess depending on the number of cpus of your computer
        pool = Pool(processess = cpu_count())
        
        # Start the Pool
        output = [pool.apply_async(self.function,
                                   args = (e, self.func_args))
                                   for e in self.elements]
                                   
        # Get results                           
        res = [p.get() for p in output if p is not None]
        end = time.time()
        
        return (res, end-start)
