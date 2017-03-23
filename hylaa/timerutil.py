'''
Timer utility functions for Hylaa. Timers are used for performance analysis and
can be referred to statically using Timers.tic(name) and Timers.toc(name)

Stanley Bak
September 2016
'''

import time 
from collections import OrderedDict

class TimerData(object):
    'Performance timer which can be started with tic() and paused with toc()'

    def __init__(self, name):
        self.name = name
        self.total_secs = 0
        self.num_calls = 0
        self.last_start_time = None

    def tic(self):
        'start the timer'
        
        if self.last_start_time is not None:
            raise RuntimeError("Timer started twice: {}".format(self.name))

        self.num_calls += 1
        self.last_start_time = time.time()

    def toc(self):
        'stop the timer'

        if self.last_start_time is None:
            raise RuntimeError("Timer stopped without being started: {}".format(self.name))

        self.total_secs += time.time() - self.last_start_time
        self.last_start_time = None

class Timers(object):
    '''
    a static class for doing timer messuarements. Use
    Timers.tic(name) and Timers.tic(name) to start and stop timers, use
    print_stats to print time statistics
    '''

    # map of timer_name -> TimerData
    timers = OrderedDict([('total', TimerData('total'))])

    def __init__(self):
        raise RuntimeError('Timers is a static class; should not be instantiated')

    @staticmethod
    def reset():
        'reset all timers'

        Timers.timers = {'total': TimerData('total')} 

    @staticmethod
    def tic(name):
        'start a timer'

        # create timer object if it doesn't exist
        if Timers.timers.get(name) is None:
            Timers.timers[name] = TimerData(name)

        Timers.timers[name].tic()

    @staticmethod
    def toc(name):
        'stop a timer'

        Timers.timers[name].toc()

    @staticmethod
    def print_stats():
        'print statistics about performance timers to stdout'

        total = Timers.timers["total"].total_secs

        skip_timers = ['total', 'frame']

        for timer in Timers.timers.values():
            if timer.last_start_time is not None:
                raise RuntimeError("timer was never stopped: {}".format(timer.name))

            # print total time last
            if timer.name in skip_timers:
                continue
    
            print "{} Time ({} calls): {:.2f} sec ({:.1f}%)".format(
                timer.name.capitalize(), timer.num_calls, timer.total_secs, 100 * timer.total_secs / total)

        if Timers.timers.get('frame') is not None:
            frame = Timers.timers["frame"]

            overhead = total - frame.total_secs
            print "Matplotlib Overhead ({} frames): {:.2f} sec ({:.1f}%)".format(
                frame.num_calls, overhead, 100 * overhead / total)

        print "Total Time: {:.2f} sec".format(total)





