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

    def __init__(self, name, parent):
        assert parent is None or isinstance(parent, TimerData)

        self.name = name
        self.total_secs = 0
        self.num_calls = 0
        self.last_start_time = None

        self.parent = parent # parent TimerData, None for top-level timers
        self.children = [] # a list of child TimerData

    def full_name(self):
        'get the full name of the timer (including ancestors)'

        if self.parent is None:
            return self.name
        else:
            return "{}.{}".format(self.parent.full_name(), self.name)

    def tic(self):
        'start the timer'

        #print "Tic({})".format(self.name)

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

    # complete map of timer_name -> TimerData
    timers = OrderedDict()

    stack = [] # stack of currently-running timers, parents at the start, children at the end

    def __init__(self):
        raise RuntimeError('Timers is a static class; should not be instantiated')

    @staticmethod
    def reset():
        'reset all timers'

        Timers.timers = OrderedDict()

    @staticmethod
    def tic(name):
        'start a timer'

        td = Timers.timers.get(name)

        # create timer object if it doesn't exist
        if td is None:
            parent = None if len(Timers.stack) == 0 else Timers.stack[-1]
            td = TimerData(name, parent)
            Timers.timers[name] = td

            if len(Timers.stack) > 0:
                Timers.stack[-1].children.append(td)
        else:
            # timer exists, make sure it's in the correct place in the heirarchy
            if len(Timers.stack) == 0:
                assert td.parent is None, "Timer changed in heirarchy. Currently no parent, previous was {}".format(
                    td.full_name)
            else:
                new_name = Timers.stack[-1].full_name() + "." + td.name

                assert td.parent == Timers.stack[-1], \
                    "Timer changed in heirarchy. Tried to start timer {}, previously at {}".format( \
                    new_name, td.full_name())

        Timers.timers[name].tic()
        Timers.stack.append(td)

    @staticmethod
    def toc(name):
        'stop a timer'

        assert Timers.stack[-1].name == name, "Out of order toc(). Expected to first stop timer {}".format(
            Timers.stack[-1].full_name())

        Timers.timers[name].toc()
        Timers.stack.pop()

    @staticmethod
    def print_stats():
        'print statistics about performance timers to stdout'

        for td in Timers.timers.values():
            if td.parent is None:
                Timers.print_stats_recursive(td, 0)

    @staticmethod
    def print_stats_recursive(td, level):
        'recursively print information about a timer'

        if td.last_start_time is not None:
            raise RuntimeError("Timer was never stopped: {}".format(td.name))

        if td.parent is None:
            percent_str = ""
        else:
            percent_str = " ({:.1f}%)".format(100 * td.total_secs / td.parent.total_secs)

        print "{}{} Time ({} calls): {:.2f} sec{}".format(" " * level * 2, \
            td.name.capitalize(), td.num_calls, td.total_secs, percent_str)

        total_children_secs = 0

        for child in td.children:
            total_children_secs += child.total_secs

            Timers.print_stats_recursive(child, level + 1)

        if len(td.children) > 0:
            other = td.total_secs - total_children_secs
            percent_str = " ({:.1f}%)".format(100 * other / td.total_secs)

            print "{}Other: {:.2f} sec{}".format(" " * (level + 1) * 2, \
                other, percent_str)
