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

    def get_child(self, name):
        'get a child timer with the given name'

        rv = None

        for child in self.children:
            if child.name == name:
                rv = child
                break

        return rv

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

    # map of timer_name -> TimerData
    top_level_timers = OrderedDict()

    stack = [] # stack of currently-running timers, parents at the start, children at the end

    def __init__(self):
        raise RuntimeError('Timers is a static class; should not be instantiated')

    @staticmethod
    def reset():
        'reset all timers'

        Timers.top_level_timers = OrderedDict()
        Timers.stack = []

    @staticmethod
    def tic(name):
        'start a timer'

        if len(Timers.stack) == 0:
            td = Timers.top_level_timers.get(name)
        else:
            td = Timers.stack[-1].get_child(name)

        # create timer object if it doesn't exist
        if td is None:
            parent = None if len(Timers.stack) == 0 else Timers.stack[-1]
            td = TimerData(name, parent)

            if len(Timers.stack) == 0:
                Timers.top_level_timers[name] = td
            else:
                Timers.stack[-1].children.append(td)

        td.tic()
        Timers.stack.append(td)

    @staticmethod
    def toc(name):
        'stop a timer'

        assert Timers.stack[-1].name == name, "Out of order toc(). Expected to first stop timer {}".format(
            Timers.stack[-1].full_name())

        Timers.stack[-1].toc()
        Timers.stack.pop()

    @staticmethod
    def print_stats():
        'print statistics about performance timers to stdout'

        for td in Timers.top_level_timers.values():
            Timers.print_stats_recursive(td, 0)

    @staticmethod
    def print_stats_recursive(td, level):
        'recursively print information about a timer'

        percent_threshold = 0.0

        if td.last_start_time is not None:
            raise RuntimeError("Timer was never stopped: {}".format(td.name))

        if td.parent is None:
            percent = 100
            percent_str = ""
        else:
            percent = 100 * td.total_secs / td.parent.total_secs
            percent_str = " ({:.1f}%)".format(percent)

        if percent >= percent_threshold:
            print "{}{} Time ({} calls): {:.2f} sec{}".format(" " * level * 2, \
                td.name.capitalize(), td.num_calls, td.total_secs, percent_str)

        total_children_secs = 0

        for child in td.children:
            total_children_secs += child.total_secs

            Timers.print_stats_recursive(child, level + 1)

        if len(td.children) > 0:
            other = td.total_secs - total_children_secs
            other_percent = 100 * other / td.total_secs

            if other_percent >= percent_threshold:
                percent_str = " ({:.1f}%)".format(other_percent)

                print "{}Other: {:.2f} sec{}".format(" " * (level + 1) * 2, \
                    other, percent_str)
