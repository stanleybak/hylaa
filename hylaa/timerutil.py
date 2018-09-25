'''
Timer utility functions for Hylaa. Timers are used for performance analysis and
can be refered to statically using Timers.tic(name) and Timers.toc(name)

Stanley Bak
September 2016
'''

import time

from termcolor import cprint

class TimerData():
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

    def get_children_recursive(self, name):
        'get all decendants with the given name (returns a list of TimerData)'

        rv = []

        if name == self.name:
            rv.append(self)

        for child in self.children:
            rv += child.get_children_recursive(name)

        return rv

    def full_name(self):
        'get the full name of the timer (including ancestors)'

        if self.parent is None:
            return self.name

        return "{}.{}".format(self.parent.full_name(), self.name)

    def tic(self):
        'start the timer'

        #print "Tic({})".format(self.name)

        if self.last_start_time is not None:
            raise RuntimeError("Timer started twice: {}".format(self.name))

        self.num_calls += 1
        self.last_start_time = time.perf_counter()

    def toc(self):
        'stop the timer'

        #print "Toc({})".format(self.name)

        if self.last_start_time is None:
            raise RuntimeError("Timer stopped without being started: {}".format(self.name))

        self.total_secs += time.perf_counter() - self.last_start_time
        self.last_start_time = None

class Timers():
    '''
    a static class for doing timer messuarements. Use
    Timers.tic(name) and Timers.tic(name) to start and stop timers, use
    print_stats to print time statistics
    '''

    top_level_timer = None

    stack = [] # stack of currently-running timers, parents at the start, children at the end

    def __init__(self):
        raise RuntimeError('Timers is a static class; should not be instantiated')

    @staticmethod
    def reset():
        'reset all timers'

        Timers.top_level_timer = None
        Timers.stack = []

    @staticmethod
    def tic(name):
        'start a timer'

        #print("Tic({})".format(name))

        if not Timers.stack:
            top = Timers.top_level_timer

            if top is not None and top.name != name:
                # overwrite old top level timer
                #print("Overwriting old top-level timer {} with new top-level timer {}".format(top.name, name))
                top = Timers.top_level_timer = None

            td = top
        else:
            td = Timers.stack[-1].get_child(name)

        # create timer object if it doesn't exist
        if td is None:
            parent = None if not Timers.stack else Timers.stack[-1]
            td = TimerData(name, parent)

            if not Timers.stack:
                Timers.top_level_timer = td
            else:
                Timers.stack[-1].children.append(td)

        td.tic()
        Timers.stack.append(td)

    @staticmethod
    def toc(name):
        'stop a timer'

        #print("Toc({})".format(name))

        assert Timers.stack[-1].name == name, "Out of order toc(). Expected to first stop timer {}".format(
            Timers.stack[-1].full_name())

        Timers.stack[-1].toc()
        Timers.stack.pop()

    @staticmethod
    def print_stats():
        'print statistics about performance timers to stdout'

        Timers.print_stats_recursive(Timers.top_level_timer, 0, None)

    @staticmethod
    def print_stats_recursive(td, level, total_time):
        'recursively print information about a timer'

        low_threshold = 5.0
        high_threshold = 50.0

        if level == 0:
            total_time = td.total_secs

        percent_total = 100 * td.total_secs / total_time

        if percent_total < low_threshold:
            def print_func(text):
                'below threshold print function'

                return cprint(text, 'grey')
        elif percent_total > high_threshold:
            def print_func(text):
                'above threshold print function'

                return cprint(text, None, attrs=['bold'])
        else:
            def print_func(text):
                'within threshold print function'

                return cprint(text, None)

        if td.last_start_time is not None:
            raise RuntimeError("Timer was never stopped: {}".format(td.name))

        if td.parent is None:
            percent = 100
            percent_str = ""
        else:
            percent = 100 * td.total_secs / td.parent.total_secs
            percent_str = " ({:.1f}%)".format(percent)

        print_func("{}{} Time ({} calls): {:.2f} sec{}".format(" " * level * 2, \
            td.name.capitalize(), td.num_calls, td.total_secs, percent_str))

        total_children_secs = 0

        for child in td.children:
            total_children_secs += child.total_secs

            Timers.print_stats_recursive(child, level + 1, total_time)

        if td.children:
            other = td.total_secs - total_children_secs
            other_percent = 100 * other / td.total_secs

            percent_other = other / total_time

            if percent_other < low_threshold:
                def other_print_func(text):
                    'below threshold print function'

                    return cprint(text, 'grey')
            elif percent_other > high_threshold:
                def other_print_func(text):
                    'above threshold print function'

                    return cprint(text, None, attrs=['bold'])
            else:
                def other_print_func(text):
                    'within threshold print function'

                    return cprint(text, None)

            percent_str = " ({:.1f}%)".format(other_percent)

            other_print_func("{}Other: {:.2f} sec{}".format(" " * (level + 1) * 2, \
                other, percent_str))
