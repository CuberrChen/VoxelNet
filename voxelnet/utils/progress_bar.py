import contextlib
import enum
import math
import os
import sys
import time

import numpy as np


def progress_str(val, *args, width=20, with_ptg=True):
    val = max(0., min(val, 1.))
    assert width > 1
    pos = round(width * val) - 1
    if with_ptg is True:
        log = '[{}%]'.format(max_point_str(val * 100.0, 4))
    log += '['
    for i in range(width):
        if i < pos:
            log += '='
        elif i == pos:
            log += '>'
        else:
            log += '.'
    log += ']'
    for arg in args:
        log += '[{}]'.format(arg)
    return log


def second_to_time_str(second, omit_hours_if_possible=True):
    second = int(second)
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    if omit_hours_if_possible:
        if h == 0:
            return '{:02d}:{:02d}'.format(m, s)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)


def progress_bar_iter(task_list, width=20, with_ptg=True, step_time_average=50, name=None):
    total_step = len(task_list)
    step_times = []
    start_time = 0.0
    name = '' if name is None else f"[{name}]"
    for i, task in enumerate(task_list):
        t = time.time()
        yield task
        step_times.append(time.time() - t)
        start_time += step_times[-1]
        start_time_str = second_to_time_str(start_time)
        average_step_time = np.mean(step_times[-step_time_average:]) + 1e-6
        speed_str = "{:.2f}it/s".format(1 / average_step_time)
        remain_time = (total_step - i) * average_step_time
        remain_time_str = second_to_time_str(remain_time)
        time_str = start_time_str + '>' + remain_time_str
        prog_str = progress_str(
            (i + 1) / total_step,
            speed_str,
            time_str,
            width=width,
            with_ptg=with_ptg)
        print(name + prog_str + '   ', end='\r')
    print("")


list_bar = progress_bar_iter

def enumerate_bar(task_list, width=20, with_ptg=True, step_time_average=50, name=None):
    total_step = len(task_list)
    step_times = []
    start_time = 0.0
    name = '' if name is None else f"[{name}]"
    for i, task in enumerate(task_list):
        t = time.time()
        yield i, task
        step_times.append(time.time() - t)
        start_time += step_times[-1]
        start_time_str = second_to_time_str(start_time)
        average_step_time = np.mean(step_times[-step_time_average:]) + 1e-6
        speed_str = "{:.2f}it/s".format(1 / average_step_time)
        remain_time = (total_step - i) * average_step_time
        remain_time_str = second_to_time_str(remain_time)
        time_str = start_time_str + '>' + remain_time_str
        prog_str = progress_str(
            (i + 1) / total_step,
            speed_str,
            time_str,
            width=width,
            with_ptg=with_ptg)
        print(name + prog_str + '   ', end='\r')
    print("")


def max_point_str(val, max_point):
    positive = bool(val >= 0.0)
    val = np.abs(val)
    if val == 0:
        point = 1
    else:
        point = max(int(np.log10(val)), 0) + 1
    fmt = "{:." + str(max(max_point - point, 0)) + "f}"
    if positive is True:
        return fmt.format(val)
    else:
        return fmt.format(-val)


class Unit(enum.Enum):
    Iter = 'iter'
    Byte = 'byte'


def convert_size(size_bytes):
    # from https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return s, size_name[i]


class ProgressBar:
    def __init__(self,
                 width=20,
                 with_ptg=True,
                 step_time_average=50,
                 speed_unit=Unit.Iter):
        self._width = width
        self._with_ptg = with_ptg
        self._step_time_average = step_time_average
        self._step_times = []
        self._start_time = 0.0
        self._total_size = None
        self._speed_unit = speed_unit

    def start(self, total_size):
        self._start = True
        self._step_times = []
        self._finished_sizes = []
        self._time_elapsed = 0.0
        self._current_time = time.time()
        self._total_size = total_size
        self._progress = 0

    def print_bar(self, finished_size=1, pre_string=None, post_string=None):
        self._step_times.append(time.time() - self._current_time)
        self._finished_sizes.append(finished_size)
        self._time_elapsed += self._step_times[-1]
        start_time_str = second_to_time_str(self._time_elapsed)
        time_per_size = np.array(self._step_times[-self._step_time_average:])
        time_per_size /= np.array(
            self._finished_sizes[-self._step_time_average:])
        average_step_time = np.mean(time_per_size) + 1e-6
        if self._speed_unit == Unit.Iter:
            speed_str = "{:.2f}it/s".format(1 / average_step_time)
        elif self._speed_unit == Unit.Byte:
            size, size_unit = convert_size(1 / average_step_time)
            speed_str = "{:.2f}{}/s".format(size, size_unit)
        else:
            raise ValueError("unknown speed unit")
        remain_time = (self._total_size - self._progress) * average_step_time
        remain_time_str = second_to_time_str(remain_time)
        time_str = start_time_str + '>' + remain_time_str
        prog_str = progress_str(
            (self._progress + 1) / self._total_size,
            speed_str,
            time_str,
            width=self._width,
            with_ptg=self._with_ptg)
        self._progress += finished_size
        if pre_string is not None:
            prog_str = pre_string + prog_str
        if post_string is not None:
            prog_str += post_string
        if self._progress >= self._total_size:
            print(prog_str + '   ')
        else:
            print(prog_str + '   ', end='\r')
        self._current_time = time.time()


class Progbar(object):
    """
    Displays a progress bar.
        It refers to https://github.com/keras-team/keras/blob/keras-2/keras/utils/generic_utils.py

    Args:
        target (int): Total number of steps expected, None if unknown.
        width (int): Progress bar width on screen.
        verbose (int): Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics (list|tuple): Iterable of string names of metrics that should *not* be
            averaged over time. Metrics in this list will be displayed as-is. All
            others will be averaged by the progbar before display.
        interval (float): Minimum visual progress update interval (in seconds).
        unit_name (str): Display name for step counts (usually "step" or "sample").
    """

    def __init__(self,
                 target,
                 width=30,
                 verbose=1,
                 interval=0.05,
                 stateful_metrics=None,
                 unit_name='step'):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stderr, 'isatty')
                                  and sys.stderr.isatty())
                                 or 'ipykernel' in sys.modules
                                 or 'posix' in sys.modules
                                 or 'PYCHARM_HOSTED' in os.environ)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None, finalize=None):
        """
        Updates the progress bar.

        Args:
            current (int): Index of current step.
            values (list): List of tuples: `(name, value_for_last_step)`. If `name` is in
                `stateful_metrics`, `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
            finalize (bool): Whether this is the last update for the progress bar. If
                `None`, defaults to `current >= self.target`.
        """

        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                # In the case that progress bar doesn't have a target value in the first
                # epoch, both on_batch_end and on_epoch_end will be called, which will
                # cause 'current' and 'self._seen_so_far' to have the same value. Force
                # the minimal value to 1 here, otherwise stateful_metric will be 0s.
                value_base = max(current - self._seen_so_far, 1)
                if k not in self._values:
                    self._values[k] = [v * value_base, value_base]
                else:
                    self._values[k][0] += v * value_base
                    self._values[k][1] += value_base
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stderr.write('\b' * prev_total_width)
                sys.stderr.write('\r')
            else:
                sys.stderr.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stderr.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0

            if self.target is None or finalize:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if finalize:
                info += '\n'

            sys.stderr.write(info)
            sys.stderr.flush()

        elif self.verbose == 2:
            if finalize:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stderr.write(info)
                sys.stderr.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)