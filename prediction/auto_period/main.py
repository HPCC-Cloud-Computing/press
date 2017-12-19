from autoperiod import *
from execfile import get_data

SEQUENCE = 0
PERIODGRAM = 1
ACF = 2

INTERVAL_BY_SECOND = 600
PEAK_PERCENT = 99
START_DAY = 6
END_DAY = 9


if __name__ == '__main__':

    data = get_data(START_DAY, END_DAY, INTERVAL_BY_SECOND)
    c = Autoperiod(data, INTERVAL_BY_SECOND, PEAK_PERCENT)
    print(c.period_hints())
    c.diagram(ACF)
