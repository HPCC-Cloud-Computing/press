from autoperiod import *
from execfile import output_result

SEQUENCE = 0
PERIODGRAM = 1
ACF = 2

INTERVAL_BY_SECOND = 600
PEAK_PERCENT = 97

if __name__ == '__main__':

    temp = ex.executed_list('output_folder/wc_day6_1_600.out')
    temp += ex.executed_list('output_folder/wc_day7_1_600.out')
    temp += ex.executed_list('output_folder/wc_day8_1_600.out')
    temp += ex.executed_list('output_folder/wc_day9_1_600.out')
    temp = temp

    c = Autoperiod(temp, INTERVAL_BY_SECOND)
#    output_result(c.period_hints(PEAK_PERCENT), INTERVAL_BY_SECOND)
    c.diagram(ACF)
