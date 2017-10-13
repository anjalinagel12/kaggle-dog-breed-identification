#!/bin/python

def init(log_dir):
    global target_str
    global speed_pattern
    global model_name_pattern
    global output_header

    target_str = "Saved"
    speed_pattern = ".*Speed: (.*) samples/sec.*"
    model_name_pattern = ".*Saved checkpoint to.*pretrained-models/(.*)-0001.params.*"
    output_header = "model-name\tave-speed(samples/sec)\tspeed-var\thi-speed\tlo-speed\tspeed-num"


def run(log_dir):
    speed_list = []

    print(output_header)

    with open(log_dir) as log_file_handle:
        line_list = log_file_handle.readlines()
        for l_idx in xrange(len(line_list)):
            line = line_list[l_idx]
            if (line.find(target_str) > -1):
                if (len(speed_list) <= 0):
                # useless log including model name
                    continue
                # speed-crawler stop
                else:
                    import re
                    model_name = re.findall(model_name_pattern, line)[0]
                    model_name = model_name[model_name.find("/")+1:]
		    #print(speed_list)
                    speed_list = map(float, speed_list)
                    speed_list.sort()
                    ave_speed = sum(speed_list)/len(speed_list)
                    speed_var = variance(speed_list, ave_speed)
                    hi_speed = speed_list[-1]
                    lo_speed = speed_list[0]
                    print("%s\t%6.2f\t%6.2f\t%6.2f\t%6.2f\t%d" % (model_name, ave_speed, speed_var, hi_speed, lo_speed, len(speed_list)))
                    # init
                    speed_list = []
                    
            elif line.find("Speed") > -1:
                import re
                speed = re.findall(speed_pattern, line)[0]
                speed_list.append(speed)
            else:
                # dont include speed or model name
                pass
                

def variance(speed_list, ave_speed):
    diff_speed_2_list = map(lambda x: (x-ave_speed)**2, speed_list)
    var = reduce(lambda x, y: x+y, diff_speed_2_list) / len(diff_speed_2_list)
    return var

if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1]
    init(log_dir)
    run(log_dir)
