# import os
# import sys
# import time
#
# cmd = '/home/cls2020/wqj/anaconda3/envs/pytorch37/bin/python3.7 ./main.py'
#
# def gpu_info():
#     gpu_status_list = os.popen('nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits').read().split('\n')[:-1]
#     gpu_info_list = []
#     for gpu_status in gpu_status_list:
#         gpu_memory, gpu_power = map(int, gpu_status.split(', '))
#         gpu_info_list.append((gpu_power, gpu_memory))
#     return gpu_info_list
#
# def narrow_setup(interval=2):
#     gpu_info_list = gpu_info()
#     i = 0
#     while not all(gpu_memory <= 9000 and gpu_power <= 180 for gpu_power, gpu_memory in gpu_info_list):  # set waiting condition
#         gpu_info_list = gpu_info()
#         i = i % 5
#         symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
#         gpu_status_str = ' | '.join([f'gpu {idx}: power {gpu_power}W, memory {gpu_memory}MiB' for idx, (gpu_power, gpu_memory) in enumerate(gpu_info_list)])
#         sys.stdout.write('\r' + gpu_status_str + ' ' + symbol)
#         sys.stdout.flush()
#         time.sleep(interval)
#         i += 1
#     print('\n' + cmd)
#     os.system(cmd)
#
# if __name__ == '__main__':
#     narrow_setup()

# author: muzhan
import os
import sys
import time

# cmd = 'python main.py'
cmd = '/home/cls2020/wqj/anaconda3/envs/pytorch37/bin/python3.7 ./main.py'


def gpu_info(gpu_index=1):
    info = os.popen('nvidia-smi|grep %').read().split('\n')[gpu_index].split('|')
    power = int(info[1].split()[-3][:-1])
    memory = int(info[2].split('/')[0].strip()[:-3])
    return power, memory
    # gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    # gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    # gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    # return gpu_power, gpu_memory


def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory > 1000 or gpu_power > 100:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()