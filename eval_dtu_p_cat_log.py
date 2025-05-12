'''
cmd1:
CUDA_VISIBLE_DEVICES=5 python mesh_extract.py -s /home/haochen/data/DTU/scan40/ -m /home/haochen/project/gaussian-splatting-ori/gaussian-splatting/output/5dd846bf-d -r 2
CUDA_VISIBLE_DEVICES=5 python evaluate_dtu_mesh.py -s /home/haochen/data/DTU/scan40/ -m exp_dtu/ori_erank/scan40 --eval_other_model /home/haochen/project/gaussian-splatting-ori/gaussian-splatting/output/5dd846bf-d/
'''

import os
import re
import time
from contextlib import contextmanager
import GPUtil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser


folder = 'erank_full_test_r2'

parser = ArgumentParser(description="Training script parameters")
parser.add_argument('--folder', type=str, default=folder, help='Folder to read logs from')


args = parser.parse_args()
folder = args.folder

scenes = ['scan105','scan106','scan110','scan114','scan118','scan122','scan24','scan37','scan40','scan55','scan63','scan65','scan69','scan83','scan97']
# scenes = ['scan105']

# gpus = [2,3,4,5,6]
all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, excludeID=[0,1,2,3], maxMemory=0.2))
gpus = list(all_available_gpus)
log_id_gpu = {scenes[i]: gpus[i % len(gpus)] for i in range(len(scenes))}


def worker(cmd1,cmd2):
    print(cmd1)
    subprocess.run(cmd1, shell=True)
    print(cmd2)
    subprocess.run(cmd2, shell=True)

def process_log(scene):
        gpu = log_id_gpu[scene]
        cmd1 = f"CUDA_VISIBLE_DEVICES={gpu} python extract_mesh_tsdf.py  -m dtu_out/{scene}/{folder} --iteration 30000" #-r 2
        cmd2 = f"CUDA_VISIBLE_DEVICES={gpu} python evaluate_dtu_mesh.py  -m dtu_out/{scene}/{folder} "
        worker(cmd1, cmd2)
        time.sleep(1)

if __name__ == '__main__':
    max_workers = len(gpus)  # 根据实际需求调整并行线程数
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_log, scenes)

    print("All tasks finished!")

    print(f'python cat_log.py --folder {folder}')

    os.system(f'python cat_log.py --folder {folder}')

