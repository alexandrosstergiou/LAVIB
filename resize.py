import os, sys
from tqdm import tqdm
from glob import glob
import pandas as pd
from multiprocessing import Pool


def worker(vid_f):
    if not vid_f[2]:
        os.system(f"ffmpeg -y -hide_banner -loglevel error -i {vid_f[0]} -s 540x540 -c:a copy {vid_f[1]}/vid.mp4")
    else:
        os.system(f"ffmpeg -n -hide_banner -loglevel error -i {vid_f[0]} {vid_f[1]}/vid.mp4")
        


d = 'data/segments/*/vid.mp4'

df_val = pd.read_csv('data/annotations/val.csv')
df_test = pd.read_csv('data/annotations/test.csv')

vids = [f"{int(row['name'])}_shot{int(row['shot'])}_{int(row['tmp_crop'])}_{int(row['vrt_crop'])}_{int(row['hrz_crop'])}" for _,row in df_val.iterrows()] + [f"{int(row['name'])}_shot{int(row['shot'])}_{int(row['tmp_crop'])}_{int(row['vrt_crop'])}_{int(row['hrz_crop'])}" for _,row in df_test.iterrows()]

files = glob(d)

fs = []
count=0
for vid in tqdm(files):
    filename = vid.split("/")[-2]
    folder = f'data/segments_downsampled/{filename}'
    
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
    is_eval = filename in vids
    if is_eval:
        count+=1
    fs.append((vid,folder,is_eval))
 
print(count,len(vids))


try:
    with Pool(4) as p:
        for _ in tqdm(p.imap_unordered(worker, fs), total=len(fs)):
            pass

except KeyboardInterrupt:
    print("Caught KeyboardInterrupt, terminating")
    p.terminate()
    p.join()


    
