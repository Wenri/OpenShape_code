import os
from multiprocessing import Pool
from huggingface_hub import hf_hub_download



def single_uncompress(file, dest="/mnt/disk1/souhail_data/openshape"):
	print(file)
	hf_hub_download(repo_id="OpenShape/openshape-training-data", filename=file, repo_type="dataset", local_dir=dest)
	if file.endswith(".tar.gz"):
		os.system("tar -xzf %s/%s -C %s" % (dest, file, dest))
	elif file.endswith(".zip"):
		os.system("unzip %s/%s -d %s" % (dest, file, dest))
	else:
		print("File extension not supported.")
	os.system("rm %s/%s" % (dest, file))


NUM_PROC = 32
pool = Pool(NUM_PROC)

# pool.apply_async(single_uncompress, ("meta_data.zip", "./"))
files = ["3D-FUTURE.tar.gz", "ABO.tar.gz", "ShapeNet.tar.gz"]
for file in files:
	pool.apply_async(single_uncompress, (file,"/mnt/disk1/souhail_data/openshape"))

for i in range(160):
	pool.apply_async(single_uncompress, ("Objaverse/000-%03d.tar.gz" % i,"/mnt/disk1/souhail_data/openshape"))
	
pool.close()
pool.join()