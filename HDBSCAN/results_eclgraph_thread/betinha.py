import os
def read_folder(path):

	files = [f for f in os.listdir(path)]
	
	try:
		file.remove("betinha.py")
	except:
		print("Não tem betinha")
	d = {}
	for i in files:
		key = i.split("_",2)[-1].split(".")[0]
		if key not in d:
			d[key] = [i]
		else:
			d[key].append(i)
			
	time_dic = {}
	for key in d:
		lista = []
		for file in d[key]:
			with open(f"{path}/{file}",'r') as f:
				file_lines = f.readlines()
			lista.append(float(file_lines[8].split(" ")[1]))
		media = sum(lista) / len(lista)
		time_dic[key] = media
	return time_dic
	
	
final_dic = {}

folders = os.listdir()
folders.remove("betinha.py")

for folder in folders:
	key = int(folder.split("M",1)[0])
	final_dic[key] = {}
	
	aux_dic = read_folder(folder)
	
	for bucket in [128,256,512,1024]:
		final_dic[key][bucket] = []
		for thread in [32,64,128,256,512]:
			final_dic[key][bucket].append(aux_dic[f"{bucket}_{thread}"])
	
	
for key in final_dic:
	print(f"RESULTADOS PARA {key}")
	print(final_dic[key])

	print("\n")
