import os
def read_folder(path):

	files = [f for f in os.listdir(path)]
	
	lista = []
	for file in files:
		with open(f"{path}/{file}",'r') as f:
			file_lines = f.readlines()
		lista.append(float(file_lines[8].split(" ")[1]))
	media = sum(lista) / len(lista)
	return media
	
	
final_dic = []

folders = os.listdir()
folders.remove("betinha.py")

for folder in folders:
	
	aux_dic = read_folder(folder)
	
	final_dic.append(aux_dic)
	

print(final_dic)
