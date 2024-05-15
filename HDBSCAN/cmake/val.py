with open("r1.txt") as l:
 	f1 = l.readlines()

f1 = f1[13:]

lista_f1 = []
lista_aux = []
for valor in f1[1:]:
	if (valor == '-\n'):
		lista_f1.append(lista_aux)
		lista_aux = []
	else:
		lista_aux.append(valor)

with open("r2.txt") as o:
     f2 = o.readlines()
     
f2 = f2[13:]

lista_f2 = []
lista_aux = []

print(len(lista_f2[61]))
print(len(lista_f1[61]))
"""for valor in f2[1:]:
	if (valor == '-\n'):
		lista_f2.append(lista_aux)
		lista_aux = []
	else:
		lista_aux.append(valor)


for index,i in enumerate(lista_f1):

	if len(set(i).intersection(lista_f2[index])) != len(set(lista_f2[index])):
		print(index)
		break


 
"""
