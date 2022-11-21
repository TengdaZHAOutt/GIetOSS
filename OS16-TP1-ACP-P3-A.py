# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:54:18 2022

@author: StickyDa
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
#lire donne
import pandas as pd
df =pd.read_csv('filedonnees_2D.csv', sep=';',header=None)
donn = df.to_numpy()
nombre = df.columns
N=len(nombre)
# Part A
'1. Représenter les points dans l’espace original (en 2D).'
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.scatter(donn[0,:],donn[1,:])
for i in range(N) :
     plt.text(donn[0,i],donn[1,i],nombre[i],va="bottom",ha="center",fontsize=8)
plt.title('Représenter les points/partie A')
plt.show()

'2. Calculer la matrice de variance-covariance.'    
corvar=np.matmul(donn,np.transpose(donn))/N

'3. Calculer les valeurs propres et les vecteurs propres.'
vp=np.linalg.eig(corvar)
vps=vp[0]#valeur propre
vpm=vp[1]#vector propre
ordre=np.argsort(-vps)
vpm=vpm[:,ordre]
vps=vps[ordre]

'4. Calculer la projection des points sur les axes obtenus (en conservant l’ensemble des axes).'
coord=np.matmul(np.transpose(donn),np.transpose(vpm))
coord1=np.matmul(np.transpose(donn),np.transpose(vpm[:,0]))
coord2=np.matmul(np.transpose(donn),np.transpose(vpm[:,1]))
'5. Représenter les points obtenus sur le premier axe, en indiquant le nom des réalisations.'
'Remarque : Les points représentés sur 1 axe doivent être représentés sur une droite horizontale,'
'puisque la dimension obtenue vaut 1.'
null=np.zeros(N)
plt.figure()
plt.plot(coord1,null,'.', markersize=10)
for i in range(N) :
    plt.text(coord1[i],null[i],nombre[i],va="bottom",ha="center",fontsize=8)
plt.plot([-40,40],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-40,40],color='silver',linestyle='-',linewidth=1)
plt.title("en axe1")
plt.show()

plt.figure()
plt.plot(coord1,coord2,'.', markersize=10)
for i in range(N) :
    plt.text(coord1[i],coord2[i],nombre[i],va="bottom",ha="center",fontsize=8)
plt.plot([-40,40],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-40,40],color='silver',linestyle='-',linewidth=1)
plt.title("en axe1 et 2")
plt.show()

'6. Tracer le cercle des corrélations. Rappel : les coefficients de corrélation sont compris entre -1 et 1.'
r1=np.sqrt(vps[0])*vpm[:,0]/np.std(donn, axis = 1)
r2=np.sqrt(vps[1])*vpm[:,1]/np.std(donn, axis = 1)
name =['1','2'] ;
plt.figure()
theta = np.linspace(0, 2*np.pi, 100)
x1 = np.cos(theta)
x2 = np.sin(theta)
fig, ax = plt.subplots(1)
ax.plot(x1, x2)
ax.set_aspect(1)
plt.xlim(-1.05,1.05)
plt.ylim(-1.05,1.05)
plt.plot(r1,r2,'*')
for i in range(len(name)) :
    plt.text(r1[i],r2[i],name[i],va="bottom",ha="center",fontsize=8)
plt.grid(linestyle='-')
plt.title('cercle des corrélations', fontsize=8)
plt.show()


    
