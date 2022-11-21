# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:54:18 2022

@author: StickyDa
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
df =pd.read_csv('temp_et_coord_14villes.csv', sep=';')
donn = df.to_numpy()
nomvilles = df.columns


"1. Représenter les villes dans l’espace longitude-latitude (figure 1)."
import matplotlib.pyplot as plt
import numpy as np
N=nomvilles.size
colors = np.random.rand(N)
color = list(np.random.choice(range(256), size=3))
plt.scatter(donn[12], donn[13],c=colors)
for i in range(N):
    plt.annotate(nomvilles[i],(donn[12,i],donn[13,i]))
    
plt.title('longitude-latitude(figure 1)')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()

"2. Représenter les 14 courbes de température en fonction du mois (figure 2)"
X=[1,2,3,4,5,6,7,8,9,10,11,12];
for i in range(N):
    Y=np.transpose(donn[0:12,i])
    plt.plot(X,Y,label=nomvilles[i])  
plt.title('Les 14 courbes de température(figure 2)')
plt.xlabel('Mois')
plt.ylabel('Température')
plt.legend()
plt.show()


"3. Centrer et réduire les données, sans utiliser de fonctions prédéfinies"
donn1 = np.zeros(shape=(12,14))
moy=np.zeros(12)
ecart=np.zeros(12)
for i in range(12):
    moy[i]=np.mean(donn[i,])
    ecart[i]=np.std(donn[i,])
for i in range(12):
    for j in range(N):
        donn1[i,j]=donn[i,j]-moy[i]
        donn1[i,j]=donn1[i,j]/ecart[i]


"4. Représenter les 14 courbes de température des données centrées-réduites en fonction du mois (figure 3)."
X=[1,2,3,4,5,6,7,8,9,10,11,12];
for i in range(N):
    Y=np.transpose(donn1[0:12,i])
    plt.plot(X,Y,label=nomvilles[i])  
plt.title('Les 14 courbes de température(figure 3)')
plt.xlabel('Mois')
plt.ylabel('Température')
plt.legend()
plt.show()

"5. Calculer la matrice de variance covariance.."
corvar=np.matmul(donn1,np.transpose(donn1))/N

"6. Calculer les valeurs propres et les vecteurs propres."
vp=np.linalg.eig(corvar)
vps=vp[0]#valeur propre
vpm=vp[1]#vector propre
ordre=np.argsort(-vps)
vpm=vpm[:,ordre]
vps=vps[ordre]

"7. Tracer la courbe de pourcentage d’inertie expliquée par chaque axe. (figure 4)"
X=[1,2,3,4,5,6,7,8,9,10,11,12];
plt.bar(X,vps/sum(vps))
plt.title("pourcentage d'inertie(figure 4)")
plt.show()

"8. Calculer la projection des points sur les axes obtenus (en conservant l’ensemble des axes)."
coord=np.matmul(np.transpose(donn1),vpm).T

"9. Représenter les points obtenus sur les 2 premiers axes, en indiquant le nom des villes. (figure 5)"
coord1=np.matmul(np.transpose(donn1),np.transpose(vpm[:,0]))
coord2=np.matmul(np.transpose(donn1),np.transpose(vpm[:,1]))
plt.figure()
plt.plot(coord1,coord2,'.', markersize=10)
for i in range(len(nomvilles)) :
    plt.text(coord1[i],coord2[i],nomvilles[i],va="bottom",ha="center",fontsize=8)
plt.plot([-5,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-4.5,2.5],color='silver',linestyle='-',linewidth=1)
plt.title("sur les 2 premiers axes(figure 5)")
plt.show()

"10. Calculer le coefficient de corrélation de la première composante avec chacun des 12 mois (vecteur"
"r1), et faire la même chose pour la seconde composante (vecteur r2)."
r1=np.sqrt(vps[0])*vpm[:,0]
r2=np.sqrt(vps[1])*vpm[:,1]

"11. Tracer le cercle des corrélations (cf code Matlab ci-dessous)."
moisname =['J', 'F','M','A','Mai','J','Jt','A', 'S', 'O', 'N', 'D'] ;
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
for i in range(len(moisname)) :
    plt.text(r1[i],r2[i],moisname[i],va="bottom",ha="center",fontsize=8)
plt.grid(linestyle='-')
plt.title('cercle des corrélations', fontsize=8)
plt.show()

##################################3
#Question 1
print((vps[0]+vps[1])/sum(vps))
#Question 2
null=np.zeros(N)
plt.figure()
plt.plot(coord1,null,'.', markersize=10)
for i in range(N) :
    plt.text(coord1[i],null[i],nomvilles[i],va="bottom",ha="center",fontsize=8)
plt.plot([-5,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
plt.show()
#Question 4
null=np.zeros(N)
plt.figure()
plt.plot(null,coord2,'.', markersize=10)
for i in range(N) :
    plt.text(null[i],coord2[i],nomvilles[i],va="bottom",ha="center",fontsize=8)
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-4.5,2.5],color='silver',linestyle='-',linewidth=1)
plt.show()
