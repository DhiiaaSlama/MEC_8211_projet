# -*- coding: utf-8 -*-
"""
MEC6616 - TPP2 : Equation de convection-diffusion 2D - Maillages de triangles vs quadrilatères

@authors: Zacharie Prigent, Matricule : 2096785 || Mohamed Dhia Slama, Matricule : 2115178
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Classe de calcul du terme source MMS
    
class Source: 
    def __init__(self, T_MMS,rho,Cp,k,u,v) :
        self.solution_analytique = T_MMS
        self.rho = rho
        self.Cp = Cp
        self.k = k
        self.u = u
        self.v = v
        
        
    def terme_source(self):
        x,y = sp.symbols('x y')

        #Application de l'opérateur sur la solution MMS
        source = sp.diff(self.rho*self.Cp*self.u*self.solution_analytique,x) + sp.diff(self.rho*self.Cp*self.v*self.solution_analytique,y) - self.k*(sp.diff(sp.diff(self.solution_analytique,x),x)+sp.diff(sp.diff(self.solution_analytique,y),y))
        #Création d'une fonction symbolique
        f_source_MMS = sp.lambdify([x,y], source, "numpy")
        
        return f_source_MMS

class Traitement: 
    def __init__(self, T_MMS,u,v,domaine) :
        self.solution_analytique = T_MMS
        self.u = u
        self.v = v
        self.domaine = domaine           
            
    def vitesse_moyenne(self):
        #Création des fonctions symboliques
        x,y = sp.symbols('x y')
        u = sp.lambdify([x,y], self.u, "numpy")
        v = sp.lambdify([x,y], self.v, "numpy")
        
        #Taille du domaine
        xmin = self.domaine[0]
        xmax = self.domaine[1]
        ymin = self.domaine[2]
        ymax = self.domaine[3]
        
        #Maillage    
        nb_pts_discretisation_xy=10
        xdom = np.linspace(xmin,xmax,nb_pts_discretisation_xy) 
        ydom = np.linspace(ymin,ymax,nb_pts_discretisation_xy)
        nb_pts=nb_pts_discretisation_xy**2
        
        #Vitesse en chaque point du maillage
        vitesse=[]
        kk = 0 
        for i in range(nb_pts_discretisation_xy):
            for j in range(nb_pts_discretisation_xy):
                composante_u = u(xdom[j],ydom[i])
                composante_v = v(xdom[j],ydom[i])
                vitesse+=[np.sqrt(composante_u**2+composante_v**2)]
                kk = max(kk,u(xdom[j],ydom[i]))
        
        #Vitesse moyenne
        vitesse=np.array(vitesse)
        V=np.sum(vitesse)/nb_pts

        return V
    
    def graphe_analytique(self): #Contour de la solution analytique MMS avec matplotlib
        #Création d'une fonction symbolique
        x,y = sp.symbols('x y')
        T_MMS = sp.lambdify([x,y], self.solution_analytique, "numpy")
        
        #Taille du domaine
        xmin = self.domaine[0]
        xmax = self.domaine[1]
        ymin = self.domaine[2]
        ymax = self.domaine[3]
        
        #Maillage    
        xdom = np.linspace(xmin,xmax,200) 
        ydom = np.linspace(ymin,ymax,200)
        xi, yi = np.meshgrid(xdom, ydom)
        
        #Evalution de la fonction analytique sur le maillage
        z_MMS=T_MMS(xi,yi)
        
        #Affichage
        ax = plt.subplot(111)
        im = ax.contourf(xi,yi,z_MMS, cmap='jet')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label("T [K]")
        ax.set_title('Solution analytique MMS',fontsize=11,color="black")
        plt.show()
            
        return
            
    def graphe_champ_vitesse(self):
        #Création des fonctions symboliques
        x,y = sp.symbols('x y')
        u = sp.lambdify([x,y], self.u, "numpy")
        v = sp.lambdify([x,y], self.v, "numpy")
        
        #Taille du domaine
        xmin = self.domaine[0]
        xmax = self.domaine[1]
        ymin = self.domaine[2]
        ymax = self.domaine[3]
        
        #Maillage    
        xdom = np.linspace(xmin,xmax,20) 
        ydom = np.linspace(ymin,ymax,20)
        xi, yi = np.meshgrid(xdom, ydom)
        
        #Evalution de la fonction analytique sur le maillage
        champ_u = u(xi,yi)
        champ_v = v(xi,yi)

        #Affichage
        ax = plt.subplot(111)
        ax.quiver(xi, yi, champ_u, champ_v, color='b', units='xy')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title('Champ de vitesse',fontsize=11,color="black")
        plt.show()
        
        return