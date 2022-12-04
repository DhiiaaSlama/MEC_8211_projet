# -*- coding: utf-8 -*-
"""
MEC6616 - TPP2 : Equation de convection-diffusion 2D - Maillages de triangles vs quadrilatères
@authors: Zacharie Prigent, Matricule : 2096785 || Mohamed Dhia Slama, Matricule : 2115178

Adapted and modified for: MEC8211 : Vérification et Validation en Modélisation Numérique 
Adapted and modified by : Houssem Eddine Youness || Elissa El-Hajj || Mohamed Dhia Slama 
Polytechnique Montreal  - University of Montreal

"""

import numpy as np
import sympy as sp
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from mesh import Mesh
from meshPlotter import MeshPlotter
import solveur as solveur
import post_processing as post_processing
import traitement_fonctionnel as traitement_fonctionnel
import  propagation_incertitude as propagation_incertitude


def mesh_creation(Type_Maillage,nb_raf,index_raf,Nx,Ny,lc,domaine):
    #Création du maillage
    print("Création du maillage:")
    if (Type_Maillage == 'QUAD') :    
        mesher = MeshGenerator()
        mesh_parameters = {'mesh_type': 'QUAD',
                           'Nx': Nx,
                           'Ny': Ny
                           }
        mesh_obj = mesher.rectangle(domaine, mesh_parameters)
        conec = MeshConnectivity(mesh_obj)
        conec.compute_connectivity()
    elif  (Type_Maillage == 'TRI'): 
        mesher = MeshGenerator()
        mesh_parameters = {'mesh_type': 'TRI',
                            'lc': lc
                            }
        
        mesh_obj = mesher.rectangle(domaine, mesh_parameters)
        conec = MeshConnectivity(mesh_obj)
        conec.compute_connectivity()
        
    # if (index_raf==nb_raf-1): 
    # plotter = MeshPlotter()
    # plotter.plot_mesh(mesh_obj, label_points=True, label_elements=True, label_faces=False)
        
    return  mesh_obj


def meshing_check(mesh_obj: Mesh):
    SOLVER = solveur.mesh_check(mesh_obj)
    print("\nTests sur le maillage:")
    #Vérfication avec la méthode d'Euler 
    Result=SOLVER.euler_check()
    print(Result)
    #Vérfication avec la éethode de divergence avec d'un champ constant 
    Somme = SOLVER.divergence_verification()
    print(Somme + "\n")
    
def CLs() :
    #Types de conditions aux frontières
    print("\nConditions limites: ")
    bc1 = int(input("CL à imposer pour l'arête de gauche : 1 pour DIRICHLET et 2 pour NEUMANN : "))
    bc2 = int(input("CL à imposer pour l'arête du bas : 1 pour DIRICHLET et 2 pour NEUMANN :"))
    bc3 = int(input("CL à imposer pour l'arête de droite : 1 pour DIRICHLET et 2 pour NEUMANN :"))
    bc4 = int(input("CL à imposer pour l'arête du haut : 1 pour DIRICHLET et 2 pour NEUMANN :"))
    print("\n")
    
    if bc1 == 1 : 
        bc1 = 'DIRICHLET'
    elif bc1 == 2 : 
        bc1 = 'NEUMANN'
    if bc2 == 1 : 
        bc2 = 'DIRICHLET'
    elif bc2 == 2 : 
        bc2 = 'NEUMANN'
    if bc3 == 1 : 
        bc3 = 'DIRICHLET'
    elif bc3 == 2 : 
        bc3 = 'NEUMANN'    
    if bc4 == 1 : 
        bc4 = 'DIRICHLET'
    elif bc4 == 2 : 
        bc4 = 'NEUMANN'   
        
    return bc1,bc2,bc3,bc4


print("*" * 80)
print("Projet MEC8211 ")
print("*" * 80)

#DONNEES DE L'ETUDE
#Solution analytique
x,y = sp.symbols('x y')
T_MMS = 400.+50.*sp.cos(np.pi*x)+100.*sp.sin(np.pi*x*y)

#Domaine : coordonnées en x et y
domaine=np.array([-1.,1.,-1.,1.])

#Volume du domaine
Volume_domain = (domaine[1]-domaine[0])*(domaine[3]-domaine[2])

#Champ de vitesse
u = (2. * x**2 - x**4 - 1.) * (y - y**3)
v = - (2. * y**2 - y**4 - 1.) * (x - x**3)

# u = x*0 +y*0 + 0.37466268336675396
# v = x*0 +y*0 

#Appel de la fonction de traitement fonctionnel
trait = traitement_fonctionnel.Traitement(T_MMS,u,v,domaine)

#Vitesse moyenne sur le domaine (pour l'analyse selon la valeur du nombre de Péclet Pe)
V = trait.vitesse_moyenne()

#Tracés des contours de la solution analytique T_MMS et du champ de vitesse
trait.graphe_analytique()
trait.graphe_champ_vitesse()

#Types de conditions aux frontières
bc1,bc2,bc3,bc4 = CLs()
# bc_types = np.array(['NEUMANN','NEU1MANN','NEUMANN','NEUMANN'])
# bc_types = np.array(['DIRICHLET','DIRICHLET','DIRICHLET','DIRICHLET'])
bc_types = np.array([bc1,bc2,bc3,bc4])
type_CLs = ["variable","variable","variable","variable"]
bc_values = [[1.,0.],[0.,0.],[0.,0.],[0.,0.]]

#Propriétés du fluide
rho = 1. #Masse volumique
Cp = 1. #Chaleur spécifique
Peclet= np.array([1,100,10000])                                          
#Peclet= np.array([10000]) #Nombre de Péclet
K=[] #Stockage des conductivités thermiques

#Type de schéma convectif
schema_conv = np.array(['centré','upwind'])

#Nombre détapes de raffinement
nb_raf = 4

#Choix de la norme d'erreur a calculer "L1" ou "L2" ou "Li"
error_type = "L2"

#Longueur caracteristique du domaine
Lc = np.amax([domaine[1]-domaine[0],domaine[3]-domaine[2]])
###################################################################################################################
########################################  Verification de code  ############################################
####################################################################################################################
#BOUCLE POUR LES DIFFERENTS NOMBRE DE PECLET 
for Index_Peclet in range(len(Peclet)):
    print("*" * 80)
    print("*" * 80 + "\n")
    print("NOMBRE DE PECLET PE = %f:\n" %Peclet[Index_Peclet])
    print("*" * 80 + "\n")
    
    #Conductivité thermique 
    k = V*Lc*Cp*rho/Peclet[Index_Peclet]
    K+=[k]
    
    #Terme source
    source = traitement_fonctionnel.Source(T_MMS,rho,Cp,k,u,v)
    q = source.terme_source()
    
    #BOUCLE DE CALCUL NUMERIQUE POUR LES 2 TYPES DE MAILLAGE :
    #Transfini de quadrilatère et non-structuré de triangles    
    for Type_maillage_index in range(2): 
        if Type_maillage_index == 0 : 
            Type_Maillage = 'QUAD'
            print("MAILLAGE TRANSFINI DE QUADRILATERES:\n")
        else:
            Type_Maillage = 'TRI'
            print("MAILLAGE NON STRUCTURE DE TRIANGLES:\n")
    
        #Configuration initiale des maillages
        Nx = 10 #Nombres de mailles en pour le premier pas de raffinement
        Ny = 10 #Nombres de mailles en y pour le premier pas de raffinement
        lc = 0.3 #Taille de maille pour le premier pas de raffinement
    
        #Initialisation de des vecteur Erreur et ordre 
        L_error_norm = np.zeros((nb_raf,2))
        Order_P = np.zeros((1,2))                
      
        #Initialisation du vecteur Mesh_size        
        Mesh_size = np.zeros((nb_raf,2)) #Taille moyenne d'un élément h et nb d'éléments par étape de raffinement
        
        #Boucle de raffinement          
        for index_raf in range(nb_raf):
            #Creation du maillage 
            mesh_obj = mesh_creation(Type_Maillage,nb_raf,index_raf,Nx,Ny,lc,domaine)

            #Vérification du maillage avec les tests d'Euler et de la divergence 
            meshing_check(mesh_obj)

            #Calcul de la solution numérique
            Resolution = solveur.resolution(mesh_obj,k,rho,Cp,u,v,bc_types,T_MMS,q,type_CLs,bc_values)
            Sol_numerique,areas = Resolution.sol_num()
        
            #Post processing : A ADAPTER AVEC LES NOUVELLES VARIABLES ET NOTATIONS
            Post_processing = post_processing.post_processing(mesh_obj, nb_raf, Sol_numerique,T_MMS,index_raf,L_error_norm,Order_P,Volume_domain,Mesh_size,areas,lc,domaine,error_type,Type_maillage_index,Index_Peclet,Peclet)
            Post_processing.main_post_processing()
                
            #Réduction de la taille du maillage pour le pas suivant
            Nx = Nx *2
            Ny = Ny *2
            lc = lc/2

####################################################################################################################
########################################  Propagation des incertitudes  ############################################
####################################################################################################################
#Solution analytique
x,y = sp.symbols('x y')
T_analytique = (1-x**2)*(1-y**2)
# T_analytique = 400.+50.*sp.cos(np.pi*x)+100.*sp.sin(np.pi*x*y)
bc_types = np.array(['DIRICHLET','DIRICHLET','DIRICHLET','DIRICHLET'])
#bc_types = np.array([bc1,bc2,bc3,bc4])
type_CLs = ["fixe","fixe","fixe","fixe"]
# type_CLs = ["variable","variable","variable","variable"]
bc_values = [0.,0.,0.,0.]
#Appel de la fonction de traitement fonctionnel
trait = traitement_fonctionnel.Traitement(T_analytique,u,v,domaine)
#Vitesse moyenne sur le domaine (pour l'analyse selon la valeur du nombre de Péclet Pe)
V = trait.vitesse_moyenne()
#Nombre détapes de raffinement
nb_raf = 4

#Conductivité thermique 
k = V*Lc*Cp*rho/Peclet[0]
# k= Conductivity_K[i]
#Plotting Analytical solution 
trait.graphe_analytique()
#Terme source
source = traitement_fonctionnel.Source(T_analytique,rho,Cp,k,u,v)
q = source.terme_source()
 
N_X = np.array([5,10,20,40]) 
# N_X = np.array([2,4,8,16]) 
L_C = np.array([0.3,0.15,0.075,0.0375]) 
# L_C = np.array([1,0.5,0.25,0.125]) 
nb_raf = len(N_X)
Incer_value_moy = np.zeros(((len(N_X),2,2)))
Incer_value_moy1 = np.zeros(((50,2,2))) 

#Taille des vecteurs d'incertitude : 2 Maillages quads et tri || 2 pour upwind ou centre 
Incer_num = np.zeros((len(Peclet),2,2))
Incer_inputs = np.zeros((len(Peclet),2,2))
U_val = np.zeros((len(Peclet),2,2))
Solution_S =  np.zeros((len(Peclet),2,2))
Error =  np.zeros((len(Peclet),2,2))
Refinement = np.zeros((len(Peclet),2,2))
sig_k=np.array([0.1,1,10]) 
type_maillage1 = np.array(['QUAD','TRI'])
#BOUCLE DE CALCUL NUMERIQUE POUR LES 2 TYPES DE MAILLAGE :
#Transfini de quadrilatère et non-structuré de triangles  

for Index_peclet in range(len(Peclet)):
    k = V*Lc*Cp*rho/Peclet[Index_peclet]
    for Type_maillage_index in range(1): 
        if Type_maillage_index == 0 : 
            Type_Maillage = 'QUAD'
            print("MAILLAGE TRANSFINI DE QUADRILATERES:\n")
        else:
            Type_Maillage = 'TRI'
            print("MAILLAGE NON STRUCTURE DE TRIANGLES:\n")
        # Type_Maillage = 'TRI'    
        # Type_maillage_index = 1
        #Initialisation du vecteur Mesh_size        
        Mesh_size = np.zeros((nb_raf,2)) #Taille moyenne d'un élément h et nb d'éléments par étape de raffinement
        #Boucle de raffinement          
        for index_raf in range(nb_raf):
            #Configuration des maillages
            Nx = N_X[index_raf] #Nombres de mailles en x
            Ny = N_X[index_raf] #Nombres de mailles en y 
            lc = L_C[index_raf] #Taille de maille tri 
        
            #Creation du maillage 
            mesh_obj = mesh_creation(Type_Maillage,nb_raf,index_raf,Nx,Ny,lc,domaine)

            #Vérification du maillage avec les tests d'Euler et de la divergence 
            meshing_check(mesh_obj)

            #Calcul de la solution numérique
            Resolution = solveur.resolution(mesh_obj,k,rho,Cp,u,v,bc_types,T_analytique,q,type_CLs,bc_values)
            Sol_numerique,areas = Resolution.sol_num()
        
            #for centered scheme
            Incer_value_moy[index_raf,Type_maillage_index,0] = np.sum(Sol_numerique[:,0])/len(Sol_numerique[:,0])
            #For upwind Scheme
            Incer_value_moy[index_raf,Type_maillage_index,1] = np.sum(Sol_numerique[:,1])/len(Sol_numerique[:,1])
        
#1 - Estimation de l'incertitude numerique : 
        #Calcul de l'incertitude numerique     
        Propagation = propagation_incertitude.propagation_incertitudess(mesh_obj,domaine)
        # Propagation.countours_numerique(Sol_numerique)
        Incer_num[Index_peclet,Type_maillage_index,:],nb_mailles_choisi = Propagation.numerical_incertitude(N_X,Incer_value_moy,Type_maillage_index,type_maillage1,Index_peclet,Peclet,L_C)
#2 - Estimation de l'incertitude des entrees :    
    #Variable aleatoire K : Conductivity 
   
    #Ecart type
        #sig_k
    #Nombre N
        N = 50
    #Creation de la distrubition de Monte-Carlo et affichage de la distribution  
        k_MonteCarlo= Propagation.variable_alea(Peclet[Index_peclet],sig_k[Index_peclet],N,Index_peclet,Peclet,V,Lc,Cp,rho)
    #Propagation.cdf_impl(k_MonteCarlo,N) # Only to test CDF PLOT 
     
        for index_k in range(N):
        #Configuration des maillages
            Nx = nb_mailles_choisi #Nombres de mailles en x
            Ny = nb_mailles_choisi #Nombres de mailles en y 
            lc = nb_mailles_choisi #Taille de maille tri 
        
            #Creation du maillage 
            mesh_obj = mesh_creation(Type_Maillage,0,0,Nx,Ny,lc,domaine)

            #Vérification du maillage avec les tests d'Euler et de la divergence 
            meshing_check(mesh_obj)

            #Calcul de la solution numérique
            Resolution = solveur.resolution(mesh_obj,k_MonteCarlo[index_k],rho,Cp,u,v,bc_types,T_analytique,q,type_CLs,bc_values)
            Sol_numerique,areas = Resolution.sol_num()
            
            
            #Testing only for centered scheme
            Incer_value_moy1[index_k,Type_maillage_index,0] = np.sum(Sol_numerique[:,0])/len(Sol_numerique[:,0])
        
            #Testing only for upwind scheme
            Incer_value_moy1[index_k,Type_maillage_index,1] = np.sum(Sol_numerique[:,1])/len(Sol_numerique[:,1])
    
    
        Incer_inputs[Index_peclet,Type_maillage_index,:] = Propagation.input_incertitude(N_X,Incer_value_moy1,N,Type_maillage_index,type_maillage1,Index_peclet,Peclet)
#3 - Calcul de S : 
    
        #Creation du maillage 
        mesh_obj = mesh_creation(Type_Maillage,0,0,Nx,Ny,lc,domaine)

        #Vérification du maillage avec les tests d'Euler et de la divergence 
        meshing_check(mesh_obj)

        #Calcul de la solution numérique
        Resolution = solveur.resolution(mesh_obj,k,rho,Cp,u,v,bc_types,T_analytique,q,type_CLs,bc_values)
        Sol_numerique,areas = Resolution.sol_num()
    
        #Testing only for centered scheme
        Solution_S[Index_peclet,Type_maillage_index,0] = np.sum(Sol_numerique[:,0])/len(Sol_numerique[:,0])
        Solution_S[Index_peclet,Type_maillage_index,1] = np.sum(Sol_numerique[:,1])/len(Sol_numerique[:,1])


#4 - Creation des donnees exprimentales   
        Incer_D, Experim_D = Propagation.estimation_u_d(V)
      
    
#5 - Determination de Delta Model 
        Error[Index_peclet,Type_maillage_index,:] = Solution_S[Index_peclet,Type_maillage_index,:] - Experim_D
        U_val[Index_peclet,Type_maillage_index,:] = np.sqrt(Incer_D**2+Incer_inputs[Index_peclet,Type_maillage_index,:]**2+Incer_num[Index_peclet,Type_maillage_index,:]**2)
        Propagation.error_bars_chart2(U_val,Error,Incer_inputs,Incer_num,Incer_D,Solution_S,Type_maillage_index,type_maillage1,Index_peclet,Peclet)


#6- Estimation de l'erreur de discretisation 
    #On ajoute raffine encore une fois le maillage pour pouvoir comparer richardson et le resultat le plus fin 
        if Type_maillage_index == 0 : 
            fine_mesh = N_X[len(N_X)-1] * 2 
        else: 
            fine_mesh = L_C[len(L_C)-1] / 2  
        
    #Configuration du maillage
        Nx = fine_mesh #Nombres de mailles en x
        Ny = fine_mesh #Nombres de mailles en y 
        lc = fine_mesh #Taille de maille tri 
        
        #Creation du maillage 
        mesh_obj = mesh_creation(Type_Maillage,0,0,Nx,Ny,lc,domaine)

        #Vérification du maillage avec les tests d'Euler et de la divergence 
        meshing_check(mesh_obj)

        #Calcul de la solution numérique
        Resolution = solveur.resolution(mesh_obj,k,rho,Cp,u,v,bc_types,T_analytique,q,type_CLs,bc_values)
        Sol_numerique,areas = Resolution.sol_num()
    
        # for centered  and upwind schemes
        Refinement[Index_peclet,Type_maillage_index,0]= np.sum(Sol_numerique[:,0])/len(Sol_numerique[:,0])
        Refinement[Index_peclet,Type_maillage_index,1] = np.sum(Sol_numerique[:,1])/len(Sol_numerique[:,1])
        
        Propagation.estimation_erreur_discr(Incer_value_moy,Refinement,N_X,Type_maillage_index,L_C,Index_peclet,Peclet,type_maillage1)
        

    
    
    
    
    