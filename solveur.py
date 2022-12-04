# -*- coding: utf-8 -*-
"""
MEC6616 - TPP2 : Equation de convection-diffusion 2D - Maillages de triangles vs quadrilatères

@authors: Zacharie Prigent, Matricule : 2096785 || Mohamed Dhia Slama, Matricule : 2115178
"""

import numpy as np
import sympy as sp
from mesh import Mesh
import scipy.sparse as sps 
from scipy.sparse.linalg.dsolve import linsolve 

#Classe de verification de maillage avec les methodes d'Euler et Gradient constant
class mesh_check: 
    def __init__(self, mesh_obj: Mesh) :
        self.mesh_obj = mesh_obj # Maillage utilise 
    
        
    #Vérfication du test d'Euler     
    def euler_check(self):
       
        number_of_elements = self.mesh_obj.get_number_of_elements()# Nombre d'éléments
        number_of_nodes = self.mesh_obj.get_number_of_nodes() #Nombre de sommets
        number_of_faces = self.mesh_obj.get_number_of_faces() # Nombre d'aretes
        number_of_holes = number_of_faces - number_of_elements - number_of_nodes + 1 # Nombre de trous
        Result = number_of_elements -number_of_faces+number_of_nodes + number_of_holes
        if (Result == 1 ):
            Resultat_string = ("Maillage verifiee avec la condition d'Euler : " + str(Result))
        else: 
            Resultat_string =("Maillage non-verifiee avec la condition d'Euler : " + str(Result))
        return Resultat_string
    
    
    #Verfication du test de la divergence
    def divergence_verification(self):
        
        face_to_node = self.mesh_obj.get_faces_to_nodes() #Tableau des arretes 
        Div = np.zeros((self.mesh_obj.get_number_of_elements(),2))   # Initialisation du tableau
        F = 1 
        
        for iface in range(self.mesh_obj.get_number_of_faces()):
            #Extraction de coordonnees des noeuds
            x1 = self.mesh_obj.get_node_to_xcoord(face_to_node[iface,0])
            y1 = self.mesh_obj.get_node_to_ycoord(face_to_node[iface,0])
            x2 = self.mesh_obj.get_node_to_xcoord(face_to_node[iface,1])
            y2 = self.mesh_obj.get_node_to_ycoord(face_to_node[iface,1])
            ds = np.sqrt((x2-x1)*(x2-x1) +(y2-y1)*(y2-y1))
            #normalisation
            n = self.normalisation(x1,x2,y1,y2)
            #Calcul du flux
            Flux = ds*F*n
            
            element_left = self.mesh_obj.get_face_to_elements(iface)[0]  
            element_right = self.mesh_obj.get_face_to_elements(iface)[1] 
            #Condition si sur l'existance de l'element de gauche
            if (element_left != -1): 
                Div[element_left]=np.round(Div[element_left] + Flux,7)
                
            #Condition si sur l'existance de l'element de droite    
            if (element_right != -1): 
                 Div[element_right]=np.round(Div[element_right] - Flux,7)
                 
        Somme1 = np.sum(Div)
        if (Somme1 == 0 ):
            Somme_string = ("Maillage verifiee avec test de divergence : " + str(Somme1))
        else: 
            Somme_string = ("Maillage non-verifiee avec test de divergence : " + str(Somme1))
        return Somme_string
    
  
    def normalisation(self, x1,x2,y1,y2):
       n = np.array([y2-y1,x1-x2])
       normalisation = np.sqrt((x2-x1)**2 + (y2-y1)**2)
       n = n/normalisation
       return n
   

class resolution:     
    def __init__(self, mesh_obj: Mesh,k,rho,Cp,u,v,bc_types,T_MMS,q,type_CLs,bc_values):
        self.mesh_obj = mesh_obj 
        self.K = k
        self.rho = rho
        self.Cp = Cp
        self.u = u
        self.v = v
        self.bc_types = bc_types
        self.func_analytique = T_MMS
        self.q = q
        self.type_CLs = type_CLs
        self.bc_values = bc_values
   
    
    def sol_num(self) :   
        number_of_elements = self.mesh_obj.get_number_of_elements()
        Phi = np.zeros([number_of_elements,2]) #colonne 1 : schéma centré, colonne 2 : schéma upwind
        number_iterations = 5 
        areas = np.zeros(number_of_elements)
        
        #Calcul de l'aire de tous les éléments
        areas = self.compute_areas(number_of_elements, areas)
        
        #Création des fonctions symboliques
        x,y = sp.symbols('x y')
        u=sp.lambdify([x,y], self.u, "numpy")
        v=sp.lambdify([x,y], self.v, "numpy")
        
        for schema in range (2): #Indice 0 : schéma centré, indice 1 : schéma upwind 
            #Initialisation du gradient au centre de chaque élément
            GRAD=np.zeros([number_of_elements,2])
            
            for iterations in range(0,number_iterations):
                #Initialisation de la matrice et du membre de droite (méthodologie sparse)
                A = sps.lil_matrix((number_of_elements,number_of_elements),dtype=np.float64)
                B = np.zeros(number_of_elements)
                
                #Faces internes
                for i_face in range(self.mesh_obj.get_number_of_boundary_faces(), self.mesh_obj.get_number_of_faces()):
                    #Extraction des coordonées a et b (noeuds formant l'arête)
                    face_nodes = self.mesh_obj.get_face_to_nodes(i_face) 
                    Coord_a,Coord_b = self.calcul_coordonnees_arete(face_nodes)
                
                    #Extraction coordonnées des points au centre des éléments
                    element_left = self.mesh_obj.get_face_to_elements(i_face)[0]  
                    element_right = self.mesh_obj.get_face_to_elements(i_face)[1]
                    xP,yP = self.calcul_coordonnees_centre_element(element_left)
                    xV,yV = self.calcul_coordonnees_centre_element(element_right)
                    
                    #Calcul des coordonnées du point au centre de l'arête 
                    xA = (Coord_a[0]+Coord_b[0])/2
                    yA = (Coord_a[1]+Coord_b[1])/2
                    
                    #Vecteur normal à l'arête
                    n = self.normalisation(Coord_a[0],Coord_b[0],Coord_a[1],Coord_b[1])
                    
                    #Calcul des deltas A, eksi et eta
                    Delt_A = np.sqrt((Coord_b[0]-Coord_a[0])**2 +(Coord_b[1]-Coord_a[1])**2 )
                    Delt_ETA = Delt_A 
                    Delt_KSI = np.sqrt((xV-xP)**2 +(yV-yP)**2)
                    
                    PNKSI,PKSIETA,E_ETA= self.calcul_coefficients(xV,xP,yV,yP,Coord_a,Coord_b,Delt_ETA,Delt_KSI,n)
                    
                    #Terme de diffusion
                    D = (1/PNKSI)*self.K*Delt_A/Delt_KSI
                    
                    #Terme de cross-diffusion avec la methode de moindre carrés
                    Grad_moy = (GRAD[element_left] + GRAD[element_right])/2
                    Sd = -self.K *(PKSIETA/PNKSI) * Delt_A* np.dot(Grad_moy,E_ETA)
                    
                    #Terme de convection
                    champ_vitesse=np.array([u(xA,yA),v(xA,yA)])
                    
                    
                    F = self.rho * np.dot(n,champ_vitesse) * Delt_A

                    #Ajout des valeurs dans A
                    if schema==0: #centré
                        A[element_left,element_left] = A[element_left,element_left] + D + F/2
                        A[element_right,element_right] = A[element_right,element_right] + D - F/2
                        A[element_left,element_right] = A[element_left,element_right] - D + F/2 
                        A[element_right,element_left] = A[element_right,element_left] - D - F/2
                    elif schema==1: #upwind
                        A[element_left,element_left] = A[element_left,element_left] + D + max(F,0)
                        A[element_right,element_right] = A[element_right,element_right] + D + max(0,-F)
                        A[element_left,element_right] = A[element_left,element_right] - D - max(0,-F) 
                        A[element_right,element_left] = A[element_right,element_left] - D - max(F,0)
                    
                    #Ajout des valeurs dans B
                    B[element_left] += Sd
                    B[element_right] -= Sd                   
                
                #Faces frontières
                for i_face in range(self.mesh_obj.get_number_of_boundary_faces()):
                    tag = self.mesh_obj.get_boundary_face_to_tag(i_face)
                    bc_type = self.bc_types[tag]
                                
                    #Extraction des noeuds a et b formant l'arête et de leurs coordonnées
                    face_nodes = self.mesh_obj.get_face_to_nodes(i_face) 
                    Coord_a,Coord_b = self.calcul_coordonnees_arete(face_nodes)
                    
                    #Vecteur normal à l'arête
                    n = self.normalisation(Coord_a[0],Coord_b[0],Coord_a[1],Coord_b[1])
                    
                    #Calcul des coordonnées du point au centre de l'arête 
                    xA = (Coord_a[0]+Coord_b[0])/2
                    yA = (Coord_a[1]+Coord_b[1])/2
                    
                    #Calcul de delta A
                    Delt_A = np.sqrt((Coord_b[0]-Coord_a[0])**2 +(Coord_b[1]-Coord_a[1])**2 )
                    
                    #Calcul des coordonnees du point au centre de l'élément de gauche
                    element_left = self.mesh_obj.get_face_to_elements(i_face)[0]  
                    xP,yP = self.calcul_coordonnees_centre_element(element_left)
                    
                    #Distances entre les coordonées des centres de l'élément et de l'arête
                    dx = xA - xP
                    dy = yA - yP
                    
                    #Terme de convection
                    champ_vitesse=np.array([u(xA,yA),v(xA,yA)])
                    
                    F = self.rho * np.dot(n,champ_vitesse) * Delt_A
                    
                    #Valeur de la CF spécifique à l'arête
                    bc_value = self.variable_bc(bc_type,xA,yA,xP,yP,dx,dy,n,tag)
                    
                    if (bc_type == 'DIRICHLET') :
                        D,Sd = self.Dirichlet(Coord_a,Coord_b,n,xA,yA,xP,yP,tag)
                        if schema==0: #centré
                            A[element_left,element_left] = A[element_left,element_left] + D         
                            B[element_left] = B[element_left] + Sd + D*bc_value - F*bc_value
                        elif schema==1: #upwind
                            A[element_left,element_left] = A[element_left,element_left] + D + max(F,0)               
                            B[element_left] = B[element_left] + Sd + D*bc_value + max(0,-F)*bc_value
                    
                    elif (bc_type == 'NEUMANN'): #ATTENTION AUX SIGNES DANS B
                        Delt_KSI=np.sqrt((xA-xP)**2 +(yA-yP)**2)
                        E_KSI = np.array([(xA-xP)/(Delt_KSI),(yA-yP)/(Delt_KSI)])                        
                        PNKSI = np.dot(n,E_KSI) 
                        A[element_left,element_left] = A[element_left,element_left] + F
                        B[element_left] = B[element_left] + self.K * Delt_A * bc_value + F * bc_value * PNKSI * Delt_KSI
                        
                #Ajout du terme source            
                for ielm in range (number_of_elements): 
                    xP,yP = self.calcul_coordonnees_centre_element(ielm)
                    B[ielm] += self.q(xP,yP) * areas[ielm] #ATTENTION AU SIGNE
                    
                #Calcul de la solution numérique
                A = A.tocsr()
                Phi[:,schema] = linsolve.spsolve(A,B)
                            
                #Reconstruction du gradient pour l'itération suivante
                GRAD=self.grad(Phi[:,schema])
        
        return Phi, areas               
    
    
    def normalisation (self, x1,x2,y1,y2):
       n = np.array([y2-y1,x1-x2])
       normalisation = np.sqrt((x2-x1)**2 + (y2-y1)**2)
       n = n/normalisation
       return n
    
    
    def calcul_coordonnees_centre_element (self, iElement) :
        #Calcul des coordonnees du centre d'un élément
        nodes= self.mesh_obj.get_element_to_nodes(iElement)
        nb_nodes=len(nodes)
        if len(nodes) < 4 : 
            x=(1/nb_nodes) *np.sum([self.mesh_obj.get_node_to_xcoord(nodes[i]) for i in range (0,nb_nodes)])
            y=(1/nb_nodes) *np.sum([self.mesh_obj.get_node_to_ycoord(nodes[i]) for i in range (0,nb_nodes)])
        else: 
            x=(1/nb_nodes) *np.sum([self.mesh_obj.get_node_to_xcoord(nodes[i]) for i in range (0,nb_nodes)])
            y=(1/nb_nodes) *np.sum([self.mesh_obj.get_node_to_ycoord(nodes[i]) for i in range (0,nb_nodes)])
            # X=self.mesh_obj.get_node_to_xcoord(nodes)
            # Y=self.mesh_obj.get_node_to_ycoord(nodes)
            # TriX1=np.asarray([X[0],X[1],X[2]])
            # TriY1=np.asarray([Y[0],Y[1],Y[2]])
            # TriX2=np.asarray([X[2],X[3],X[1]])
            # TriY2=np.asarray([Y[2],Y[3],Y[1]])
            # xc1=np.mean(TriX1)
            # yc1=np.mean(TriY1)
            # xc2=np.mean(TriX2)
            # yc2=np.mean(TriY2)
            # A1=1/2*abs((X[1]-X[0])*(Y[2]-Y[0])-(X[2]-X[0])*(Y[1]-Y[0]))
            # A2=1/2*abs((X[3]-X[2])*(Y[0]-Y[2])-(X[0]-X[2])*(Y[3]-Y[2]))
            # x=(xc1*A1+xc2*A2)/(A1+A2)
            # y=(yc1*A1+yc2*A2)/(A1+A2)
        
        
        return x,y
    
    
    def calcul_coordonnees_arete (self, face_nodes) :
        #a = 1er noeud dans le vecteur face_nodes, b = 2e noeud dans le vecteur face_nodes
        Coord_a = self.mesh_obj.get_node_to_xycoord(face_nodes[0])
        Coord_b = self.mesh_obj.get_node_to_xycoord(face_nodes[1])
        return Coord_a,Coord_b
    
    
    def calcul_coefficients(self,xA,xP,yA,yP,Coord_a,Coord_b,Delt_ETA,Delt_KSI,n) :
        E_KSI = np.array([(xA-xP)/(Delt_KSI),(yA-yP)/(Delt_KSI)])
        E_ETA = np.array([(Coord_b[0]-Coord_a[0])/Delt_ETA, (Coord_b[1]-Coord_a[1])/Delt_ETA])
        #Calcul des coefficients PNKSI et PKSIETA 
        PNKSI = np.dot(n,E_KSI) 
        PKSIETA = np.dot(E_KSI,E_ETA)
        return PNKSI, PKSIETA, E_ETA
    
    
    def compute_areas (self, number_of_elements,areas):         
        for i in range(number_of_elements): 
            nodes = [self.mesh_obj.get_element_to_nodes(i)]
            area_matrix = [np.zeros([2, 2]) for i in range(len(nodes[0]))]
            for j in range(len(nodes[0])):
                i_nodes = nodes[0][j]
                x, y = self.mesh_obj.get_node_to_xycoord(i_nodes)[0], self.mesh_obj.get_node_to_xycoord(i_nodes)[1]
                #Contruction des matrices aire
                area_matrix[j][:][0] = [x, y]
                area_matrix[j-1][:][1] = [x, y]   
            #Aire pour tous types de maillage
            areas[i] = np.sum([np.linalg.det(area_matrix[k]) for k in range(len(nodes[0]))])/2            
        return  areas   
            
    
    def variable_bc(self,bc_type,xA,yA,xP,yP,dx,dy,n,tag):
        if self.type_CLs[tag] =="variable":
            if (bc_type == 'DIRICHLET') :
                x,y = sp.symbols('x y')
                sol_analytique=sp.lambdify([x,y], self.func_analytique, "numpy")
                bc_value = sol_analytique(xA,yA)
            elif (bc_type == 'NEUMANN'): 
                x,y = sp.symbols('x y')
                M = np.array([sp.diff(self.func_analytique,x),sp.diff(self.func_analytique,y)])
                d_func_analytique = np.dot(M,n)
                d_func_analytique = sp.lambdify([x,y], d_func_analytique, "numpy")
                bc_value = d_func_analytique(xA,yA)  
        elif self.type_CLs[tag] =="fixe" :  
            bc_value = self.bc_values[tag]
        return bc_value
    
    
    def Dirichlet(self,Coord_a,Coord_b,n,xA,yA,xP,yP,tag): 
        #Calcul des deltas A, ksi et eta
        Delt_A =np.sqrt((Coord_b[0]-Coord_a[0])**2 +(Coord_b[1]-Coord_a[1])**2 )
        Delt_ETA = Delt_A
        Delt_KSI = np.sqrt((xA-xP)**2 +(yA-yP)**2)
        
        #Vecteurs
        PNKSI,PKSIETA,E_ETA = self.calcul_coefficients(xA,xP,yA,yP,Coord_a,Coord_b,Delt_ETA,Delt_KSI,n) 
        
        #Terme de diffusion 
        D = (1/PNKSI)*self.K*Delt_A/Delt_KSI
        
        
        #Cross-diffusion term avec la methode de moindre carrés 
        if self.type_CLs[tag]== "variable":
            x,y = sp.symbols('x y')
            sol_analytique=sp.lambdify([x,y], self.func_analytique, "numpy")
            node1 = sol_analytique(Coord_a[0],Coord_a[1])
            node2 = sol_analytique(Coord_b[0],Coord_b[1])
            Sd = -self.K *(PKSIETA/PNKSI) * Delt_A * (node2- node1)/Delt_ETA
        elif self.type_CLs[tag]== "fixe" : 
            Sd = 0 
        return D,Sd
    
    
    def grad(self,Phi):
            number_of_elements = self.mesh_obj.get_number_of_elements()
            
            #Initialisation de la matrice et du membre de droite
            # ATA = sps.lil_matrix((number_of_elements,2,2),dtype=np.float64)
            # ATAI = sps.lil_matrix((number_of_elements,2,2),dtype=np.float64)
            ATA = np.zeros(((number_of_elements,2,2)))
            ATAI = np.zeros(((number_of_elements,2,2)))
            GRAD = np.zeros((number_of_elements,2))
            B = np.zeros((number_of_elements,2)) 
            
            #Internal faces
            for i_face in range(self.mesh_obj.get_number_of_boundary_faces(), self.mesh_obj.get_number_of_faces()):
                ALS = np.zeros((2,2))
                element_left = self.mesh_obj.get_face_to_elements(i_face)[0]  
                element_right = self.mesh_obj.get_face_to_elements(i_face)[1]
                xA,yA = self.calcul_coordonnees_centre_element(element_right) #Xd ,Yd
                xP,yP = self.calcul_coordonnees_centre_element(element_left) #Xg, Yg
                dx = xA- xP ; dy = yA-yP
                #Remplissage de ALS puis ATA
                ALS[0,0] = dx*dx
                ALS[1,0] = dx*dy
                ALS[0,1] = dy*dx
                ALS[1,1] = dy*dy
                ATA[element_left] += ALS
                ATA[element_right] += ALS
                #Remplissage du vecteur B 
                dphi = Phi[element_right] - Phi[element_left]
                B[element_left,0] += dx*dphi
                B[element_left,1] += dy*dphi
                B[element_right,0] += dx*dphi
                B[element_right,1] += dy*dphi
            
            #Boundary faces  
            for i_face in range(0,self.mesh_obj.get_number_of_boundary_faces()):
                tag = self.mesh_obj.get_boundary_face_to_tag(i_face)  #Numéro du tag de la facefrontière
                bc_type = self.bc_types[tag]  #Type de condition frontière (Dirichlet ou Neumann)
                ALS = np.zeros((2,2))
                
                if bc_type != 'LIBRE':
                    #Extraction des coordonees a et b (noeuds formant l'arete)
                    face_nodes = self.mesh_obj.get_face_to_nodes(i_face) 
                    Coord_a,Coord_b = self.calcul_coordonnees_arete(face_nodes)
                    #Calcul des coordonnees du point milieu de l'arete 
                    Xta = (Coord_a[0]+Coord_b[0])/2
                    Yta = (Coord_a[1]+Coord_b[1])/2
                    #Vecteur normal
                    n = self.normalisation(Coord_a[0],Coord_b[0],Coord_a[1],Coord_b[1])
                    #Coordonnées du point au centre de l'élément
                    element_left = self.mesh_obj.get_face_to_elements(i_face)[0]  
                    Xtg,Ytg = self.calcul_coordonnees_centre_element(element_left)
                    dx = Xta - Xtg
                    dy = Yta - Ytg
                    #Calcul de la CF variable
                    bc_value = self.variable_bc(bc_type,Xta,Yta,Xtg,Ytg,dx,dy,n,tag)
                    #Calcul du gradient 
                    dphi = bc_value - Phi[element_left]
                    if bc_type == 'NEUMANN':
                        dphi=np.dot([dx, dy], n)*bc_value
                        dx= np.dot([dx, dy], n) * n[0]
                        dy= np.dot([dx, dy], n) * n[1]
                    ALS[0,0]=dx*dx
                    ALS[1,0]=dx*dy
                    ALS[0,1]=dx*dy
                    ALS[1,1]=dy*dy
                    ATA[element_left] += ALS
                    # Remplisage du membre de droite
                    B[element_left,0] += dx*dphi
                    B[element_left,1] += dy*dphi

            #Compute
            # ATAI = np.array([sps.linalg.inv(ATA[i]) for i in range(number_of_elements)])
            # ATAI = ATAI.tocsr()
            ATAI = np.array([np.linalg.inv(ATA[i]) for i in range(number_of_elements)])
            GRAD = np.array([np.dot(ATAI[i], B[i]) for i in range(number_of_elements)])        
            return GRAD
    