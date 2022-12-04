# -*- coding: utf-8 -*-
"""
MEC6616 - Aerodynamique numerique

@authors: Zacharie Prigent, Matricule : 2096785 || Mohamed Dhia Slama, Matricule : 2115178
"""

import numpy as np
import sympy as sp
from mesh import Mesh
import pyvista as pv
import pyvistaqt as pvQt
from meshPlotter import MeshPlotter
from solveur import resolution
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText

class post_processing :
    
    def __init__(self, mesh_obj: Mesh, nb_raf,Solution_numerique,T_MMS,index_raf,L_error_norm,P_order,Volume_domain,Mesh_size,areas,LC,domaine,error_type,Type_maillage_index,Index_Peclet,Peclet) :
        self.mesh_obj = mesh_obj # Maillage utilise
        self.nb_raf=nb_raf
        self.Solution_numerique = Solution_numerique
        self.T_MMS = T_MMS
        self.index_raf = index_raf 
        self.L_error_norm = L_error_norm 
        self.P_order = P_order
        self.Volume_domain = Volume_domain
        self.Mesh_size = Mesh_size
        self.areas = areas
        self.LC = LC
        self.domaine=domaine
        self.error_type = error_type
        self.Type_maillage_index = Type_maillage_index
        self.Index_Peclet = Index_Peclet
        self.Peclet = Peclet
        self.nb_interpolation=10

    def countours_numerique(self):
        #Contour de la solution numérique avec pyvista pour le schéma centré
        plotter1 = MeshPlotter()
        nodes1, elements1 = plotter1.prepare_data_for_pyvista(self.mesh_obj)
        pv_mesh1 = pv.PolyData(nodes1, elements1)
        pv_mesh1['Solution numérique pour le schéma centré'+ ' (Peclet =' + str(self.Peclet[self.Index_Peclet]) + ')'] = self.Solution_numerique[:,0]
        pl = pvQt.BackgroundPlotter()
        #Message dans la console 
        print("Voir la solution numerique du schéma centré dans la fenêtre de PyVista")
        # Tracé du champ
        pl.add_mesh(pv_mesh1, show_edges=True, scalars="Solution numérique pour le schéma centré"+ " (Peclet =" + str(self.Peclet[self.Index_Peclet]) + ")", cmap="jet")
        # Tracé des iso-lignes
        pv_mesh1 = pv_mesh1.compute_cell_sizes(length=False, volume=False)
        pv_mesh1 = pv_mesh1.cell_data_to_point_data()
        pv_mesh1['Champ T (noeuds)'] = pv_mesh1.active_scalars
        contours = pv_mesh1.contour(isosurfaces=15, scalars="Champ T (noeuds)")
        pl.add_mesh(contours, color='k', show_scalar_bar=False, line_width=2)
        pl.camera_position = 'xy'
        #pl.add_text('Champ scalaire T', position="upper_edge")
        #Affichage 
        pl.show()   
        
        #Contour de la solution numérique avec pyvista pour le schéma upwind         
        plotter1 = MeshPlotter()
        nodes1, elements1 = plotter1.prepare_data_for_pyvista(self.mesh_obj)
        pv_mesh1 = pv.PolyData(nodes1, elements1)
        pv_mesh1['Solution numérique pour le schéma upwind'+ ' (Peclet =' + str(self.Peclet[self.Index_Peclet]) + ')'] = self.Solution_numerique[:,1]
        pl = pvQt.BackgroundPlotter()
        #Affichage
        #Message dans la console 
        print("Voir la solution numerique du schéma centré dans la fenêtre de PyVista")
        # Tracé du champ
        pl.add_mesh(pv_mesh1, show_edges=True, scalars="Solution numérique pour le schéma upwind"+ " (Peclet =" + str(self.Peclet[self.Index_Peclet]) + ")", cmap="jet")
        # Tracé des iso-lignes
        pv_mesh1 = pv_mesh1.compute_cell_sizes(length=False, volume=False)
        pv_mesh1 = pv_mesh1.cell_data_to_point_data()
        pv_mesh1['Champ T (noeuds)'] = pv_mesh1.active_scalars
        contours = pv_mesh1.contour(isosurfaces=15, scalars="Champ T (noeuds)")
        pl.add_mesh(contours, color='k', show_scalar_bar=False, line_width=2)
        pl.camera_position = 'xy'
        #pl.add_text('Champ scalaire T', position="upper_edge")
        #Affichage 
        pl.show()   
       
        
        return 
    
    def contour_numerique_2(self) : 
        # Affichage de champ scalaire avec pyvista
        plotter = MeshPlotter()
        x, y = sp.symbols('x y')
        T = 10 + 4*sp.sin(np.pi*x/2) - 3*sp.cos(np.pi*y/2) + 2.5*sp.sin(np.pi*x*y/4)
        fT = sp.lambdify([x, y], T, 'numpy')
        cell_centers = np.zeros((self.mesh_obj.get_number_of_elements(), 2))
        for i_element in range(self.mesh_obj.get_number_of_elements()):
            center_coords = np.array([0.0, 0.0])
            nodes = self.mesh_obj.get_element_to_nodes(i_element)
            for node in nodes:
                center_coords[0] += self.mesh_obj.get_node_to_xcoord(node)
                center_coords[1] += self.mesh_obj.get_node_to_ycoord(node)
            center_coords /= nodes.shape[0]
            cell_centers[i_element, :] = center_coords
        nodes, elements = plotter.prepare_data_for_pyvista(self.mesh_obj)
        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh['Champ T'] = fT(cell_centers[:, 0], cell_centers[:, 1])

        pl = pvQt.BackgroundPlotter()
        # Tracé du champ
        pl.add_mesh(pv_mesh, show_edges=True, scalars="Champ T", cmap="RdBu")
        # Tracé des iso-lignes
        nodes_xcoords = self.mesh_obj.get_nodes_to_xcoord()
        nodes_ycoords = self.mesh_obj.get_nodes_to_ycoord()
        pv_mesh['Champ T (noeuds)'] = fT(nodes_xcoords, nodes_ycoords)
        contours = pv_mesh.contour(isosurfaces=15, scalars="Champ T (noeuds)")
        pl.add_mesh(contours, color='k', show_scalar_bar=False, line_width=2)
        pl.camera_position = 'xy'
        pl.add_text('Champ scalaire T', position="upper_edge")
        pl.show()
        return 
    
    def calcul_coordonnees_centre_element (self, iElement) :
        #Calcul des coordonnees du centre d'un élément
        nodes= self.mesh_obj.get_element_to_nodes(iElement)
        nb_nodes=len(nodes)
        x=(1/nb_nodes) *np.sum([self.mesh_obj.get_node_to_xcoord(nodes[i]) for i in range (0,nb_nodes)])
        y=(1/nb_nodes) *np.sum([self.mesh_obj.get_node_to_ycoord(nodes[i]) for i in range (0,nb_nodes)])
        return x,y  

    def coupes_y(self):
        for schema in range (2): #Indice 0 : schéma centré, indice 1 : schéma upwind 
            #Coordonnées des centres de chaque élément du maillage
            number_of_elements = self.mesh_obj.get_number_of_elements()
            elements_xy = np.zeros((number_of_elements,2))
            for i in range(number_of_elements):
                elements_xy[i,0],elements_xy[i,1] = resolution.calcul_coordonnees_centre_element(self,i)
        
            Lx = self.domaine[1]-self.domaine[0] #Longueur sur l'axe des x 
            Ly = self.domaine[3]-self.domaine[2] #Longueur sur l'axe des y
        
            #Nombre de points à interpoler
            Nb_interpoler = self.nb_interpolation
        
            #Initialisation de la solution interpolée
            Phi_num_interpoler = np.zeros(Nb_interpoler)
        
            #Création des points à tracer
            Points = np.zeros((Nb_interpoler,2)) # 
            dx=Lx/Nb_interpoler
            y_coupe=(self.domaine[3]+self.domaine[2])/2 #Coupe au centre du domaine en y
            for i in range(Nb_interpoler):
                Points[i,0] = self.domaine[0] + dx/2+dx*i #Coordonnée sur x  # Ligne : modifiee : Consideration du domaine negative 
                Points[i,1] = y_coupe  #Coordonnée sur y 
        
            #Recherche de voisins d'intérêt pour les points d'interpolation : moyenne pondérée des voisions proches
            voisins=[]
            for i in range (Nb_interpoler):
                v=[]
                for j in range (number_of_elements):
                    rayon=np.sqrt((elements_xy[j,0]-Points[i,0])**2+(elements_xy[j,1]-Points[i,1])**2)
                    rayon_cible=3*np.sqrt(self.areas[j]/np.pi)
                    #On recherche les voisins proches dans une zone circulaire autour de l'élément
                    #Le rayon initial du cercle est celui d'un cercle dont l'aire est celle de l'élément
                    #On utilise un facteur de sécurité de 3 pour être sûr de trouver suffisament de voisins et pour lisser les discontinuités
                    if rayon<=rayon_cible:
                        v+=[[rayon,self.Solution_numerique[j,0]]] #Ligne Modifiee : Ajout de la deuxieme colonne 
                v=np.array(v)
                voisins+=[v]
            voisins=np.array(voisins,dtype=np.ndarray)
                
            #Interpolations des valeurs de la solution numérique aux points de la coupe
            for i in range (Nb_interpoler):
                v=voisins[i]
                nb_voisins=np.shape(v)[0]
                ponderation=np.array([1-v[j,0]/rayon_cible for j in range(0,nb_voisins)])
                Phi_num_interpoler[i]=np.sum(ponderation*v[:,1].T)/np.sum(ponderation)            
       
            #Affichage
            x = np.array([self.domaine[0] + dx/2+dx*i for i in range (Nb_interpoler)])  # Ligne modifiee
            plt.plot(x,Phi_num_interpoler ,"s", label="Solution numérique")
    
            X=np.linspace(self.domaine[0],self.domaine[1],100) # a automatiser # Ligne modifiee
            Y = np.linspace(y_coupe,y_coupe,100) 
            x,y = sp.symbols('x y')
            sol_analytique = sp.lambdify([x,y], self.T_MMS, "numpy")
            plt.plot(X,sol_analytique(X,Y), label="Solution MMS")
            plt.xlim(self.domaine[0],self.domaine[1])
            y_titre=str(round(y_coupe,2))
            type_element=["quadrangles","triangles"]
            type_solveur = ["schéma centré","schéma upwind"]
            plt.title(label="Coupe en y=" + y_titre +  " pour " + str(number_of_elements) + " éléments (" + type_element[self.Type_maillage_index] +")" + " avec le " +type_solveur[schema]  + " (Peclet =" + str(self.Peclet[self.Index_Peclet]) + ")" ,fontsize=11,color="black")
            plt.xlabel("Distance x [m]")
            plt.ylabel("Temperature [K]")
            plt.legend()
            plt.show()
        return
    
    
    def compute_sol_analy(self):
        #Nombre d'éléments dans notre maillage
        Number_of_elements =self.mesh_obj.get_number_of_elements()
    
        #Solution analytique pour chaque élément
        Phi_analy=np.zeros(Number_of_elements) 
        
        #Initialisation  de tableau des coordonnees des centres 
        Centres = np.zeros((Number_of_elements,2))
        
        for iElement in range(Number_of_elements):
            Centres[iElement,0],Centres[iElement,1] = resolution.calcul_coordonnees_centre_element(self,iElement)
            x,y = sp.symbols('x y')
            sol_analytique = sp.lambdify([x,y], self.T_MMS, "numpy")
            Phi_analy[iElement] = sol_analytique(Centres[iElement,0],Centres[iElement,1])
            
        return Phi_analy
    
    
    def normes(self, _U1, _U2, _Methode):
       #Calcul de la norme sur les vecteurs redimensionnés
       if _Methode == "L1":
           L = np.sum(np.absolute(_U2 - _U1)*self.areas)/self.Volume_domain
       elif _Methode == "L2":
           # V=np.sum(self.areas)
           L = np.sqrt(np.sum(np.square(_U2 - _U1)*self.areas)/self.Volume_domain)
       elif _Methode == "Li":
           L = np.amax(np.absolute(_U2 - _U1))
       else:
           self.ErrorMessage("Specifiez la norme L1,L2 ou L_i")
       return L
   
    
    def compute_ordre(self,schema):
        #Calcul de h
        h=self.Mesh_size[:,0]
        
        #Calcul de p
        p=np.polyfit(np.log([h[i] for i in range(self.nb_raf)]),np.log([self.L_error_norm[i,schema] for i in range(self.nb_raf)]), 1)
        self.P_order[0,schema]=p[0]
        
        # #Calcul de r
        # r=self.Mesh_size[self.index_raf-1,0]/self.Mesh_size[self.index_raf,0]
        
        # #Calcul de p
        # self.P_order[0,schema]= np.log(self.L_error_norm[self.index_raf-1,schema]/self.L_error_norm[self.index_raf,schema])/np.log(r)
            
        return
    
    
    def error_chart(self,schema): 
        n_x = np.array(self.Mesh_size[:,0])
    
        a,b = np.polyfit(np.log(n_x[1:4]), np.log(self.L_error_norm[1:4,schema]), 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        text = AnchoredText('Ordre de convergence p = ' + str(round(self.P_order[0,schema],2)), loc='lower right')
        ax.plot(n_x[1:4],self.L_error_norm[1:4,schema],'ko',label='$\Vert e_{\mathbf{u}}\Vert_{2}$')
        ax.plot(n_x[1:4], self.L_error_norm[1:4,schema], '-k', label='$\Vert e_{\mathbf{u}}\Vert_{2}= \ %3.2f \; h^{%3.2f}$' %(np.exp(b),a))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(b=True, which='minor', color='grey', linestyle='--')
        ax.grid(b=True, which='major', color='k', linestyle='-')
        plt.ylabel('$\Vert e \Vert_2$')
        plt.xlabel('Taille de maille h')
        type_solveur = ["schéma centré","schéma upwind"]
        if self.Type_maillage_index == 0 : 
            String = "Erreur pour un maillage transfini de quadrilatères pour le " + type_solveur[schema] + " (Peclet =" + str(self.Peclet[self.Index_Peclet]) + ")"
        else: 
            String = "Erreur pour un maillage non-structuré de triangles pour le " + type_solveur[schema] + " (Peclet =" + str(self.Peclet[self.Index_Peclet]) + ")"
            
        plt.title(label=String ,fontsize=11,color="black")
        plt.legend()
        ax.add_artist(text)
        plt.show()
        return 

    
    def main_post_processing (self):        
        self.Mesh_size[self.index_raf,1] = self.mesh_obj.get_number_of_elements()
        self.Mesh_size[self.index_raf,0] = np.sqrt((1/self.mesh_obj.get_number_of_elements())*(np.sum(self.areas)))
        
        #Calcul de la fonction analytique 
        Solution_analytique = self.compute_sol_analy()
        # #Affichage
        # if (self.index_raf==self.nb_raf-1):
        #     plotter1 = MeshPlotter()
        #     nodes1, elements1 = plotter1.prepare_data_for_pyvista(self.mesh_obj)
        #     pv_mesh1 = pv.PolyData(nodes1, elements1)
        #     pv_mesh1['Solution analytique'] = Solution_analytique
        #     pl = pvQt.BackgroundPlotter()
        #     #Tracé du champ
        #     print("Voir la solution analytique dans la fenêtre de PyVista\n")
        #     pl.add_mesh(pv_mesh1, show_edges=True, scalars="Solution analytique", cmap="jet")
        #     pl.show()
        
        #Création des contours pour les solutions analytique et numérique pour le maillage le plus fin
        # if (self.index_raf==self.nb_raf-1):
        #     print("Traitement des données :")
        #self.contour_numerique_2()
        self.countours_numerique()
        
        #Creation des coupes en y 
        self.coupes_y()
        
        for schema in range(2):
            #Calcul de L'erreur (l'utilisateur peut choisir dans le main entre les trois types de normes pour calculer l'erreur)
            self.L_error_norm[self.index_raf,schema] = self.normes(Solution_analytique,self.Solution_numerique[:,schema],self.error_type)
                  
            #Calcul de l'ordre 
            if (self.index_raf ==self.nb_raf-1) :
                self.compute_ordre(schema)
            # #Creation de la courbe d'erreur log.log 
            if (self.index_raf == self.nb_raf-1):
                self.error_chart(schema)
        
        return 