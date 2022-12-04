# -*- coding: utf-8 -*-
"""
MEC6616 - PROJET : Equation de convection-diffusion 2D - Maillages de triangles vs quadrilatères : Propagation des incertitudes

@authors:  Mohamed Dhia Slama, Matricule : 2115178
"""
import numpy as np
import pandas as pd
import sympy as sp
import pyvista as pv
import pyvistaqt as pvQt
from meshPlotter import MeshPlotter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from mesh import Mesh
from meshPlotter import MeshPlotter
import solveur as solveur
import post_processing as post_processing
import traitement_fonctionnel as traitement_fonctionnel
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
import statistics
from scipy.stats import norm
import scipy as scipy
import matplotlib.mlab as mlab

class propagation_incertitudess(): 
    
    def __init__(self,mesh_obj: Mesh,domaine):
        self.mesh_obj = mesh_obj #Objet de maillage 
        self.domaine = domaine # Domaine d'etude
        
    
    def mesh_creation(self,Type_Maillage,nb_raf,index_raf,Nx,Ny,lc,domaine):
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
            
            
    def countours_numerique(self,Solution_numerique):
        #Contour de la solution numérique avec pyvista pour le schéma centré
        plotter1 = MeshPlotter()
        nodes1, elements1 = plotter1.prepare_data_for_pyvista(self.mesh_obj)
        pv_mesh1 = pv.PolyData(nodes1, elements1)
        pv_mesh1['Solution numérique pour le schéma centré'] = Solution_numerique[:,0]
        pl = pvQt.BackgroundPlotter()
        #Message dans la console 
        print("Voir la solution numerique du schéma centré dans la fenêtre de PyVista")
        # Tracé du champ
        pl.add_mesh(pv_mesh1, show_edges=True, scalars="Solution numérique pour le schéma centré", cmap="jet")
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
        pv_mesh1['Solution numérique pour le schéma upwind'] = Solution_numerique[:,1]
        pl = pvQt.BackgroundPlotter()
        #Affichage
        #Message dans la console 
        print("Voir la solution numerique du schéma centré dans la fenêtre de PyVista")
        # Tracé du champ
        pl.add_mesh(pv_mesh1, show_edges=True, scalars="Solution numérique pour le schéma upwind", cmap="jet")
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
    def error_bars_chart(self,GCI_final,N_X,k_in_micron2,Type_maillage_index,type_maillage1,i_schema,Index_peclet,Peclet,L_C):
        #Schema centre & upwind
        if Type_maillage_index == 0 : 
            x = N_X
        else: 
            x = L_C
        y = k_in_micron2[:,Type_maillage_index,i_schema]
        xerr = k_in_micron2[1:,Type_maillage_index,i_schema]*(GCI_final[:]/2/100) 
        fig, ax = plt.subplots()
        ax.errorbar(x[1:], y[1:],yerr=xerr,fmt='.k',ecolor = 'red',color='black')
        ax.set_xlabel('Maillage')
        ax.set_ylabel('Temperature en (K)')
        if i_schema == 0 : 
            ax.set_title('U_num avec le schema centre pour un maillage '+type_maillage1[Type_maillage_index] + ',Pe=' + str(Peclet[Index_peclet]))
        else: 
            ax.set_title('U_num avec le schema upwind pour un maillage '+type_maillage1[Type_maillage_index]+  ',Pe=' + str(Peclet[Index_peclet]))
        plt.show()   
    def error_bars_chart2(self,U_val,Error,Incer_inputs,Incer_num,Incer_D,Solution_S,Type_maillage_index,type_maillage1,Index_peclet,Peclet):
        for i_schema in range(2): 
            #Schema centre & upwind
            x = ['U_input' ,'U_num','U_val','Error','U_exp']
            test = Solution_S[Index_peclet,Type_maillage_index,0]
            y = [test,test,test,test,test]
            xerr = [Incer_inputs[Index_peclet,Type_maillage_index,i_schema],Incer_num[Index_peclet,Type_maillage_index,i_schema],U_val[Index_peclet,Type_maillage_index,i_schema],Error[Index_peclet,Type_maillage_index,i_schema],Incer_D]
            fig, ax = plt.subplots()
            ax.errorbar(x, y,yerr=xerr,fmt='k',ecolor = 'red',color='black')
            ax.set_xlabel('Type incertitude')
            ax.set_ylabel('Temperature en (K)')
            if i_schema == 0 : 
                ax.set_title('Incertitudes (centre) pour un maillage '+type_maillage1[Type_maillage_index] + ',Pe=' + str(Peclet[Index_peclet]))
            else: 
                ax.set_title('Incertitudes (upwind) pour un maillage '+type_maillage1[Type_maillage_index] + ',Pe=' + str(Peclet[Index_peclet]))
            plt.show()      
        
    def numerical_incertitude(self,N_X,k_in_micron2,Type_maillage_index,type_maillage1,Index_peclet,Peclet,L_C):
        Inc_numerical = np.zeros(2)
        for i_schema in range(2):
            GCI_final = np.zeros(len(N_X)-1)
            if i_schema == 0 :
                formel_order = 2
            else: 
                formel_order = 1
            if Type_maillage_index == 0 : 
                r = N_X[len(N_X)-1]/N_X[len(N_X)-2]
            else: 
                r = L_C[len(N_X)-2]/L_C[len(N_X)-1]
            
            L = len(k_in_micron2[:,Type_maillage_index,0]) -1
            
            # L = len(N_X) -1 
            #Calcul de l'ordre observe 
            Order_P = np.abs(np.log((k_in_micron2[L,Type_maillage_index,i_schema] - k_in_micron2[L-1,Type_maillage_index,i_schema])/(k_in_micron2[L-1,Type_maillage_index,i_schema]-k_in_micron2[L-2,Type_maillage_index,i_schema]))/np.log(r))
            #Plotting GRID RESOLUTION
            if Type_maillage_index == 0 : 
                self.plot_grid_convergence(k_in_micron2,Order_P,Type_maillage_index,type_maillage1,Index_peclet,Peclet,N_X,i_schema) 
            else: 
                self.plot_grid_convergence(k_in_micron2,Order_P,Type_maillage_index,type_maillage1,Index_peclet,Peclet,L_C,i_schema) 
            # pour 1 % d'erreur   
            order_kk = (0.01*formel_order+formel_order)
            print("Observed order : "+str(Order_P))
            if (Order_P > 0.9*formel_order) and ( order_kk > Order_P) : 
                test = True
            else: 
                test = False 
            
            if test == True:
                print("Observed and formel orders are equal -> Numerical incertitude is zero, Extrapolation of richardson can be performed!")
                Inc_numerical[i_schema] = 0 
            else:
                #Implementation de la methode GCI 
                GCI_inter = abs((Order_P-formel_order)/formel_order)
                #Verification de la region asymptotique et implementation de GCI 
                if GCI_inter <= 0.1 : 
                    F_s = 1.15
                    P = formel_order
                else: 
                    F_s =3 
                    P= min(max(0.5,Order_P),formel_order)
                #Calcul de GCI 
                for i in range(len(N_X)-1): 
                    GCI_final[i] = F_s/((r**P)-1)*abs((k_in_micron2[L-3+i,Type_maillage_index,i_schema] - k_in_micron2[L-2+i,Type_maillage_index,i_schema])/k_in_micron2[L-3+i,Type_maillage_index,i_schema]) *100
        
                #Recherche de l'asym maximal
                Asymp = GCI_final[len(GCI_final)-2]/((r**Order_P)*GCI_final[len(GCI_final)-1])
                index_choice = 0 
                if Asymp >= 0.95: #Hypothese
                    print('We have reached the asymptotic region, GCI can be performed!')
                    print('Numerical error % +/- :',str(GCI_final[len(GCI_final)-1]/2))
                    self.error_bars_chart(GCI_final,N_X,k_in_micron2,Type_maillage_index,type_maillage1,i_schema,Index_peclet,Peclet,L_C)
                    if type_maillage1[Type_maillage_index] == 'QUAD' :
                        print("Mesh refinement used configuration : "+str(N_X)) 
                    else: 
                        print("Mesh refinement used configuration : "+str(L_C))
                        
                    
                else: 
                    print('Asymptotic region not reached, GCI can not be performed, mesh refinement is required!') 
                    
                
                Inc_numerical[i_schema]= k_in_micron2[L,Type_maillage_index,i_schema]*(GCI_final[L-1]*0.01)
                
        index_choice = int(input("Please choose the mesh for the inputs incertitude calculations : "))
        return Inc_numerical,N_X[index_choice]
    
    def plot_grid_convergence(self,k_in_micron2,Order_P,Type_maillage_index,type_maillage1,Index_peclet,Peclet,N_X,i_schema):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        text = AnchoredText('P = ' + str(round(Order_P,2)), loc='lower left')
        ax.plot(N_X,k_in_micron2[:,Type_maillage_index,i_schema] ,'-ko',label='Resultats numerique')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(b=True, which='minor', color='grey', linestyle='--')
        ax.grid(b=True, which='major', color='k', linestyle='--')
        plt.ylabel('Temperature en (DegK)')
        plt.xlabel('Resolution du grid')
        type_solveur = ["schéma centré","schéma upwind"]
        if i_schema == 0 : 
            String = "Analyse d incertitude pour un schema centre avec " + type_maillage1[Type_maillage_index] + " (Peclet =" + str(Peclet[Index_peclet]) + ")"
        else: 
            String = "Analyse d incertitude pour un schema upwind avec " + type_maillage1[Type_maillage_index] + " (Peclet =" + str(Peclet[Index_peclet]) + ")"
        plt.title(label=String ,fontsize=11,color="black")
        plt.legend()
        ax.add_artist(text)
        plt.show()
        
        return
    def estimation_erreur_discr(self,Incer_value_moy,Refinement,N_X,Type_maillage_index,L_C,Index_peclet,Peclet,type_maillage1):
        for i_schema in range(2): 
            #on recalcul l'ordre 
            if Type_maillage_index == 0 : 
                r = N_X[len(N_X)-1]/N_X[len(N_X)-2]
            else: 
                r = L_C[len(N_X)-2]/L_C[len(N_X)-1]
            L = len(Incer_value_moy[:,Type_maillage_index,0]) -1
            #Calcul de l'ordre observe 
            Order_P = np.abs(np.log((Incer_value_moy[L,Type_maillage_index,i_schema] - Incer_value_moy[L-1,Type_maillage_index,i_schema])/(Incer_value_moy[L-1,Type_maillage_index,i_schema]-Incer_value_moy[L-2,Type_maillage_index,i_schema]))/np.log(r))
            
            F_richardson = Incer_value_moy[L,Type_maillage_index,i_schema] + ((Incer_value_moy[L,Type_maillage_index,i_schema]-Incer_value_moy[L-1,Type_maillage_index,i_schema])/((r**Order_P)-1))
            Discre_error = np.abs(Refinement[Index_peclet,Type_maillage_index,i_schema] -F_richardson)/F_richardson
            
            #Creation d'un vecteur avec L +1 raffinement 
            Vect_refi = np.zeros(L+2)
            N_X11 = np.zeros(L+2)
            L_C11 = np.zeros(L+2)
            #Vect_refi = np.hstack(Incer_value_moy[:,Type_maillage_index,i_schema], Refinement[Index_peclet,Type_maillage_index,i_schema])
            Vect_refi[0:L+1] = Incer_value_moy[:,Type_maillage_index,i_schema]
            Vect_refi[L+1] = Refinement[Index_peclet,Type_maillage_index,i_schema]
            
            if Type_maillage_index == 0 : 
                N_X11[0:L+1] = N_X
                N_X11[L+1] = N_X[L]*2
                #N_X11 = np.hstack(N_X,N_X[L]*2)
            else: 
                L_C11[0:L+1] = L_C
                L_C11[L+1] = L_C[L]/2
                # L_C11 = np.hstack(L_C,L_C[L]/2)
                
            #Plotting     
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            text = AnchoredText('Disc_Error = ' + str(round(Discre_error,4)), loc='lower left',prop=dict(size=10))
            if Type_maillage_index == 0 :
                ax.plot(N_X11,Vect_refi,'ko',label='Resultats numerique')
                ax.plot(N_X11[-1], F_richardson,'ko',color='red',label='Extrapolation de Richardson')
            else: 
                ax.plot(L_C11,Vect_refi,'ko',label='Resultats numerique')
                ax.plot(L_C11[-1], F_richardson,'ko',color='red',label='Extrapolation de Richardson')
            
            
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            ax.grid(b=True, which='minor', color='grey', linestyle='--')
            ax.grid(b=True, which='major', color='k', linestyle='--')
            plt.ylabel('Temperature en (DegK)',fontsize=11)
            plt.xlabel('Resolution du grid',fontsize=11)
        
            if i_schema == 0 : 
                String = "Estimation de l'erreur de discretisation (centre) " + type_maillage1[Type_maillage_index] + " (Peclet =" + str(Peclet[Index_peclet]) + ")"
            else: 
                String = "Estimation de l'erreur de discretisation (upwind) " + type_maillage1[Type_maillage_index] + " (Peclet =" + str(Peclet[Index_peclet]) + ")"
            plt.title(label=String ,fontsize=11,color="black")
            plt.legend(fontsize=11)
            ax.add_artist(text)
            plt.show()
            
            
        
        return 
    
    def input_incertitude(self,N_X,Incer_value_moy1,N,Type_maillage_index,type_maillage1,Index_peclet,Peclet): 
        Variance = np.zeros(2)
        
        Variance[0] = statistics.variance(Incer_value_moy1[:,Type_maillage_index,0])
        Variance[1] = statistics.variance(Incer_value_moy1[:,Type_maillage_index,1])
        ecart_type = np.sqrt(Variance) *2
        #Plotting CDF for the results    
        self.cdf_impl(Incer_value_moy1,N,Type_maillage_index,type_maillage1,ecart_type,Index_peclet,Peclet)
        self.plot_PDF_incer_input(Incer_value_moy1,ecart_type,N,Type_maillage_index,type_maillage1,Index_peclet,Peclet)
        return ecart_type
    
    def plot_PDF_incer_input(self,Incer_value_moy1,ecart_type,N,Type_maillage_index,type_maillage1,Index_peclet,Peclet):
          for i in range(2):   
              mu, sigma = norm.fit(Incer_value_moy1[:,Type_maillage_index,i])
              text = AnchoredText('U_input = 2*Sig =  ' + str(round(ecart_type[0],4)), loc='upper right',prop=dict(size=10))
              fig = plt.figure()
              ax = fig.add_subplot(111)
              _ ,bins, _ = ax.hist(Incer_value_moy1[:,Type_maillage_index,i] ,density=True)
              mu, sigma = scipy.stats.norm.fit(Incer_value_moy1[:,Type_maillage_index,i])
              best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
              ax.plot(bins, best_fit_line)
              if i == 0 :
                  plt.title('PDF de la temperature pour le schema centre avec un maillage '+type_maillage1[Type_maillage_index]+ ',Pe=' + str(Peclet[Index_peclet]),fontsize=11)
              else: 
                  plt.title('PDF de la temperature pour le schema upwind avec un maillage '+type_maillage1[Type_maillage_index]+ ',Pe=' + str(Peclet[Index_peclet]),fontsize=11)
              ax.add_artist(text)
              plt.ylabel('SRQ',fontsize=11)
              plt.xlabel('Densite de probabilite',fontsize=11) 
              plt.show()
        
            
    
            
    def variable_alea(self,alea_moy,alea_sig,N,Index_peclet,Peclet,V,Lc,Cp,rho): 
           #utiliser une distribution normale 
            # alea_moy=5
            # alea_sig=.5
            # N=200
            eva = np.random.normal(alea_moy,alea_sig,N)
            data = pd.DataFrame(eva)
            data.columns =['Nombre_de_Peclet']
            sns.set_style('white')
            # f = Fitter(eva,distributions= get_common_distributions())
            # f.fit()
            # f.summary()
            sns.set_context("paper", font_scale = 2)
            sns.displot(data=data, x="Nombre_de_Peclet", kind="hist",kde=True, bins = 10, aspect = 1.5).set(title='Normal distribution for Peclet, case:'+ ',Pe=' + str(Peclet[Index_peclet]))
            
            plt.show()
            # # tracer l'histogramme normalisé
            # plt.hist(eva, 40, density=True)
            eva1 = V*Lc*Cp*rho/eva
            # plt.xlabel('Variable incertaine aléatoire ')
            # plt.ylabel('Distribution')
            # plt.title('Histogramme pour la variable ')
            # plt.show()
            return eva
        
    def variable_epis(self,epi_min,epi_max,M):
            #utiliser une distribution uniforme
            # epi_min=1.25
            # epi_max=1.75
            # M=50
            eve = np.random.uniform(epi_min,epi_max,M)
            # tracer l'histogramme normalisé
            plt.hist(eve, 20, density=True)
            plt.xlabel('Variable incertaine épistémique X2')
            plt.ylabel('Distribution')
            plt.title('Histogramme pour la variable X2')
            plt.show()
            
    def cdf_impl(self,k_in_micron2,N,Type_maillage_index,type_maillage1,ecart_type,Index_peclet,Peclet):
            #For centered scheme
            data = pd.DataFrame(k_in_micron2[:,Type_maillage_index,0])
            data.columns =['Temperature_K']
            variance = statistics.variance(k_in_micron2[:,Type_maillage_index,0])
            fig = sns.ecdfplot(data=data,x='Temperature_K')#.set(title='CDF de la temperature pour le schema centre avec un maillage '+type_maillage1[Type_maillage_index]+ ',Pe=' + str(Peclet[Index_peclet]))
            fig.axes.set_title('CDF de la temperature pour le schema centre avec un maillage '+type_maillage1[Type_maillage_index]+ ',Pe=' + str(Peclet[Index_peclet]),fontsize=11)
            #plt.text(3+0.2, 4.5, 'Variance  = ' + str(variance), horizontalalignment='left', size='medium', color='black', weight='semibold')
            #plt.text(x=0.01, y=0.01, s='U_input =2*Sig=  '+str(ecart_type[0]), color='green')
            plt.grid()
           
            plt.show()
             #For upwind scheme
            data1 = pd.DataFrame(k_in_micron2[:,Type_maillage_index,1])
            data1.columns =['Temperature_K']
            variance1 = statistics.variance(k_in_micron2[:,Type_maillage_index,1])
            fig1 = sns.ecdfplot(data=data1,x='Temperature_K')#.set(title='CDF de la temperature pour le schema upwind avec un maillage '+type_maillage1[Type_maillage_index]+ ',Pe=' + str(Peclet[Index_peclet]))
            fig1.axes.set_title('CDF de la temperature pour le schema upwind avec un maillage '+type_maillage1[Type_maillage_index]+ ',Pe=' + str(Peclet[Index_peclet]),fontsize=11)
            #plt.text(x=0.01, y=0.01, s='U_input =2*Sig=  '+str(ecart_type[1]), color='green')
            #plt.text(0, 0, 'Variance  = ' + str(variance1), horizontalalignment='left', size='medium', color='black', weight='semibold')
            plt.grid()
           
            plt.show()   
    def estimation_u_d(self,V):
       V = V+0.1*V
       original = np.array([V,V,V,V,V,V,V,V,V,V])
       noise = np.random.normal(0, 0.05, original.shape)
       new_signal = original + noise
      
       plt.plot(new_signal,label='noisy')
       plt.plot(original,label='original')
       plt.legend()
       plt.title('données expérimentales avec parasite',fontsize=11)
      
       plt.xlabel('Nombre d experiences',fontsize=11)
       plt.ylabel('Temperature en (K)',fontsize=11)
       plt.grid()
       plt.show() 
       
       variance = statistics.variance(noise)
       ecart_type_D = np.sqrt(variance) *2 # Pour un interval de confiance de 95%
       
       return ecart_type_D  ,V       
            
    def variance(data):
     # Number of observations
         n = len(data)
     # Mean of the data
         mean = sum(data) / n
         deviations = [(x - mean) ** 2 for x in data]
         # Variance
         variance = sum(deviations) / n
         ecart_type = np.sqrt(variance)
         return variance ,ecart_type               
            
    def main_propagation (self,N_X,Incer_value_moy):
        
        Incertidude_numerique=self.numerical_incertitude(self,N_X,Incer_value_moy)
        
        
        #Configuration initiale des maillages
        Nx = 10 #Nombres de mailles en pour le premier pas de raffinement
        Ny = 10 #Nombres de mailles en y pour le premier pas de raffinement
        lc = 0.3 #Taille de maille pour le premier pas de raffinement
       # mesh_obj = self.mesh_creation(Type_Maillage,nb_raf,index_raf,Nx,Ny,lc,self.domaine)
        
        
    
    
    
    

