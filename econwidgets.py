# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:33:59 2024

@author: sb636
"""

import ipywidgets as widgets
import numpy as np
import pandas as pd
from scipy.stats import norm, binom, lognorm, t, pearsonr
import scipy.special as ss
import re

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.colors import ListedColormap
from IPython.display import display, HTML

#------------------------------------------------------------------------------
# Mathematics widgets

def linear_widget(xMax_init = 15, yMax_init = 15, a_init = 2, b_init = -4, Aflag_init = True, 
                  xA_init = 4, yA_init = 7, Bflag_init = False, xB_init = 10, yB_init = 10, 
                  eqFlag1_init = True, eqFlag2_init = False):
    
    # Declare widgets for interactive input
    xMax_slider = widgets.IntSlider(min=5,
                                 max=100,
                                 step=1,
                                 description=r'Max. $x$:',
                                 value = xMax_init,
                                 continuous_update =True)
    yMax_slider = widgets.IntSlider(min=5,
                                 max=100,
                                 step=1,
                                 description=r'Max. $y$:',
                                 value = yMax_init,
                                 continuous_update =True)
    a_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$a$:',
                                 value = a_init,
                                 continuous_update =True)
    b_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$b$:',
                                 value=b_init,
                                 continuous_update =True)
    Aflag_check = widgets.Checkbox(value = Aflag_init,
                                   description='Include point A',
                                   disabled=False,
                                   indent=True) 
    xA_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$x_A$:',
                                 value = xA_init,
                                 continuous_update =True)
    yA_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$y_A$:',
                                 value = yA_init,
                                 continuous_update =True)
    Bflag_check = widgets.Checkbox(value = Bflag_init,
                                   description='Include point B',
                                   disabled=False,
                                   indent=True)
    xB_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$x_B$:',
                                 value = xB_init,
                                 continuous_update =True)
    yB_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$y_B$:',
                                 value = yB_init,
                                 continuous_update =True)
    eqFlag1_check = widgets.Checkbox(value = eqFlag1_init,
                                   description='Show Intercept/Slope',
                                   disabled=False,
                                   indent=True)
    eqFlag2_check = widgets.Checkbox(value = eqFlag2_init,
                                   description=r'Use $(A,B)$ for Slope',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    widgets.jslink((xMax_slider,'value'),(yMax_slider,'value'))

    def linear_plot(xMax, yMax, a, b, Aflag, xA, yA, Bflag, xB, yB, eqFlag1, eqFlag2):

        # create an 'x' vector, calculate 'y' vector
        x = np.arange(-xMax,xMax)
        y = a*x + b
        if b < 0:
            sig = '-'
        else:
            sig = '+'

        # Create figure, plot linear linear_function
        fig, ax = plt.subplots(figsize=(20,10))
        ax.plot(x, y,'b', linewidth=2, alpha=0.6,
                    label=r'$\quad y = {:.2f} x {:s} {:.2f}$'.format(a,sig,abs(b)))

        # Add markers for the points if requested
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        if Aflag is True:
            ax.scatter(xA, yA, s=mrkrSize, c='k',
                       label=r'$A \; ({:.2f},{:.2f})$'.format(xA, yA))
        if Bflag is True:
            ax.scatter(xB, yB, s=mrkrSize, c='r', alpha=0.6,
                       label=r'$B \; ({:.2f},{:.2f})$'.format(xB, yB))

        # Add intercept/slope information, with dotted lines if requested
        if eqFlag2 is True:
            y1 = yA
            y2 = yB
        else:
            y1 = a*xA + b
            y2 = a*xB + b

        # Format slope data and location depending of sign of slope
        if y2 > y1:
            offset = -1.5
        else:
            offset = 0.5
            
        if xB > xA:
            dy = y2-y1
        else:
            dy = y1-y2

        if eqFlag1 is True:
            ax.plot([0.5,0.5],[0,b],'r--',linewidth=1)
            ax.plot([0.25,0.75],[b,b],'r--',linewidth=1)
            ax.plot([xA,xB],[y1,y1],'r--',linewidth=1)
            ax.plot([xB,xB],[y1,y2],'r--',linewidth=1)
            ax.annotate(r'$b = {:.2f}$'.format(b),[0.5,b/2], xytext = [1,b/2],
                        xycoords ='data', fontsize = 25, color = 'r', alpha = 0.8,
                        clip_on = True)
            ax.annotate(r'$\Delta x = {:.2f}$'.format(abs(xA-xB)),
                        [(xA+xB)/2,y1], xytext = [(xA+xB)/2,y1+offset],
                        xycoords ='data', fontsize = 25, color = 'r', alpha = 0.8,
                        clip_on = True)
            ax.annotate(r'$\Delta y = {:.2f}$'.format(dy),
                        [xB,(y1+y2)/2], xytext = [xB+0.5,(y1+y2)/2],
                        xycoords ='data', fontsize = 25, color = 'r', alpha = 0.8,
                        clip_on = True)

        # Add legend and format axes to look nice
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = -yMax)
        ax.set_xlim(right = xMax, left = -xMax)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
    
    out = widgets.interactive_output(linear_plot, {'xMax': xMax_slider,
                                                   'yMax': yMax_slider,
                                                   'a': a_slider,
                                                   'b': b_slider,
                                                   'Aflag': Aflag_check,
                                                   'xA': xA_slider, 
                                                   'yA': yA_slider,                                                       
                                                   'Bflag': Bflag_check,
                                                   'xB': xB_slider, 
                                                   'yB': yB_slider, 
                                                   'eqFlag1': eqFlag1_check,
                                                   'eqFlag2': eqFlag2_check})

    output = widgets.VBox([out,
                  widgets.HBox([xMax_slider,
                                yMax_slider,
                                eqFlag2_check]),
                  widgets.HBox([a_slider, 
                                b_slider,
                                eqFlag1_check]),
                  widgets.HBox([xA_slider,
                                yA_slider,
                                Aflag_check]),
                  widgets.HBox([xB_slider,
                                yB_slider,
                                Bflag_check])])
    display(output)
    
    
def system_widget(xMax_init = 15, yMax_init = 15, a_init = 3, b_init = -4, c_init = -7, 
                  d_init = -2, e_init = 6, f_init = 8):
    
    # Declare widgets for interactive input
    xMax_slider = widgets.IntSlider(min=5,
                                 max=100,
                                 step=1,
                                 description=r'Max. $x$:',
                                 value = xMax_init,
                                 continuous_update =True)
    yMax_slider = widgets.IntSlider(min=5,
                                 max=100,
                                 step=1,
                                 description=r'Max. $y$:',
                                 value = yMax_init,
                                 continuous_update =True)
    a_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'Eq1. $x$ param.:',
                                 value = a_init,
                                 continuous_update =True)
    b_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'Eq1. $y$ param.:',
                                 value=b_init,
                                 continuous_update =True)
    c_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= 'Eq1. constant:',
                                 value=c_init,
                                 continuous_update =True)
    d_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'Eq2. $x$ param.',
                                 value=d_init,
                                 continuous_update =True)
    e_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'Eq2. $y$ param.',
                                 value=e_init,
                                 continuous_update =True)
    f_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= 'Eq1. constant:',
                                 value=f_init,
                                 continuous_update =True)
    
    # Link widgets as required
    widgets.jslink((xMax_slider,'value'),(yMax_slider,'value'))

    def system_plot(xMax, yMax, a, b, c, d, e, f):

        # Get y=f(x) function parameters from implicit equation
        a1 = -a/b
        b1 = c/b
        a2 = -d/e
        b2 = f/e

        # create an 'x' vector, calculate 'y' vectors for equations
        x = np.arange(-xMax,xMax)
        y1 = a1*x + b1
        y2 = a2*x + b2
        if b1 < 0:
            sig1 = '-'
        else:
            sig1 = '+'
        if b2 < 0:
            sig2 = '-'
        else:
            sig2 = '+'

        # Create figure, plot linear linear_function
        fig, ax = plt.subplots(figsize=(20,10))
        ax.plot(x, y1,'b', linewidth=2, alpha=0.6,
                    label=r'$\quad y = {:.2f} x {:s} {:.2f}$'.format(
                            a1,sig1,abs(b1)))
        ax.plot(x, y2,'r', linewidth=2, alpha=0.6,
                    label=r'$\quad y = {:.2f} x {:s} {:.2f}$'.format(
                            a2,sig2,abs(b2)))

        # Add markers for the solution, if it exists
        if a1 == a2:
            if b1 == b2:
                solLabel = 'Infinite solutions'
            else:
                solLabel = 'No solution'
        else:
            solLabel = 'Unique Solution'

            xSol = (b2 - b1)/(a1 - a2)
            ySol = a1*xSol + b1

            mrkrSize = 2*rcParams['lines.markersize'] ** 2
            ax.scatter(xSol, ySol, s=mrkrSize, c='k',
                       label=r'Solution $({:.2f},{:.2f})$'.format(xSol, ySol))

        # Add annotations
        if b < 0:
            sigEq1 = '-'
        else:
            sigEq1 = '+'
        if e < 0:
            sigEq2 = '-'
        else:
            sigEq2 = '+'
        ax.annotate(solLabel,[0.75,0], xytext = [0.75,0.05],
                    xycoords ='axes fraction', fontsize = 25, clip_on = True)
        ax.annotate(r"$\{$",[0,30], fontsize=80, xycoords='axes points',alpha = 0.6)
        ax.annotate(r'{:}x {:s} {:}y = {:}'.format(a,sigEq1,abs(b),c),[40,60],
                    xycoords ='axes points', color = 'b', alpha = 0.6,
                    fontsize = 25)
        ax.annotate(r'{:}x {:s} {:}y = {:}'.format(d,sigEq2,abs(e),f),[40,30],
                    xycoords ='axes points',  color = 'r', alpha = 0.6,
                    fontsize = 25)

        # Add legend and format axes to look nice
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = -yMax)
        ax.set_xlim(right = xMax, left = -xMax)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()

    out = widgets.interactive_output(system_plot, {'xMax': xMax_slider,
                                                   'yMax': yMax_slider,
                                                   'a': a_slider,
                                                   'b': b_slider,
                                                   'c': c_slider,
                                                   'd': d_slider, 
                                                   'e': e_slider,                                                       
                                                   'f': f_slider})

    output = widgets.VBox([out,
                  widgets.HBox([xMax_slider,
                                yMax_slider]),
                  widgets.HBox([a_slider, 
                                b_slider,
                                c_slider]),
                  widgets.HBox([d_slider,
                                e_slider,
                                f_slider])])
    display(output)
    
def quadratic_widget(xMax_init = 15, yMax_init = 15, a_init = 1, b_init = 0, c_init = 0, 
                     a2_init = 1, b2_init = 0, c2_init = 0, 
                  altFlag_init = False, rootFlag_init = False, maxFlag_init = False):
    
    # Declare widgets for interactive input
    xMax_slider = widgets.IntSlider(min=5,
                                 max=100,
                                 step=1,
                                 description=r'Max. $x$:',
                                 value = xMax_init,
                                 continuous_update =True)
    yMax_slider = widgets.IntSlider(min=5,
                                 max=100,
                                 step=1,
                                 description=r'Max. $y$:',
                                 value = yMax_init,
                                 continuous_update =True)
    a_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$a$ param.:',
                                 value = a_init,
                                 continuous_update =True)
    b_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$b$ param.:',
                                 value=b_init,
                                 continuous_update =True)
    c_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$c$ param.:',
                                 value=c_init,
                                 continuous_update =True)
    a2_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$a$ param.:',
                                 value = a_init,
                                 continuous_update =True)
    b2_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$b$ param.:',
                                 value=b_init,
                                 continuous_update =True)
    c2_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$c$ param.:',
                                 value=c_init,
                                 continuous_update =True)
    altFlag_check = widgets.Checkbox(value = altFlag_init,
                                   description='Include comparison',
                                   disabled=False,
                                   indent=True)    
    rootFlag_check = widgets.Checkbox(value = rootFlag_init,
                                   description='Include roots',
                                   disabled=False,
                                   indent=True)
    maxFlag_check = widgets.Checkbox(value = maxFlag_init,
                                   description='Include extremum',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    widgets.jslink((xMax_slider,'value'),(yMax_slider,'value'))

    def quadratic_plot(yMax,xMax, a, b, c, a2, b2, c2, altFlag, rootFlag, maxFlag):

        # create an 'x' vector, calculate 'y' vector
        x = np.arange(-xMax,xMax,2*xMax/500)
        y = a*x**2 + b*x + c
        if b < 0:
            sigB = '-'
        else:
            sigB = '+'

        if c < 0:
            sigC = '-'
        else:
            sigC = '+'
        
        # generate comparator equation
        y_alt = a2*x**2 + b2*x + c2
        if b2 < 0:
            sigB_alt = '-'
        else:
            sigB_alt = '+'

        if c2 < 0:
            sigC = '-'
        else:
            sigC_alt = '+'

        # Create figure, plot quadratic function
        fig, ax = plt.subplots(figsize=(20,10))
        if altFlag == True:
            ax.plot(x, y_alt,'b--', linewidth=2, alpha=0.75,
                    label=r'$\quad y = {:.2f} x^2 {:s}{:.2f}x {:s}{:.2f}$'.format(
                    a2,sigB_alt,abs(b2),sigC_alt,abs(c2)))
        
        ax.plot(x, y,'b', linewidth=2, alpha=0.6,
                    label=r'$\quad y = {:.2f} x^2 {:s}{:.2f}x {:s}{:.2f}$'.format(
                    a,sigB,abs(b),sigC,abs(c)))

        # Add markers for the roots if requested and if they exist
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        if rootFlag is True:
            discr = b**2-4*a*c
            if discr < 0:
                solLabel = r'D = {:.2f} - No real roots'.format(discr)
            elif discr == 0:
                solLabel = 'D = 0 - One real root'
                xR = -b/(2*a)
                ax.scatter(xR, 0, s=mrkrSize, c='k',
                           label=r'$y=0 \; , \; ({:.2f},{:.2f})$'.format(xR, 0))
            else:
                solLabel = 'D = {:.2f} - Two real roots'.format(discr)
                xR1 = (-b - discr**0.5)/(2*a)
                xR2 = (-b + discr**0.5)/(2*a)
                ax.scatter(xR1, 0, s=mrkrSize, c='k',
                           label=r'$y=0 \; , \; ({:.2f},{:.2f})$'.format(xR1, 0))
                ax.scatter(xR2, 0, s=mrkrSize, c='k', alpha=0.6,
                           label=r'$y=0 \; , \; ({:.2f},{:.2f})$'.format(xR2, 0))

            ax.annotate(solLabel,[0.75,0], xytext = [0.66,0.05],
                        xycoords ='axes fraction', fontsize = 25, clip_on = True)

        # Add maximum information, with dotted line, if requested
        if maxFlag is True:
            xM = -b/(2*a)
            yM = a*xM**2 + b*xM + c
            if yM > 0:
                offset = -1.5
            else:
                offset = 1
            ax.scatter([xM,xM], [0,yM], s=mrkrSize, c='r',
                       label=r'$y_{ext}$' + '$\; , \; ({:.2f},{:.2f})$'.format(
                       xM, yM))
            ax.annotate(r'$x_{ext} = $'+r'$-\frac{b}{2a}$',[xM,0], 
                        xytext = [xM-1,offset],xycoords ='data', fontsize = 25, 
                        color = 'r', alpha = 0.6, clip_on = True)
            if not yM == 0:
                ax.plot([xM,xM],[0,yM],'r--',linewidth=1)

        # Add legend and format axes to look nice
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = -yMax)
        ax.set_xlim(right = xMax, left = -xMax)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()

    out = widgets.interactive_output(quadratic_plot, {'xMax': xMax_slider,
                                                      'yMax': yMax_slider,
                                                       'a': a_slider,
                                                       'b': b_slider,
                                                       'c': c_slider,
                                                       'a2': a2_slider,
                                                       'b2': b2_slider,
                                                       'c2': c2_slider,
                                                       'altFlag': altFlag_check,  
                                                       'rootFlag': rootFlag_check,
                                                       'maxFlag' : maxFlag_check})

    output = widgets.VBox([out,
                  widgets.HBox([xMax_slider,
                                yMax_slider,
                                rootFlag_check,
                                maxFlag_check]),
                  widgets.HBox([a_slider, 
                                b_slider,
                                c_slider]),
                  widgets.HBox([a2_slider, 
                                b2_slider,
                                c2_slider,
                                altFlag_check])])
    display(output)
    
def logarithm_widget(xMax_init = 5, yMax_init = 1, bases_init = ['None','10','None'], 
                     flags_init = [False,False,False], pts_init = [2,4,8]):
    
    # Declare widgets for interactive input
    xMax_slider = widgets.IntSlider(min=10**-6,
                                 max=10**6,
                                 step=1,
                                 description=r'Max. $x$:',
                                 value = xMax_init,
                                 continuous_update =True)
    yMax_slider = widgets.IntSlider(min=0.5,
                                 max=30,
                                 step=1,
                                 description=r'Max. $y$:',
                                 value = yMax_init,
                                 continuous_update =True)
    curve1_list = widgets.Dropdown(options=['None','e', '2', '10'],
                                value = bases_init[0],
                                description=r'base $N^{\circ}1$:',
                                disabled=False)
    curve2_list = widgets.Dropdown(options=['None','e', '2', '10'],
                                value = bases_init[1],
                                description=r'base $N^{\circ}2$:',
                                disabled=False)
    curve3_list = widgets.Dropdown(options=['None','e', '2', '10'],
                                value = bases_init[2],
                                description=r'base $N^{\circ}3$:',
                                disabled=False)
    x1_slider = widgets.FloatSlider(min=10**-6,
                                 max=10**6,
                                 description= r'$x_1$:',
                                 value = pts_init[0],
                                 continuous_update =True)
    x2_slider = widgets.FloatSlider(min=10**-6,
                                 max=10**6,
                                 description= r'$x_2$:',
                                 value = pts_init[1],
                                 continuous_update =True)
    x3_slider = widgets.FloatSlider(min=10**-6,
                                 max=10*6,
                                 description= r'$x_3$:',
                                 value = pts_init[2],
                                 continuous_update =True)
    pt1Flag_check = widgets.Checkbox(value = flags_init[0],
                                   description='Include pt. 1',
                                   disabled=False,
                                   indent=True)    
    pt2Flag_check = widgets.Checkbox(value = flags_init[1],
                                   description='Include pt. 2',
                                   disabled=False,
                                   indent=True)
    pt3Flag_check = widgets.Checkbox(value = flags_init[2],
                                   description='Include pt. 3',
                                   disabled=False,
                                   indent=True)

    def logarithmic_plot(yMax, xMax, curve1, curve2, curve3, pt1Flag, pt2Flag, pt3Flag, x1, x2, x3):
    
        # repackage inputs into lists
        bases = [curve1, curve2, curve3]
        flags = [pt1Flag, pt2Flag, pt3Flag]
        pts = [x1, x2, x3]
        
        # Prepare dicts of log functions and color codes
        logdict = {'e': lambda x: np.log(x) ,
                   '2': lambda x: np.log2(x),
                   '10': lambda x: np.log10(x)}

        colors = {'e': 'b',
                   '2': 'r',
                   '10': 'g'}

        # create an 'x' vector
        x = np.arange(1/500,xMax,xMax/500)

        # Create figure, plot logarithmic functions
        fig, ax = plt.subplots(figsize=(20,10))

        for base in bases:
            if not base == 'None':
                y = logdict[base](x)
                ax.plot(x, y,colors[base], linewidth=2, alpha=0.6,
                        label=r'$\quad y = \log_{'+'{:s}'.format(base)+r'}(x)$')

        # Add markers for the points if requested
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        base0 = bases[0]
        for flag in flags:
            xP = pts.pop(0)
            if flag is True and not base0 == 'None':
                yP = logdict[base0](xP) 
                ax.scatter(xP, yP, s=mrkrSize, c=colors[base0])
                ax.plot([xP,xP],[0,yP], colors[base0]+'--',linewidth=1)
                ax.plot([0,xP],[yP,yP], colors[base0]+'--',linewidth=1)
                ax.annotate('{:.2f}'.format(yP),[0,yP], xytext = [0.1,yP+0.1],
                            color=colors[base0], xycoords ='data', fontsize = 25, 
                            alpha=0.6, clip_on = True)

        # Add legend and format axes to look nice
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = -yMax)
        ax.set_xlim(right = xMax, left = 0)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()

    out = widgets.interactive_output(logarithmic_plot, {'xMax': xMax_slider,
                                                       'yMax': yMax_slider,
                                                       'curve1': curve1_list,
                                                       'curve2': curve2_list,
                                                       'curve3':  curve3_list,
                                                       'pt1Flag': pt1Flag_check,
                                                       'pt2Flag': pt2Flag_check,
                                                       'pt3Flag': pt3Flag_check,
                                                       'x1': x1_slider,
                                                       'x2': x2_slider,
                                                       'x3': x3_slider})
    
    output = widgets.VBox([out,
                  widgets.HBox([xMax_slider,
                                yMax_slider,
                                curve1_list,
                                curve2_list,
                                curve3_list]),
                  widgets.HBox([pt1Flag_check,
                                x1_slider]),
                  widgets.HBox([pt2Flag_check,
                                x2_slider]),
                  widgets.HBox([pt3Flag_check,
                                x3_slider])])
    display(output)
    
def exponential_widget(xMax_init = 5, yMax_init = 1, bases_init = ['None','10','None'], 
                     flags_init = [False,False,False], pts_init = [2,4,8], invert_init = False):
    
    # Declare widgets for interactive input
    xMax_slider = widgets.IntSlider(min=1,
                                 max=50,
                                 step=1,
                                 description=r'Max. $x$:',
                                 value = xMax_init,
                                 continuous_update =True)
    yMax_slider = widgets.IntSlider(min=0,
                                 max=10000,
                                 step=1,
                                 description=r'Max. $y$:',
                                 value = yMax_init,
                                 continuous_update =True)
    curve1_list = widgets.Dropdown(options=['None','e', '2', '10'],
                                value = bases_init[0],
                                description=r'base $N^{\circ}1$:',
                                disabled=False)
    curve2_list = widgets.Dropdown(options=['None','e', '2', '10'],
                                value = bases_init[1],
                                description=r'base $N^{\circ}2$:',
                                disabled=False)
    curve3_list = widgets.Dropdown(options=['None','e', '2', '10'],
                                value = bases_init[2],
                                description=r'base $N^{\circ}3$:',
                                disabled=False)
    x1_slider = widgets.FloatSlider(min=-50,
                                 max=50,
                                 description= r'$x_1$:',
                                 value = pts_init[0],
                                 continuous_update =True)
    x2_slider = widgets.FloatSlider(min=-50,
                                 max=50,
                                 description= r'$x_2$:',
                                 value = pts_init[1],
                                 continuous_update =True)
    x3_slider = widgets.FloatSlider(min=-50,
                                 max=50,
                                 description= r'$x_3$:',
                                 value = pts_init[2],
                                 continuous_update =True)
    pt1Flag_check = widgets.Checkbox(value = flags_init[0],
                                   description='Include pt. 1',
                                   disabled=False,
                                   indent=True)    
    pt2Flag_check = widgets.Checkbox(value = flags_init[1],
                                   description='Include pt. 2',
                                   disabled=False,
                                   indent=True)
    pt3Flag_check = widgets.Checkbox(value = flags_init[2],
                                   description='Include pt. 3',
                                   disabled=False,
                                   indent=True)
    invertFlag_check = widgets.Checkbox(value = invert_init,
                                   description='Invert x',
                                   disabled=False,
                                   indent=True)

    def exponential_plot(yMax, xMax, curve1, curve2, curve3, pt1Flag, pt2Flag, pt3Flag, x1, x2, x3,
                        invert):
    
        # repackage inputs into lists
        bases = [curve1, curve2, curve3]
        flags = [pt1Flag, pt2Flag, pt3Flag]
        pts = [x1, x2, x3]
        
        # Prepare dicts of log functions and color codes
        if invert is True:
            expdict = {'e': lambda x: np.exp(-x) ,
                       '2': lambda x: 2**(-x),
                       '10': lambda x: 10**(-x)}
            sig = '-'
        else:
            expdict = {'e': lambda x: np.exp(x) ,
                       '2': lambda x: 2**x,
                       '10': lambda x: 10**x}
            sig = ''
            
        colors = {'e': 'b',
                   '2': 'r',
                   '10': 'g'}

        # create an 'x' vector
        x = np.arange(-xMax,xMax,2*xMax/500)

        # Create figure, plot logarithmic function
        fig, ax = plt.subplots(figsize=(20,10))

        for base in bases:
            if not base == 'None':
                y = expdict[base](x)
                ax.plot(x, y,colors[base], linewidth=2, alpha=0.6,
                        label=r'$\quad y = {:s}^'.format(base) + '{' + '{:s}'.format(sig) + 'x}$')

        # Add markers for the points if requested
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        base0 = bases[0]
        for flag in flags:
            xP = pts.pop(0)
            if flag is True and not base0 == 'None':
                yP = expdict[base0](xP) 
                ax.scatter(xP, yP, s=mrkrSize, c=colors[base0])
                ax.plot([xP,xP],[0,yP], colors[base0]+'--',linewidth=1)
                ax.plot([0,xP],[yP,yP], colors[base0]+'--',linewidth=1)
                ax.annotate('{:.2f}'.format(yP),[0,yP], xytext = [0.1,yP+0.1],
                            color=colors[base0], xycoords ='data', fontsize = 25, 
                            alpha=0.6, clip_on = True)

        # Add legend and format axes to look nice
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = 0)
        ax.set_xlim(right = xMax, left = -xMax)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()

    out = widgets.interactive_output(exponential_plot, {'xMax': xMax_slider,
                                                       'yMax': yMax_slider,
                                                       'curve1': curve1_list,
                                                       'curve2': curve2_list,
                                                       'curve3':  curve3_list,
                                                       'pt1Flag': pt1Flag_check,
                                                       'pt2Flag': pt2Flag_check,
                                                       'pt3Flag': pt3Flag_check,
                                                       'x1': x1_slider,
                                                       'x2': x2_slider,
                                                       'x3': x3_slider,
                                                       'invert':invertFlag_check})
    
    output = widgets.VBox([out,
                  widgets.HBox([xMax_slider,
                                yMax_slider,
                                curve1_list,
                                curve2_list,
                                curve3_list]),
                  widgets.HBox([pt1Flag_check,
                                x1_slider,
                                invertFlag_check]),
                  widgets.HBox([pt2Flag_check,
                                x2_slider]),
                  widgets.HBox([pt3Flag_check,
                                x3_slider])])
    display(output)

def inverse_widget(xMax_init = 5, yMax_init = 1, bases_init = ['None','10','None'], 
                     flags_init = [False,False,False], pts_init = [2,4,8]):
    
    # Declare widgets for interactive input
    xMax_slider = widgets.IntSlider(min=10**-6,
                                 max=10**6,
                                 step=1,
                                 description=r'Max. $x$:',
                                 value = xMax_init,
                                 continuous_update =True)
    yMax_slider = widgets.IntSlider(min=0,
                                 max=30,
                                 step=1,
                                 description=r'Max. $y$:',
                                 value = yMax_init,
                                 continuous_update =True)
    curve1_list = widgets.Dropdown(options=['None','e', '2', '10'],
                                value = bases_init[0],
                                description=r'base $N^{\circ}1$:',
                                disabled=False)
    curve2_list = widgets.Dropdown(options=['None','e', '2', '10'],
                                value = bases_init[1],
                                description=r'base $N^{\circ}2$:',
                                disabled=False)
    curve3_list = widgets.Dropdown(options=['None','e', '2', '10'],
                                value = bases_init[2],
                                description=r'base $N^{\circ}3$:',
                                disabled=False)
    x1_slider = widgets.FloatSlider(min=-50,
                                 max=50,
                                 description= r'$x_1$:',
                                 value = pts_init[0],
                                 continuous_update =True)
    x2_slider = widgets.FloatSlider(min=-50,
                                 max=50,
                                 description= r'$x_2$:',
                                 value = pts_init[1],
                                 continuous_update =True)
    x3_slider = widgets.FloatSlider(min=-50,
                                 max=50,
                                 description= r'$x_3$:',
                                 value = pts_init[2],
                                 continuous_update =True)
    pt1Flag_check = widgets.Checkbox(value = flags_init[0],
                                   description='Include pt. 1',
                                   disabled=False,
                                   indent=True)    
    pt2Flag_check = widgets.Checkbox(value = flags_init[1],
                                   description='Include pt. 2',
                                   disabled=False,
                                   indent=True)
    pt3Flag_check = widgets.Checkbox(value = flags_init[2],
                                   description='Include pt. 3',
                                   disabled=False,
                                   indent=True)

    def inverse_plot(yMax, xMax, curve1, curve2, curve3, pt1Flag, pt2Flag, pt3Flag, x1, x2, x3):
    
        # repackage inputs into lists
        bases = [curve1, curve2, curve3]
        flags = [pt1Flag, pt2Flag, pt3Flag]
        pts = [x1, x2, x3]
        
        # Prepare dicts of log functions and color codes
        logdict = {'e': lambda x: np.log(x) ,
                  '2': lambda x: np.log2(x),
                  '10': lambda x: np.log10(x)}
        expdict = {'e': lambda x: np.exp(x) ,
                   '2': lambda x: 2**x,
                   '10': lambda x: 10**x}

        colors = {'e': 'b',
                   '2': 'r',
                   '10': 'g'}

        # create an 'x' vector
        xLog = np.arange(1/500,xMax,xMax/500) 
        xExp = np.arange(-xMax,xMax,2*xMax/500)

        # Create figure, plot logarithmic function
        fig, ax = plt.subplots(figsize=(20,10))

        for base in bases:
            ax.plot(xExp, xExp,'k--', linewidth=1, alpha=0.6)
            if not base == 'None':
                yLog = logdict[base](xLog)
                yExp = expdict[base](xExp)
                ax.plot(xExp, yExp,colors[base]+'--', linewidth=2, alpha=0.6,
                        label=r'$\quad y = {:s}^x$'.format(base))
                ax.plot(xLog, yLog,colors[base], linewidth=2, alpha=0.6,
                       label=r'$\quad y = \log_{'+'{:s}'.format(base)+r'}(x)$')

        # Add markers for the points if requested
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        base0 = bases[0]
        for flag in flags:
            xP = pts.pop(0)
            if flag is True and not base0 == 'None':
                yP = logdict[base0](xP) 
                ax.scatter(xP, yP, s=mrkrSize, c=colors[base0])
                ax.plot([xP,xP],[0,yP], colors[base0]+'--',linewidth=1)
                ax.plot([0,xP],[yP,yP], colors[base0]+'--',linewidth=1)
                ax.annotate('{:.2f}'.format(yP),[0,yP], xytext = [0.1,yP+0.1],
                            color=colors[base0], xycoords ='data', fontsize = 25, 
                            alpha=0.6, clip_on = True)
                ax.annotate('{:.2f}'.format(xP),[xP,0], xytext = [xP+0.1,0.1],
                            color=colors[base0], xycoords ='data', fontsize = 25, 
                            alpha=0.6, clip_on = True)
                
                ax.scatter(yP, xP, s=mrkrSize, c=colors[base0])
                ax.plot([yP,yP],[0,xP], colors[base0]+'--',linewidth=1)
                ax.plot([0,yP],[xP,xP], colors[base0]+'--',linewidth=1)
                ax.annotate('{:.2f}'.format(xP),[0,xP], xytext = [0.1,xP+0.1],
                            color=colors[base0], xycoords ='data', fontsize = 25, 
                            alpha=0.6, clip_on = True)
                ax.annotate('{:.2f}'.format(yP),[yP,0], xytext = [yP+0.1,0.1],
                            color=colors[base0], xycoords ='data', fontsize = 25, 
                            alpha=0.6, clip_on = True)

        # Add legend and format axes to look nice
        ax.axis('equal')
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = -yMax)
        ax.set_xlim(right = xMax, left = -xMax)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()

    out = widgets.interactive_output(inverse_plot, {'xMax': xMax_slider,
                                                       'yMax': yMax_slider,
                                                       'curve1': curve1_list,
                                                       'curve2': curve2_list,
                                                       'curve3':  curve3_list,
                                                       'pt1Flag': pt1Flag_check,
                                                       'pt2Flag': pt2Flag_check,
                                                       'pt3Flag': pt3Flag_check,
                                                       'x1': x1_slider,
                                                       'x2': x2_slider,
                                                       'x3': x3_slider})
    
    output = widgets.VBox([out,
                  widgets.HBox([xMax_slider,
                                yMax_slider,
                                curve1_list,
                                curve2_list,
                                curve3_list]),
                  widgets.HBox([pt1Flag_check,
                                x1_slider]),
                  widgets.HBox([pt2Flag_check,
                                x2_slider]),
                  widgets.HBox([pt3Flag_check,
                                x3_slider])])
    display(output)

def quadratic_slope_widget(xMax_init = 10, yMax_init = 15, a_init = 0.4, b_init = -6, c_init = 25, 
                     x1_init = 2.5, x2_init = 6, x1Flag_init = True, x2Flag_init = False):
    
    # Declare widgets for interactive input
    xMax_slider = widgets.IntSlider(min=5,
                                 max=100,
                                 step=1,
                                 description=r'Max. $x$:',
                                 value = xMax_init,
                                 continuous_update =False)
    yMax_slider = widgets.IntSlider(min=5,
                                 max=100,
                                 step=1,
                                 description=r'Max. $y$:',
                                 value = yMax_init,
                                 continuous_update =False)
    a_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$a$ param.:',
                                 value = a_init,
                                 continuous_update =False)
    b_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$b$ param.:',
                                 value=b_init,
                                 continuous_update =False)
    c_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$c$ param.:',
                                 value=c_init,
                                 continuous_update =False)
    x1_slider = widgets.FloatSlider(min=0,
                                 max=100,
                                 description= r'$x_1$:',
                                 value = x1_init,
                                 continuous_update =False)
    x2_slider = widgets.FloatSlider(min=0,
                                 max=100,
                                 description= r'$x_2$:',
                                 value=x2_init,
                                 continuous_update =False)
    x1Flag_check = widgets.Checkbox(value = x1Flag_init,
                                   description='Include point 1',
                                   disabled=False,
                                   indent=True)    
    x2Flag_check = widgets.Checkbox(value = x2Flag_init,
                                   description='Include point 2',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    widgets.jslink((xMax_slider,'value'),(x1_slider,'max'))
    widgets.jslink((xMax_slider,'value'),(x2_slider,'max'))
    
    def quadratic_function(yMax,xMax, a, b, c, x1, x2, x1Flag, x2Flag):

        # create an 'x' vector, calculate 'y' vector
        x = np.arange(-xMax,xMax,2*xMax/500)
        y = a*x**2 + b*x + c
        if b < 0:
            sigB = '-'
        else:
            sigB = '+'

        if c < 0:
            sigC = '-'
        else:
            sigC = '+'

        y1 = a*x1**2 + b*x1 + c
        y2 = a*x2**2 + b*x2 + c

        xTan1 = np.asarray([x1-0.5,x1+0.5])
        xTan2 = np.asarray([x2-0.5,x2+0.5])

        yTan1 = np.asarray([y1-0.5*(2*a*x1+b),y1+0.5*(2*a*x1+b)])
        yTan2 = np.asarray([y2-0.5*(2*a*x2+b),y2+0.5*(2*a*x2+b)])

        chordA = (y2-y1)/(x2-x1)
        chordB = y2-chordA*x2
        chordX = np.asarray([min(x1,x2)-0.5,max(x1,x2)+0.5])
        chordY = chordA*chordX + chordB

        # Create figure, plot quadratic function
        fig, ax = plt.subplots(figsize=(20,10))
        ax.plot(x, y,'b', linewidth=2, alpha=0.6,
                    label=r'$\quad y = {:.2f} x^2 {:s}{:.2f}x {:s}{:.2f}$'.format(
                    a,sigB,abs(b),sigC,abs(c)))

        # Add markers for the points if requested and chord
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        if x1Flag is True:
            ax.scatter(x1, y1, s=mrkrSize, c='r')        
            ax.plot(xTan1, yTan1,'r', linewidth=2, alpha=0.6)

        if x2Flag is True:
            ax.scatter(x2, y2, s=mrkrSize, c='r')        
            ax.plot(xTan2, yTan2,'r', linewidth=2, alpha=0.6)

        if x1Flag is True and x2Flag is True:
            ax.plot(chordX, chordY,'r--', linewidth=2, alpha=0.6)

        # Add legend and format axes to look nice
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = 0)
        ax.set_xlim(right = xMax, left = 0)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()

    out = widgets.interactive_output(quadratic_function, {'xMax': xMax_slider,
                                                          'yMax': yMax_slider,
                                                           'a': a_slider,
                                                           'b': b_slider,
                                                           'c': c_slider,
                                                           'x1': x1_slider,
                                                           'x2': x2_slider,
                                                           'x1Flag': x1Flag_check,  
                                                           'x2Flag': x2Flag_check})

    output = widgets.VBox([out,
                  widgets.HBox([xMax_slider,
                                yMax_slider,
                                x1Flag_check,
                                x2Flag_check]),
                  widgets.HBox([a_slider, 
                                b_slider,
                                c_slider]),
                  widgets.HBox([x1_slider, 
                                x2_slider,
                                ])])
    display(output)
    
def derivative_widget(xMin_init = -5, xMax_init = 5, yMin_init = -5, yMax_init = 5, 
                           a_init = 4,  b_init = -3, c_init = -6, d_init = 2, 
                           d1Flag_init = True, d2Flag_init = False, areaFlag_init = False):
    
    # Declare widgets for interactive input
    xMin_slider = widgets.IntSlider(min=-100,
                                 max=0,
                                 step=1,
                                 description=r'Min. $x$:',
                                 value = xMin_init,
                                 continuous_update =False)
    yMin_slider = widgets.IntSlider(min=-100,
                                 max=0,
                                 step=1,
                                 description=r'Min. $y$:',
                                 value = yMin_init,
                                 continuous_update =False)
    xMax_slider = widgets.IntSlider(min=0,
                                 max=100,
                                 step=1,
                                 description=r'Max. $x$:',
                                 value = xMax_init,
                                 continuous_update =False)
    yMax_slider = widgets.IntSlider(min=0,
                                 max=100,
                                 step=1,
                                 description=r'Max. $y$:',
                                 value = yMax_init,
                                 continuous_update =False)
    a_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$a$ param.:',
                                 value = a_init,
                                 continuous_update =False)
    b_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$b$ param.:',
                                 value=b_init,
                                 continuous_update =False)
    c_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$c$ param.:',
                                 value=c_init,
                                 continuous_update =False)
    d_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$d$ param.:',
                                 value=d_init,
                                 continuous_update =False)
    d1Flag_check = widgets.Checkbox(value = d1Flag_init,
                                   description='show $dy$',
                                   disabled=False,
                                   indent=True)    
    d2Flag_check = widgets.Checkbox(value = d2Flag_init,
                                   description='show $d^2y$',
                                   disabled=False,
                                   indent=True)
    areaFlag_check = widgets.Checkbox(value = areaFlag_init,
                                   description='show highlight',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
#     widgets.jslink((xMax_slider,'value'),(x1_slider,'max'))
#     widgets.jslink((xMax_slider,'value'),(x2_slider,'max'))
    
    def derivative_plot(xMin, xMax, yMin, yMax, a, b, c, d, d1Flag, d2Flag, areaFlag):

        # Create an 'x' vector
        x = np.arange(xMin,xMax,(xMax-xMin)/500)

        # Function and derivatives
        f = lambda x: a*x**3 + b*x**2 + c*x + d
        df = lambda x: 3*a*x**2 + 2*b*x + c
        d2f = lambda x: 6*a*x + 2*b

        # Turning points (if any)
        D = (2*b)**2 - 4*(3*a)*c
        if D > 0:
            roots = [(-2*b - D**0.5)/(2*3*a), (-2*b + D**0.5)/(2*3*a)]
        elif D == 0:
            roots = [(-2*b)/(2*3*a)]
        else:
            roots = []

        # Labels
        sig = []
        for param in [b,c,d]:
            if param >= 0:
                sig.append('+')
            else:
                sig.append('-')

        lbl_y = r'$y = {:.2f}x^3 {:s} {:.2f}x^2 {:s} {:.2f}x {:s} {:.2f}$'
        lbl_dy = ' = {:.2f}x^2 {:s} {:.2f}x {:s} {:.2f}$'
        lbl_d2y = ' = {:.2f}x {:s} {:.2f}$'

        # Create figure, plot function and derivatives as required
        fig, ax = plt.subplots(figsize=(20,10))
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        ax.plot(x, f(x),'b', linewidth=2, alpha = 0.6, label=lbl_y.format(
                a, sig[0], abs(b), sig[1], abs(c), sig[2], abs(d)))
        if d1Flag is True:
            ax.plot(x, df(x),'r', linewidth=2, alpha = 0.6, 
                    label=r'$\frac{dy}{dx}' + lbl_dy.format(
                        3*a, sig[0], abs(2*b), sig[1], abs(c)))

        if d2Flag is True:
            ax.plot(x, d2f(x),'g', linewidth=2, alpha = 0.6, 
                    label=r'$\frac{d^2y}{dx^2}' + lbl_d2y.format(
                        6*a, sig[0], abs(2*b)))

        # Add markers for the turning points, with dotted lines and areas
        if areaFlag is True:
            f_fillPos = np.where(df(x) > 0, 1e6, -1e6)
            f_fillNeg = np.where(df(x) < 0, 1e6, -1e6)      
            ax.fill(np.append(x,[xMax, xMin]),
                    np.append(f_fillPos,[-1e6,-1e6]),
                    'g',alpha = 0.15, label = r'$\frac{dy}{dx} > 0$')        
            ax.fill(np.append(x,[xMax, xMin]),
                    np.append(f_fillNeg,[-1e6,-1e6]),
                    'r',alpha = 0.15, label = r'$\frac{dy}{dx} < 0$')
            if roots:
                for root in roots:
                    ax.scatter([root,root], [0,f(root)], s=mrkrSize, c='k', alpha=0.6)
                    ax.plot([root,root], [0,f(root)],'k--',linewidth=1)

        # Add legend and format axes to look nice
        ax.legend(loc='lower center', frameon=False,prop={'size':20},ncol=3,
            bbox_to_anchor=(0.5, -0.25))

        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = yMin)
        ax.set_xlim(right = xMax, left = xMin)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()

    out = widgets.interactive_output(derivative_plot, {'xMin': xMin_slider,
                                                       'xMax': xMax_slider,
                                                       'yMin': yMin_slider,
                                                       'yMax': yMax_slider,
                                                       'a': a_slider,
                                                       'b': b_slider,
                                                       'c': c_slider,
                                                       'd': d_slider,
                                                       'd1Flag': d1Flag_check,  
                                                       'd2Flag': d2Flag_check,
                                                       'areaFlag': areaFlag_check})
    
    output = widgets.VBox([out,
                  widgets.HBox([xMin_slider,
                                xMax_slider,
                                yMin_slider,
                                yMax_slider]),
                  widgets.HBox([a_slider, 
                                b_slider,
                                c_slider,
                                d_slider]),
                  widgets.HBox([d1Flag_check,
                                d2Flag_check,
                                areaFlag_check])
                          ])
    display(output)
    
def multivar_widget(curve_init = 'sines', xMin_init = -2, xMax_init = 2, xVal_init = 0, yVal_init = 0, 
                      elev_init = 60, azim_init = 35, xCut_init = False, yCut_init = False):
    
    # Declare widgets for interactive input
    curve_list = widgets.Dropdown(options=['sines',
                                           'exps',
                                           'min example',
                                           'max example',
                                           'saddle example',
                                           'dirty min example'],
                                value = curve_init,
                                description='function:',
                                disabled=False)
    xMin_slider = widgets.IntSlider(min=-100,
                                 max=0,
                                 step=1,
                                 description=r'Min. $x$,$y$:',
                                 value = xMin_init,
                                 continuous_update =False)    
    xMax_slider = widgets.IntSlider(min=0,
                                 max=100,
                                 step=1,
                                 description=r'Max. $x$,$y$:',
                                 value = xMax_init,
                                 continuous_update =False)
    xVal_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 step=1,
                                 description=r'$x$ coord:',
                                 value = xVal_init,
                                 continuous_update =False)
    yVal_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 step=1,
                                 description=r'$y$ coord:',
                                 value = yVal_init,
                                 continuous_update =False)
    elev_slider = widgets.FloatSlider(min=-90,
                                 max=90,
                                 description= 'Elevation:',
                                 value = elev_init,
                                 continuous_update =False)
    azim_slider = widgets.FloatSlider(min=0,
                                 max=360,
                                 description= 'Azimuth:',
                                 value = azim_init,
                                 continuous_update =False)
    xCut_check = widgets.Checkbox(value = xCut_init,
                                   description='Fix $x$',
                                   disabled=False,
                                   indent=True)    
    yCut_check = widgets.Checkbox(value = yCut_init,
                                   description='Fix $y$',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    widgets.jslink((xMax_slider,'value'),(xVal_slider,'max'))
    widgets.jslink((xMin_slider,'value'),(xVal_slider,'min'))
    widgets.jslink((xMax_slider,'value'),(yVal_slider,'max'))
    widgets.jslink((xMin_slider,'value'),(yVal_slider,'min'))
    
    def multivar_plot(curve, xMin, xMax, xVal, yVal, elev, azim , xCut, yCut):
        
        # XOR on the cuts, to avoid double cuts.
        if xCut is True and yCut is True:
            xCut = False
            yCut = False

        fundict = {'exps': lambda x, y: 7*x*y/np.exp(x**2 + y**2),
                   'sines': lambda x, y: np.sin(2*x)*np.cos(2*y),
                   'min example': lambda x, y: 4*x**2 + 4*y**2,
                   'max example': lambda x, y: -4*x**2 - 4*y**2,
                   'saddle example': lambda x, y: -4*x**2 + 4*y**2,
                   'dirty min example': lambda x, y: x**2 + 3*x*y+ y**2,
                  }
        
        f = fundict[curve]

        xFull = np.linspace(xMin, xMax, 100)
        yFull = np.linspace(xMin, xMax, 100)
        XFull, YFull = np.meshgrid(xFull, yFull)
        ZFull = f(XFull, YFull)
        
        if xCut is True:
            resX = int(np.floor((xMax-xVal)/(xMax-xMin)*100))
            x = np.linspace(xVal, xMax, resX)
        else:
            x = xFull

        if yCut is True:
            resY = int(np.floor((yVal-xMin)/(xMax-xMin)*100))
            y = np.linspace(xMin, yVal, resY)
        else:
            y = yFull

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        zMin = 1.5*min(ZFull.flatten())
        zMax = 1.5*max(ZFull.flatten())

        # Create figure, plot function and derivatives as required
        fig = plt.subplots(figsize=(20,10))
        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, Z, cmap='viridis')
        if xCut is True:
            ax.plot3D(X[:,0], Y[:,0], Z[:,0], 'r', linewidth=2)
            ax.plot3D([xVal, xVal], [-xMax,xMax], [zMin,zMin], 'r--', linewidth=2)

        if yCut is True:
            ax.plot3D(X[-1,:], Y[-1,:], Z[-1,:], 'r', linewidth=2)
            ax.plot3D([-xMax,xMax], [yVal, yVal], [zMin,zMin], 'r--', linewidth=2)

        ax.plot3D([xVal], [yVal], [f(xVal, yVal)], marker = 'o', color = 'r', zorder=5)

        ax.set_xlim(xMin,xMax)
        ax.set_ylim(xMin,xMax)
        ax.set_zlim(zMin,zMax)
        ax.view_init(elev, azim)
        ax.invert_xaxis()
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1))
        ax.set_zlabel(r'$z$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        plt.tight_layout()

    out = widgets.interactive_output(multivar_plot, {'curve':curve_list,
                                                       'xMin': xMin_slider,
                                                       'xMax': xMax_slider,
                                                       'xVal': xVal_slider,
                                                       'yVal': yVal_slider,
                                                       'elev': elev_slider,
                                                       'azim': azim_slider,
                                                       'xCut': xCut_check,  
                                                       'yCut': yCut_check})
        
    output = widgets.VBox([out,
                           curve_list,
                  widgets.HBox([xMin_slider,
                                xMax_slider,
                                xVal_slider,
                                yVal_slider]),
                  widgets.HBox([elev_slider, 
                                azim_slider,
                                xCut_check,
                                yCut_check])
                          ])
    display(output)
    
def matrix_widget(xMax_init = 10, yMax_init = 10, mode_init = 'A', A_base = [[1,0],[0,1]],
                  B_base = [[1,0],[0,1]], D_base = [[3,4,5],[-1,2,-1]], fixScale_init = False):
    
    # Declare widgets for interactive input
    boxLayout = {'width': '175px'}
    
    # style = {'description_width': 'initial'}
    xMax_slider = widgets.IntSlider(min=0.1,
                                 max=1000,
                                 step=1,
                                 description=r'Maximum $x$:',
                                 value = xMax_init,
                                 continuous_update =True)
    yMax_slider = widgets.IntSlider(min=0.1,
                                 max=1000,
                                 step=1,
                                 description=r'Maximum $y$:',
                                 value = yMax_init,
                                 continuous_update =True)
    mode_list = widgets.Dropdown(options=['A','B', 'AB', 'BA'],
                                value = mode_init,
                                description=r'Transformation$:',
                                disabled=False)
    
    a1_box = widgets.FloatText(description= r'$a_{1,1}$:', value = A_base[0][0], 
                               layout=boxLayout)
    a2_box = widgets.FloatText(description= r'$a_{1,2}$:', value = A_base[0][1], 
                               layout=boxLayout)
    a3_box = widgets.FloatText(description= r'$a_{2,1}$:', value = A_base[1][0], 
                               layout=boxLayout)
    a4_box = widgets.FloatText(description= r'$a_{2,2}$:', value = A_base[1][1], 
                               layout=boxLayout)

    b1_box = widgets.FloatText(description= r'$b_{1,1}$:', value = B_base[0][0], 
                               layout=boxLayout)
    b2_box = widgets.FloatText(description= r'$b_{1,2}$:', value = B_base[0][1], 
                               layout=boxLayout)
    b3_box = widgets.FloatText(description= r'$b_{2,1}$:', value = B_base[1][0], 
                               layout=boxLayout)
    b4_box = widgets.FloatText(description= r'$b_{2,2}$:', value = B_base[1][1], 
                               layout=boxLayout)
    
    d1_box = widgets.FloatText(description= r'$d_{1,1}$:', value = D_base[0][0], 
                               layout=boxLayout)
    d2_box = widgets.FloatText(description= r'$d_{1,2}$:', value = D_base[0][1], 
                               layout=boxLayout)
    d3_box = widgets.FloatText(description= r'$d_{1,3}$:', value = D_base[0][2], 
                               layout=boxLayout)
    d4_box = widgets.FloatText(description= r'$d_{2,1}$:', value = D_base[1][0], 
                               layout=boxLayout)
    d5_box = widgets.FloatText(description= r'$d_{2,2}$:', value = D_base[1][1], 
                               layout=boxLayout)
    d6_box = widgets.FloatText(description= r'$d_{2,3}$:', value = D_base[1][2], 
                               layout=boxLayout)
    
    fixScale_check = widgets.Checkbox(value = fixScale_init,
                                   description='Fix 2nd axis',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    widgets.jslink((xMax_slider,'value'),(yMax_slider,'value'))

    def matrix_plot(xMax, yMax, mode, a1, a2 ,a3 ,a4 , b1, b2, b3, b4, 
                    d1, d2, d3, d4, d5, d6, fixScale):
        
        # Package inputs into lists
        A_base = [[a1,a2],
                  [a3,a4]]
        B_base = [[b1,b2],
                  [b3,b4]]
        D_base = [[d1,d2,d3],
                  [d4,d5,d6]]

        # Convert lists to Numpy matrices
        A = np.asmatrix(A_base)
        B = np.asmatrix(B_base)
        D = np.asmatrix(D_base)

        # Setup tranformation matrix T, get determinant of T
        if mode == 'A':
            T = A
        elif mode == 'B':
            T = B
        elif mode == 'AB':
            T = A*B
        elif mode == 'BA':    
            T= B*A
        Det = np.linalg.det(T)
        
        # Apply matrix to get transformed data
        R = T*D
        
        # Convert back to array for plotting
        Dplot = np.asarray(D_base)
        Rplot = np.asarray(R)

        # Find scale of x/y axes for both plots. Fix if required
        if fixScale is True or Det == 0:
            xMaxTrans = xMax
            yMaxTrans = yMax
            xGridMax = xMax
            yGridMax = yMax
        else:
            xMaxTrans = (abs(Det)**0.5)*xMax
            yMaxTrans = (abs(Det)**0.5)*xMax
            xGridMax = max(xMax,xMaxTrans)
            yGridMax = max(yMax,yMaxTrans)

        # Generate grids 
        gridStep = max(xGridMax,yGridMax)/10
        NxGrids = int(xGridMax/gridStep)
        xGrid = []
        xGridTransformed = []
        currX = gridStep
        for i in range(NxGrids):
            for sig in [-1,1]:
                line = [[sig*currX,sig*currX],
                        [-yGridMax-gridStep/2,yGridMax+gridStep/2]]
                lineTrans = T*np.asmatrix(line)

                xGrid.append(line)
                xGridTransformed.append(np.asarray(lineTrans))

            currX += gridStep

        NyGrids = int(yGridMax/gridStep)
        yGrid = []
        yGridTransformed = []
        currY = gridStep
        for i in range(NyGrids):
            for sig in [-1,1]:
                line = [[-xGridMax-gridStep/2,xGridMax+gridStep/2],
                        [sig*currY,sig*currY]]
                lineTrans = T*np.asmatrix(line)

                yGrid.append(line)
                yGridTransformed.append(np.asarray(lineTrans))

            currY += gridStep

        # Transform axis for plot
        axisTransformed = []
        axisStyle = ['g--','b--']
        axisLabel = ['Original $x$ axis', 'Original $y$ axis']
        xAxis = [[-xGridMax-gridStep/2,xGridMax+gridStep/2],[0,0]]
        yAxis = [[0,0], [-yGridMax-gridStep/2,yGridMax+gridStep/2]]
        xAxisTrans = T*np.asmatrix(xAxis)
        yAxisTrans = T*np.asmatrix(yAxis)
        axisTransformed.append(np.asarray(xAxisTrans))
        axisTransformed.append(np.asarray(yAxisTrans))

        # Create figure
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,10))

        # plot gridlines
        for line in xGrid:
            ax[0].plot(line[0],line[1],'k',alpha = 0.2)

        for line in yGrid:
            ax[0].plot(line[0],line[1],'k',alpha = 0.2)

        # plot data
        ax[0].scatter(Dplot[0,:],Dplot[1,:], s=mrkrSize, c='r',alpha = 0.6,
          label = r'Initial data $D$' )
        ax[0].plot(Dplot[0,:],Dplot[1,:],'r',alpha = 0.6)
        ax[0].fill(Dplot[0,:],Dplot[1,:],'r',alpha = 0.2)

        # Add legend and format axes to look nice
        ax[0].axis('equal')
        ax[0].legend(loc='upper left', frameon=False,prop={'size':20})
        ax[0].autoscale(enable=True, axis='both', tight=True)
        ax[0].set_ylim(top = yMax, bottom = -yMax)
        ax[0].set_xlim(right = xMax, left = -xMax)
        ax[0].spines['bottom'].set_position('zero')
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['left'].set_position('zero')
        ax[0].spines['right'].set_visible(False)
        ax[0].set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax[0].set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax[0].plot(1, 0, ">k", transform=ax[0].get_yaxis_transform(), clip_on=False)
        ax[0].plot(0, 1, "^k", transform=ax[0].get_xaxis_transform(), clip_on=False)
        ax[0].tick_params(labelsize=20)

        # plot transformed gridlines
        for line in xGridTransformed:
            ax[1].plot(line[0,:],line[1,:],'k',alpha = 0.2)

        for line in yGridTransformed:
            ax[1].plot(line[0,:],line[1,:],'k',alpha = 0.2)

        for line in axisTransformed:
            ax[1].plot(line[0,:],line[1,:],axisStyle.pop(0),alpha = 0.6, 
                       linewidth=2, label = axisLabel.pop(0))

        # Plot transformed data
        ax[1].scatter(Rplot[0,:],Rplot[1,:], s=mrkrSize, c='r',alpha = 0.6,
          label = r'Transformed data $'+mode+'D$' )
        ax[1].plot(Rplot[0,:],Rplot[1,:],'r',alpha = 0.6)
        ax[1].fill(Rplot[0,:],Rplot[1,:],'r',alpha = 0.2)

        ax[1].annotate(r'$|'+mode + '| = {:.4f}$'.format(Det),[0.75,0], 
                       xytext = [0.55,0.05], xycoords ='axes fraction', 
                       fontsize = 25, clip_on = True)

        # Add legend and format axes to look nice
        ax[1].axis('equal')
        ax[1].legend(loc='upper left', frameon=False,prop={'size':20})
        ax[1].autoscale(enable=True, axis='both', tight=True)
        ax[1].set_ylim(top = yMaxTrans, bottom = -yMaxTrans)
        ax[1].set_xlim(right = xMaxTrans, left = -xMaxTrans)
        ax[1].spines['bottom'].set_position('zero')
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['left'].set_position('zero')
        ax[1].spines['right'].set_visible(False)
        ax[1].set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax[1].set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax[1].plot(1, 0, ">k", transform=ax[1].get_yaxis_transform(), clip_on=False)
        ax[1].plot(0, 1, "^k", transform=ax[1].get_xaxis_transform(), clip_on=False)
        ax[1].tick_params(labelsize=20)

        fig.subplots_adjust(hspace=0.4, wspace=0.6)
        plt.tight_layout()
    
    out = widgets.interactive_output(matrix_plot, {'xMax': xMax_slider,
                                                   'yMax': yMax_slider,
                                                   'mode': mode_list, 
                                                   'a1': a1_box,
                                                   'a2': a2_box,
                                                   'a3': a3_box,
                                                   'a4': a4_box,
                                                   'b1': b1_box,
                                                   'b2': b2_box,
                                                   'b3': b3_box,
                                                   'b4': b4_box,
                                                   'd1': d1_box,
                                                   'd2': d2_box,
                                                   'd3': d3_box,
                                                   'd4': d4_box,
                                                   'd5': d5_box,
                                                   'd6': d6_box,                                                       
                                                   'fixScale': fixScale_check})
    
    output = widgets.VBox([out,
                  widgets.HBox([xMax_slider,
                                yMax_slider,
                                fixScale_check]),
                  widgets.HBox([
                        widgets.HBox([widgets.VBox([a1_box,a3_box]),
                                      widgets.VBox([a2_box,a4_box])]),
                        widgets.HBox([widgets.VBox([b1_box,b3_box]),
                                      widgets.VBox([b2_box,b4_box])])
                                ]),
                        widgets.HBox([widgets.VBox([d1_box,d4_box]),
                                      widgets.VBox([d2_box,d5_box]),
                                      widgets.VBox([d3_box,d6_box]),
                                      mode_list])
                          ])
    
    display(output)
    
def taylor_widget(xMin_init = -6, xMax_init = 6, yMin_init = -1.5, yMax_init = 1.5, 
                  func_init = 'Sine', numTerms_init = 6):

    # Declare widgets for interactive input
    xMin_slider = widgets.FloatSlider(min=-1000,
                                 max=1000,
                                 description=r'Min $x$:',
                                 value = xMin_init,
                                 continuous_update =False)
    xMax_slider = widgets.FloatSlider(min=-1000,
                                 max=1000,
                                 description=r'Max $x$:',
                                 value = xMax_init,
                                 continuous_update =False)
    yMin_slider = widgets.FloatSlider(min=-1000,
                                 max=1000,
                                 description=r'Min $y$:',
                                 value = yMin_init,
                                 continuous_update =False)
    yMax_slider = widgets.FloatSlider(min=-1000,
                                 max=1000,
                                 description=r'Max $y$:',
                                 value = yMax_init,
                                 continuous_update =False)
    func_list = widgets.Dropdown(options=['Exponential', 'Logarithm', 'Sine'],
                                value = func_init,
                                description=r'Pick function:',
                                disabled=False)
    numTerms_slider = widgets.IntSlider(min=1,
                                 max=24,
                                 description=r'N° of terms:',
                                 value = numTerms_init,
                                 continuous_update =False)
    
    # Link widgets as required
    widgets.jslink((xMin_slider,'value'),(xMax_slider,'min'))
    widgets.jslink((xMax_slider,'value'),(xMin_slider,'max'))
    widgets.jslink((yMin_slider,'value'),(yMax_slider,'min'))
    widgets.jslink((yMax_slider,'value'),(yMin_slider,'max'))

    def taylor_plot(xMin, xMax, yMin, yMax, func, numTerms):

        # Prepare dicts of log functions and color codes
        labels = {'Exponential':'e^x',
                  'Logarithm':'ln(x)',
                  'Sine':'sin(x)'}
        
        funcDict = {'Exponential': lambda x: np.exp(x),
                   'Logarithm': lambda x: np.log(x.clip(1e-6)),
                   'Sine': lambda x: np.sin(x)}

        taylorDict = {'Exponential': lambda x,n: (x**n)/np.math.factorial(n),
                   'Logarithm': lambda x,n: ((-1)**(n+1))*((x-1)**n)/n if n > 0 else 0,
                   'Sine': lambda x,n: ((-1)**n)*(x**(2*n+1))/np.math.factorial(2*n+1)}

        colors = ['b','r','g','m','c','y',
                  'b--','r--','g--','m--','c--','y--',
                  'b:','r:','g:','m:','c:','y:',
                  'b-.','r-.','g-.','m-.','c-.','y-.']

        # Create an 'x' vector
        x = np.arange(xMin,xMax,(xMax-xMin)/500)

        # Create figure, calculate and plot Taylor function for each term added
        fig, ax = plt.subplots(figsize=(20,10))

        yTaylor = np.zeros(len(x))
        for n in range(numTerms):         
            taylorTerm = np.asarray(
                            np.clip(taylorDict[func](x,n), -1000, 1000), 
                            dtype = np.double)
            yTaylor += taylorTerm
            ax.plot(x, yTaylor,colors[n], linewidth=2, alpha=0.6, 
                    label='Term {:d}'.format(n))

        y = funcDict[func](x)
        ax.plot(x, y,'k', linewidth=2, label=r'$\quad y = {:s}$'.format(labels[func]))

        # Add legend and format axes to look nice
        ax.legend(loc='lower center', frameon=False,prop={'size':20},ncol=6,
                   bbox_to_anchor=(0.5, -0.25))

        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = yMin)
        ax.set_xlim(right = xMax, left = xMin)
        
        if yMin > 0:
            ax.spines['bottom'].set_position(('data',yMin))
        elif yMax < 0:
            ax.spines['bottom'].set_position(('data',yMax))
        else:       
            ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        if xMin > 0:
            ax.spines['left'].set_position(('data',xMin))
        elif xMax < 0:
            ax.spines['left'].set_position(('data',xMax))
        else:
            ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
    
    out = widgets.interactive_output(taylor_plot, {'xMin': xMin_slider,
                                                'xMax': xMax_slider,
                                                'yMin': yMin_slider,
                                                'yMax': yMax_slider,
                                                'func': func_list, 
                                                'numTerms': numTerms_slider})

    output = widgets.VBox([out,
                  widgets.HBox([xMin_slider,
                               xMax_slider,
                               yMin_slider,
                               yMax_slider]),
                  widgets.HBox([func_list, 
                                numTerms_slider])])
    display(output)
    
#------------------------------------------------------------------------------
# Economic application widgets
    
def utility_widget(curve_init = 'CD', a_init = 0.5, xMin_init = 0, xMax_init = 12, xPrice_init = 16,
                   yPrice_init = 10, budget_init = 100,
                      elev_init = 30, azim_init = 290, zCut_init = False,
                  constraintCut_init = False):
    
    # Declare widgets for interactive input
    curve_list = widgets.Dropdown(options=['CD'],
                                value = curve_init,
                                description='function:',
                                disabled=False)
    a_slider = widgets.FloatSlider(min=0,
                                 max=1,
                                 step=0.1,
                                 description=r'$\alpha$:',
                                 value = a_init,
                                 continuous_update =False)
    xMin_slider = widgets.IntSlider(min=-100,
                                 max=0,
                                 step=1,
                                 description=r'Min. $x$,$y$:',
                                 value = xMin_init,
                                 continuous_update =False)    
    xMax_slider = widgets.IntSlider(min=0,
                                 max=100,
                                 step=1,
                                 description=r'Max. $x$,$y$:',
                                 value = xMax_init,
                                 continuous_update =False)
    xPrice_slider = widgets.FloatSlider(min=0,
                                 max=100,
                                 step=1,
                                 description=r'Price of $x$:',
                                 value = xPrice_init,
                                 continuous_update =False)
    yPrice_slider = widgets.FloatSlider(min=0,
                                 max=100,
                                 step=1,
                                 description=r'Price of $y$:',
                                 value = yPrice_init,
                                 continuous_update =False)
    budget_slider = widgets.FloatSlider(min=0,
                                 max=500,
                                 step=1,
                                 description=r'Budget $M$:',
                                 value = budget_init,
                                 continuous_update =False)
    elev_slider = widgets.FloatSlider(min=-90,
                                 max=90,
                                 description= 'Elevation:',
                                 value = elev_init,
                                 continuous_update =False)
    azim_slider = widgets.FloatSlider(min=0,
                                 max=360,
                                 description= 'Azimuth:',
                                 value = azim_init,
                                 continuous_update =False)
    zCut_check = widgets.Checkbox(value = zCut_init,
                                   description='Fix $U$',
                                   disabled=False,
                                   indent=True)
    constraintCut_check = widgets.Checkbox(value = constraintCut_init,
                                   description='Fix $M$',
                                   disabled=False,
                                   indent=True)  
    
    # Link widgets as required
    
    def optim_plot(curve, a, xMin, xMax, xPrice, yPrice, budget, elev, azim , zCut, constraintCut):

        fundict = {'CD': lambda x, y: (x**a)*(y**(1-a)),
                  }
        fundictAlt = {'CD': lambda x, z: (x**(-a/(1-a)))*(z**(1/(1-a))),
                  }

        # if zCut is True and constraintCut is False:
            # zCut = False
        
        f = fundict[curve]
        fAlt = fundictAlt[curve]

        res = 100
        xFull = np.linspace(1e-3, xMax, res)
        yFull = np.linspace(1e-3, xMax, res)
        XFull, YFull = np.meshgrid(xFull, yFull)
        ZFull = f(XFull, YFull)
        zMax = max(ZFull.flatten())

        if constraintCut is True:
            xConstraint = np.linspace(0, budget/xPrice, res)
            yConstraint = budget/yPrice - xPrice/yPrice*xConstraint
            zConstraint = f(xConstraint, yConstraint)
            optInd = np.argmax(zConstraint)
            xOpt = xConstraint[optInd]
            yOpt = yConstraint[optInd]
            zVal = zConstraint[optInd]
            
        if zCut is True:
            resZ = int(np.floor((zVal/zMax)*res))
            z = np.linspace(0, zVal, resZ)
            zCover = (zVal/zMax)
            viridis = cm.get_cmap('viridis', 512)
            myCmap = ListedColormap(
                       viridis(np.linspace(0, zCover, 256))
                       )
        else:
            z = np.linspace(0, zMax, res)
            myCmap = 'viridis'

        X, Z = np.meshgrid(xFull, z)
        Y = fAlt(X, Z)
        
        Y = np.clip(Y,0,xMax)
        
        # Slice  budget constraint 
        if constraintCut is True:
            mask = Y < budget/yPrice - xPrice/yPrice*X
            Y[mask] = budget/yPrice - xPrice/yPrice*X[mask]
            
        Z = f(X,Y)
        
        # Create figure
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        fig = plt.figure(figsize=(20,10))

        # Plot 3D utility function
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(X, Y, Z, cmap=myCmap)
        
        if constraintCut is True:
            ax.plot3D(xConstraint, yConstraint, zConstraint, 'b', linewidth=2)
            ax.plot3D(xConstraint, yConstraint, 0*zConstraint, 'b--', linewidth=2, zorder = -5)
        
        if zCut is True:
            xIC = X[-1,:]
            yIC = Y[-1,:]
            zIC = Z[-1,:]
            maskIC = yIC < xMax
            ax.plot3D(xIC[maskIC], yIC[maskIC], zIC[maskIC], 'r', linewidth=2)
            ax.plot3D(xIC[maskIC], yIC[maskIC], 0*zIC[maskIC], 'r--', linewidth=2, zorder = -5)
        
        if constraintCut is True and zCut is True:
            ax.plot3D([xOpt],[yOpt], [zVal], marker = 'o', color = 'r', zorder = 5)
            ax.plot3D([0,xOpt],[yOpt,yOpt],[0,0],'r--',linewidth=1)
            ax.plot3D([xOpt,xOpt],[0,yOpt],[0,0],'r--',linewidth=1)
            ax.plot3D([xOpt,xOpt],[yOpt,yOpt],[0,zVal],'r--',linewidth=1)


        ax.set_xlim(xMin,xMax)
        ax.set_ylim(xMin,xMax)
        ax.set_zlim(0,1.5*zMax)
        ax.view_init(elev, azim)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1))
        ax.set_zlabel(r'$U(x,y)$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        
        # Plot Indifference curve map
        ax = fig.add_subplot(1, 2, 2)
        numIC = 6
        zStepVec = np.linspace(zMax/numIC, zMax-1/numIC, numIC)
        for zStep in zStepVec:
            ax.plot(xFull, fAlt(xFull, zStep), 'k--', alpha=0.3)
        
        if constraintCut is True:
            ax.plot(xConstraint, yConstraint, 'b', linewidth=2, alpha=0.6,
                    label=r'Budget constraint')
            
        if zCut is True:
            ax.plot(xFull, fAlt(xFull, zVal), 'r', linewidth=2, alpha=0.6,
                    label=r'Highest reachable IC')
            # Add markers for the optimal point points, with dotted lines
            ax.scatter(xConstraint[optInd], yConstraint[optInd], s=mrkrSize, c='r', alpha=0.6,
                        label='Optimal point')
            ax.plot([0,xOpt],[yOpt,yOpt],'r--',linewidth=1)
            ax.plot([xOpt,xOpt],[0,yOpt],'r--',linewidth=1)

        if constraintCut is True:
            ax.legend(loc='upper right', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = xMax, bottom = 0)
        ax.set_xlim(right = xMax, left = 0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$x$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$y$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        ax.tick_params(labelsize=20)
        
        plt.tight_layout()

    out = widgets.interactive_output(optim_plot, {'curve': curve_list,
                                                   'a': a_slider,
                                                   'xMin': xMin_slider,
                                                   'xMax': xMax_slider,
                                                   'xPrice': xPrice_slider,
                                                   'yPrice': yPrice_slider,
                                                   'budget':budget_slider,
                                                   'elev': elev_slider,
                                                   'azim': azim_slider,
                                                   'zCut': zCut_check,
                                                   'constraintCut': constraintCut_check})
        
    output = widgets.VBox([out,
                  widgets.HBox([curve_list,
                                a_slider, 
                                budget_slider]),
                  widgets.HBox([xMin_slider,
                                xMax_slider,
                                xPrice_slider,
                                yPrice_slider]),
                  widgets.HBox([elev_slider, 
                                azim_slider,
                                constraintCut_check,
                                zCut_check])
                          ])
    display(output)
    
def elasticities_widget(Qmax_init = 15, Pmax_init = 30, Qval_init = 5, a_d_init = 2,
    b_d_init = 25, a_s_init = 2, b_s_init = -3, Dflag_init = True, Sflag_init = False):
        
    # Declare widgets for interactive input
    Qmax_slider = widgets.IntSlider(min=5,
                                 max=1000,
                                 step=1,
                                 description=r'Maximum $Q$:',
                                 value = Qmax_init,
                                 continuous_update =False)
    Pmax_slider = widgets.IntSlider(min=5,
                                 max=1000,
                                 step=1,
                                 description=r'Maximum $P$:',
                                 value = Pmax_init,
                                 continuous_update =False)
    Qval_slider = widgets.FloatSlider(min=0.001,
                                 max=15,
                                 description='Pick a Quantity:',
                                 value = Qval_init,
                                 continuous_update =False)
    a_d_slider = widgets.FloatSlider(min=0.001,
                                 max=100,
                                 description= r'$a_d$:',
                                 value = a_d_init,
                                 continuous_update =False)
    b_d_slider = widgets.FloatSlider(min=0,
                                 max=1000,
                                 description= r'$b_d$:',
                                 value=b_d_init,
                                 continuous_update =False)
    a_s_slider = widgets.FloatSlider(min=0.001,
                                 max=100,
                                 description= r'$a_s$:',
                                 value = a_s_init,
                                 continuous_update =False)
    b_s_slider = widgets.FloatSlider(min=-1000,
                                 max=1000,
                                 description= r'$b_s$:',
                                 value = b_s_init,
                                 continuous_update =False)
    Dflag_check = widgets.Checkbox(value = Dflag_init,
                                   description='Include Demand',
                                   disabled=False,
                                   indent=True) 
    Sflag_check = widgets.Checkbox(value = Sflag_init,
                                   description='Include Supply',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    widgets.jslink((Qmax_slider,'value'),(Qval_slider,'max'))
    widgets.jslink((b_d_slider,'value'),(b_s_slider,'max'))

    def elasticity_plot(Qmax, Pmax, Qval, a_d, b_d, a_s, b_s, Dflag, Sflag):

        # create a quantity vector, calculate supply/demand vectors
        Q = np.arange(0,Qmax)
        P_s = a_s*Q + b_s
        P_d = -a_d*Q + b_d

        # Calculate equilibrium quantity/price
        Qeq = (b_d-b_s)/(a_s + a_d)
        Peq = a_s*Qeq + b_s

        # Calculate prices for selected value
        Pval_s = a_s*Qval + b_s
        Pval_d = -a_d*Qval + b_d

        # Create figure, plot supply/demand
        fig, ax = plt.subplots(figsize=(20,10))
        if Sflag is True:
            ax.plot(Q, P_s,'r', linewidth=2, alpha=0.6,
                    label=r'Inverse Supply $\quad P_s = a_s Q + b_s$')
        if Dflag is True:
            ax.plot(Q, P_d,'b', linewidth=2, alpha=0.6,
                    label=r'Inverse Demand $\quad P_d = -a_d Q + b_d$')

        # Add markers for the price/quantity points, with dotted lines
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        if Sflag is True and Dflag is True:
            ax.scatter(Qeq, Peq, s=mrkrSize, c='k', label='Equilibrium')       
            ax.scatter(Qval, Pval_s, s=mrkrSize, c='k', alpha=0.6, 
                       label='Selection')
            ax.scatter(Qval, Pval_d, s=mrkrSize, c='k', alpha=0.6)

            ax.plot([Qeq,Qeq],[0,Peq],'k--',linewidth=1)
            ax.plot([0,Qeq],[Peq,Peq],'k--',linewidth=1)
            ax.plot([0,Qval],[Pval_d,Pval_d],'k--',linewidth=1)
            ax.plot([Qval,Qval],[0,max(Pval_s,Pval_d)],'k--',linewidth=1)
            ax.plot([0,Qval],[Pval_s,Pval_s],'k--',linewidth=1)
            ax.annotate(r'$Q^*$',[Qeq,0], xytext = [Qeq+0.15,0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)
            ax.annotate(r'$P^*$',[0,Peq], xytext = [0.15,Peq+0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)
            ax.annotate(r'$Q$',[Qval,0], xytext = [Qval+0.15,0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)
            ax.annotate(r'$P_s$',[0,Pval_s], xytext = [0.15,Pval_s+0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)
            ax.annotate(r'$P_d$',[0,Pval_d], xytext = [0.15,Pval_d+0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)

        elif Sflag is True and Dflag is False:
            ax.scatter(Qval, Pval_s, s=mrkrSize, c='k', alpha=0.6, 
                       label='Selection')
            ax.plot([0,Qval],[Pval_s,Pval_s],'k--',linewidth=1)
            ax.plot([Qval,Qval],[0,Pval_s],'k--',linewidth=1)    
            ax.annotate(r'$Q$',[Qval,0], xytext = [Qval+0.15,0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)
            ax.annotate(r'$P_s$',[0,Pval_s], xytext = [0.15,Pval_s+0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)

        elif Sflag is False and Dflag is True:
            ax.scatter(Qval, Pval_d, s=mrkrSize, c='k', alpha=0.6, 
                       label='Selection')
            ax.plot([0,Qval],[Pval_d,Pval_d],'k--',linewidth=1)
            ax.plot([Qval,Qval],[0,Pval_d],'k--',linewidth=1)
            ax.annotate(r'$Q$',[Qval,0], xytext = [Qval+0.15,0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)
            ax.annotate(r'$P_d$',[0,Pval_d], xytext = [0.15,Pval_d+0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)

        # Add elasticity annotations
        if Sflag is True:
            EsStr = r'$E_p^S = {:.2f}$'.format(Pval_s/(Qval*a_s))
            ax.annotate(EsStr,[Qval,Pval_s],
                        xytext = [Qval+0.25,Pval_s-1], 
                        xycoords ='data',
                        fontsize = 25,
                        clip_on = True)
        if Dflag is True:
            EdStr = r'$E_p^D = {:.2f}$'.format(-Pval_d/(Qval*a_d))
            ax.annotate(EdStr,[Qval,Pval_d],
                        xytext = [Qval+0.25,Pval_d],
                        xycoords ='data',
                        fontsize = 25,
                        clip_on = True)

        # Add legend and format axes to look nice
        if Sflag is True or Dflag is True:
            ax.legend(loc='upper center', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = Pmax, bottom = 0)
        ax.set_xlim(right = Qmax, left = 0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$Q$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$P$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
    
    out = widgets.interactive_output(elasticity_plot, {'Qmax': Qmax_slider,
                                                       'Pmax': Pmax_slider,
                                                       'Qval': Qval_slider, 
                                                       'a_d': a_d_slider,
                                                       'b_d': b_d_slider,
                                                       'a_s': a_s_slider, 
                                                       'b_s': b_s_slider, 
                                                       'Dflag': Dflag_check,
                                                       'Sflag': Sflag_check})

    output = widgets.VBox([out,
                  widgets.HBox([Qval_slider,
                               Qmax_slider,
                               Pmax_slider]),
                  widgets.HBox([Dflag_check, 
                                a_d_slider, 
                                b_d_slider]),
                  widgets.HBox([Sflag_check, 
                                a_s_slider, 
                                b_s_slider])])
    display(output)

def slope_from_elasticities_widget(Qmax_init = 15, Pmax_init = 10, Qeq_init = 7.5, Peq_init = 5,
    E_s_init = 2.5, E_d_init = -1.5, Dflag_init = True, Sflag_init = True):
        
    # Declare widgets for interactive input
    Qmax_slider = widgets.IntSlider(min=5,
                                 max=1000,
                                 step=1,
                                 description=r'Maximum $Q$:',
                                 value = Qmax_init,
                                 continuous_update =False)
    Pmax_slider = widgets.IntSlider(min=5,
                                 max=1000,
                                 step=1,
                                 description=r'Maximum $P$:',
                                 value = Pmax_init,
                                 continuous_update =False)
    Qeq_slider = widgets.FloatSlider(min=0.001,
                                 max=15,
                                 description='Pick a Quantity:',
                                 value = Qeq_init,
                                 continuous_update =False)
    Peq_slider = widgets.FloatSlider(min=0.001,
                                 max=200,
                                 description= 'Pick a Price:',
                                 value = Peq_init,
                                 continuous_update =False)
    E_d_slider = widgets.FloatSlider(min=-15,
                                 max=-0.001,
                                 description= 'Price elasticity of demand:',
                                 value=E_d_init,
                                 continuous_update =False)
    E_s_slider = widgets.FloatSlider(min=0.001,
                                 max=15,
                                 description= 'Price elasticity of supply:',
                                 value = E_s_init,
                                 continuous_update =False)
    Dflag_check = widgets.Checkbox(value = Dflag_init,
                                   description='Include Demand',
                                   disabled=False,
                                   indent=True) 
    Sflag_check = widgets.Checkbox(value = Sflag_init,
                                   description='Include Supply',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    widgets.jslink((Qmax_slider,'value'),(Qeq_slider,'max'))
    widgets.jslink((Pmax_slider,'value'),(Peq_slider,'max'))

    def slope_plot(Qmax, Pmax, Qeq, Peq , E_d, E_s, Dflag, Sflag):

        # Calculate demand supply curve parameters from input    
        a_s = E_s*Peq/Qeq
        b_s = Peq-a_s*Qeq
        a_d = E_d*Peq/Qeq
        b_d = Peq-a_d*Qeq

        # create a quantity vector, calculate supply/demand vectors
        Q = np.asarray([Qeq-0.05*Qmax,Qeq+0.05*Qmax])
        P_s = a_s*Q + b_s
        P_d = a_d*Q + b_d

        # Create figure, plot supply/demand approximations with arrows
        fig, ax = plt.subplots(figsize=(20,10))
        if Sflag is True:
            ax.plot(Q, P_s,'r', linewidth=2, alpha=0.6,
                    label= 'Local approximation of Supply curve')
            ax.annotate('', xy=(Q[0],P_s[0]), xytext=(Q[0]-0.01,P_s[0]-0.01*a_s),
                        arrowprops={'arrowstyle': '<-', 
                                    'lw': 2, 
                                    'color': 'r', 
                                    'alpha':0.6})
            ax.annotate('', xy=(Q[1],P_s[1]), xytext=(Q[1]+0.01,P_s[1]+0.01*a_s),
                        arrowprops={'arrowstyle': '<-', 
                                    'lw': 2, 
                                    'color': 'r', 
                                    'alpha':0.6})

        if Dflag is True:
            ax.plot(Q, P_d,'b', linewidth=2, alpha=0.6,
                    label= 'Local approximation of Demand curve')
            ax.annotate('', xy=(Q[0],P_d[0]), xytext=(Q[0]-0.01,P_d[0]-0.01*a_d),
                        arrowprops={'arrowstyle': '<-', 
                                    'lw': 2, 
                                    'color': 'b', 
                                    'alpha':0.6})
            ax.annotate('', xy=(Q[1],P_d[1]), xytext=(Q[1]+0.01,P_d[1]+0.01*a_d),
                        arrowprops={'arrowstyle': '<-', 
                                    'lw': 2, 
                                    'color': 'b', 
                                    'alpha':0.6})

        # Add markers for the price/quantity points, with dotted lines
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        if Sflag is True and Dflag is True:
            ax.scatter(Qeq, Peq, s=mrkrSize, c='k', label='Equilibrium')       

            ax.plot([Qeq,Qeq],[0,Peq],'k--',linewidth=1)
            ax.plot([0,Qeq],[Peq,Peq],'k--',linewidth=1)
            ax.annotate(r'$Q^*={:.2f}$'.format(Qeq),[Qeq,0], xytext = [Qeq+0.15,0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)
            ax.annotate(r'$P^*={:.2f}$'.format(Peq),[0,Peq], xytext = [0.15,Peq+0.25], 
                        xycoords ='data', fontsize = 25, clip_on = True)

    #    # Add elasticity annotations
        if Sflag is True:
            EsStr = r'$E_p^S = {:.2f}$'.format(E_s)
            ax.annotate(EsStr,[Q[1],P_s[1]],
                        xytext = [Q[1]+0.25,P_s[1]], 
                        xycoords ='data',
                        fontsize = 25,
                        clip_on = True)
        if Dflag is True:
            EdStr = r'$E_p^D = {:.2f}$'.format(E_d)
            ax.annotate(EdStr,[Q[1],P_d[1]],
                        xytext = [Q[1]+0.25,P_d[1]-0.5], 
                        xycoords ='data',
                        fontsize = 25,
                        clip_on = True)

        # Add legend and format axes to look nice
        if Sflag is True or Dflag is True:
            ax.legend(loc='upper center', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = Pmax, bottom = 0)
        ax.set_xlim(right = Qmax, left = 0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$Q$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$P$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
    
    out = widgets.interactive_output(slope_plot, {'Qmax': Qmax_slider,
                                                   'Pmax': Pmax_slider,
                                                   'Qeq': Qeq_slider, 
                                                   'Peq': Peq_slider,
                                                   'E_d': E_d_slider,
                                                   'E_s': E_s_slider, 
                                                   'Dflag': Dflag_check,
                                                   'Sflag': Sflag_check})

    output = widgets.VBox([out,
                  widgets.HBox([Qmax_slider,
                               Pmax_slider,
                               Qeq_slider,
                               Peq_slider]),
                  widgets.HBox([Dflag_check, 
                                E_d_slider, 
                                Sflag_check,
                                E_s_slider])])
    display(output) 
    
def tax_widget(Qmax_init = 15, Pmax_init = 30, Tval_init = 0, a_d_init = 2,
    b_d_init = 25, a_s_init = 2, b_s_init = -3, Rflag_init = False, Lflag_init = False):
        
    # Declare widgets for interactive input
    Qmax_slider = widgets.IntSlider(min=5,
                                 max=1000,
                                 step=1,
                                 description=r'Maximum $Q$:',
                                 value = Qmax_init,
                                 continuous_update =False)
    Pmax_slider = widgets.IntSlider(min=5,
                                 max=1000,
                                 step=1,
                                 description=r'Maximum $P$:',
                                 value = Pmax_init,
                                 continuous_update =False)
    Tval_slider = widgets.FloatSlider(min=0.001,
                                 max=15,
                                 description='Set tax level:',
                                 value = Tval_init,
                                 continuous_update =False)
    a_d_slider = widgets.FloatSlider(min=0.001,
                                 max=100,
                                 description= r'$a_d$:',
                                 value = a_d_init,
                                 continuous_update =False)
    b_d_slider = widgets.FloatSlider(min=0,
                                 max=1000,
                                 description= r'$b_d$:',
                                 value=b_d_init,
                                 continuous_update =False)
    a_s_slider = widgets.FloatSlider(min=0.001,
                                 max=100,
                                 description= r'$a_s$:',
                                 value = a_s_init,
                                 continuous_update =False)
    b_s_slider = widgets.FloatSlider(min=-1000,
                                 max=1000,
                                 description= r'$b_s$:',
                                 value = b_s_init,
                                 continuous_update =False)
    Rflag_check = widgets.Checkbox(value = Rflag_init,
                                   description='Show revenue',
                                   disabled=False,
                                   indent=True) 
    Lflag_check = widgets.Checkbox(value = Lflag_init,
                                   description='Show loss',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    widgets.jslink((Qmax_slider,'value'),(Tval_slider,'max'))
    widgets.jslink((b_d_slider,'value'),(b_s_slider,'max'))

    def tax_plot(Qmax, Pmax, Tval, a_d, b_d, a_s, b_s, Rflag, Lflag):

        # create a quantity vector, calculate supply/demand vectors
        Q = np.arange(0,Qmax)
        P_s = a_s*Q + b_s
        P_sT = a_s*Q + b_s + Tval
        P_d = -a_d*Q + b_d

        # Calculate equilibrium quantity/price without tax
        Qeq = (b_d-b_s)/(a_s + a_d)
        Peq = a_s*Qeq + b_s

        # Calculate quantity/prices for selected tax
        QT = (b_d-b_s-Tval)/(a_s + a_d) 
        Pval_s = a_s*QT + b_s
        Pval_d = -a_d*QT + b_d

        # Create figure, plot supply/demand curves
        fig, ax = plt.subplots(figsize=(20,10))
        ax.plot(Q, P_s,'r', linewidth=2, alpha=0.6,
                label=r'Inverse Supply, no tax $\quad P_s = a_s Q + b_s$')
        ax.plot(Q, P_sT,'r--', linewidth=2, alpha=0.6,
                label=r'Inverse Supply, with tax $\quad P_s = a_s Q + b_s + T$')
        ax.plot(Q, P_d,'b', linewidth=2, alpha=0.6,
                label=r'Inverse Demand $\quad P_d = -a_d Q + b_d$')

        # Add markers for the price/quantity points, with dotted lines
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        ax.scatter(Qeq, Peq, s=mrkrSize, c='k', 
                   label=r'Equilibrium, no tax ($Q^*$={:.2f},$P^*$={:.2f})'.format(Qeq,Peq))
        ax.scatter(QT, Pval_s, s=mrkrSize, c='k', alpha=0.6, 
                   label='Equilibrium, with tax')
        ax.scatter(QT, Pval_d, s=mrkrSize, c='k', alpha=0.6)
        
        # Plot areas if requested
        if Rflag is True:
            ax.fill([0,QT,QT,0],[Peq,Peq,Pval_d,Pval_d],'b',alpha = 0.2,
                    label = 'Tax revenue from consumers')
            ax.fill([0,QT,QT,0],[Peq,Peq,Pval_s,Pval_s],'r',alpha = 0.2,
                    label = 'Tax revenue from suppliers')
        if Lflag is True:
            ax.fill([QT,Qeq,QT],[Pval_d,Peq,Pval_s],'k',alpha = 0.2,
                    label = 'Deadweight loss')
        
        # Add dotted lines
        ax.plot([Qeq,Qeq],[0,Peq],'k--',linewidth=1)
        ax.plot([0,Qeq],[Peq,Peq],'k--',linewidth=1)
        ax.plot([0,QT],[Pval_d,Pval_d],'k--',linewidth=1)
        ax.plot([QT,QT],[0,max(Pval_s,Pval_d)],'k--',linewidth=1)
        ax.plot([0,QT],[Pval_s,Pval_s],'k--',linewidth=1)
        ax.annotate(r'$Q^*, no tax$',[Qeq,0], xytext = [Qeq+0.15,0.25], 
                    xycoords ='data', fontsize = 25, clip_on = True)
        ax.annotate(r'$Q^*, tax$',[QT,0], xytext = [QT+0.15,0.25], 
                    xycoords ='data', fontsize = 25, clip_on = True)
        ax.annotate(r'$P^*$',[0,Peq], xytext = [0.15,Peq+0.25], 
                    xycoords ='data', fontsize = 25, clip_on = True)
        ax.annotate(r'$T={:.2f}$'.format(Tval),[0,(Pval_s+Pval_d)/2], xytext = [1,(Pval_s+Pval_d)/2], 
                    xycoords ='data', fontsize = 25, clip_on = True)
        ax.annotate(r'$P_s$',[0,Pval_s], xytext = [0.15,Pval_s+0.25], 
                    xycoords ='data', fontsize = 25, clip_on = True)
        ax.annotate(r'$P_d$',[0,Pval_d], xytext = [0.15,Pval_d+0.25], 
                    xycoords ='data', fontsize = 25, clip_on = True)
        
        # Add legend and format axes to look nice
        ax.legend(loc='upper right', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = Pmax, bottom = 0)
        ax.set_xlim(right = Qmax, left = 0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$Q$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'$P$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
    
    out = widgets.interactive_output(tax_plot, {'Qmax': Qmax_slider,
                                               'Pmax': Pmax_slider,
                                               'Tval': Tval_slider, 
                                               'a_d': a_d_slider,
                                               'b_d': b_d_slider,
                                               'a_s': a_s_slider, 
                                               'b_s': b_s_slider,
                                               'Rflag': Rflag_check,
                                               'Lflag': Lflag_check})

    output = widgets.VBox([out,
                  widgets.HBox([Tval_slider,
                               Qmax_slider,
                               Pmax_slider]),
                  widgets.HBox([Rflag_check,
                                a_d_slider, 
                                b_d_slider]),
                  widgets.HBox([Lflag_check,
                                a_s_slider, 
                                b_s_slider])])
    display(output)
    
def uk_IO_widget(shockSector_init = ['Financial and insurance'], shockSign_init = 'Negative',
                 shockSize_init = 10, NumRounds_init = 5, Normalise_init = False, 
                 plotType_init = 'Shocks over time'):
    
    shockSector_list = widgets.SelectMultiple(options=['Agriculture',
                                        'Production',
                                        'Construction',
                                        'Distribution, transport, hotels and restaurants',
                                        'Information and communication',
                                        'Financial and insurance',
                                        'Real estate',
                                        'Professional and support activities',
                                        'Government, health & education',
                                        'Other services'],
                                value = shockSector_init,
                                disabled=False,
                                layout={'width': 'max-content'})
    shockSign_list = widgets.Dropdown(options=['Positive', 'Negative'],
                                value = shockSign_init,
                                description='Sign:',
                                disabled=False)
    plotType_list = widgets.Dropdown(options=['Shocks over time', 'Cumulative shocks'],
                                value = plotType_init,
                                description='Diagram:',
                                disabled=False)
    shockSize_slider = widgets.IntSlider(min=1,
                                         max=80,
                                         step=1,
                                         description= r'% size :',
                                         value = shockSize_init,
                                         disabled=False,
                                         continuous_update=False)
    NumRounds_slider = widgets.IntSlider(min=1,
                                         max=20,
                                         step=1,
                                         description= r'N° of rounds:',
                                         value = NumRounds_init,
                                         disabled=False,
                                         continuous_update=False)
    Normalise_check = widgets.Checkbox(value = Normalise_init,
                                   description='Display shocks in %',
                                   disabled=False,
                                   indent=True)
    
    def uk_IO_plot(shockSector, shockSign, shockSize, NumRounds, Normalise, 
                   plotType):

        sectorDict = {'Agriculture' : 0,
                    'Production' : 1,
                    'Construction' : 2,
                    'Distribution, transport, hotels and restaurants' : 3,
                    'Information and communication' : 4,
                    'Financial and insurance' : 5,
                    'Real estate' : 6,
                    'Professional and support activities' : 7,
                    'Government, health & education' : 8,
                    'Other services' : 9}

        colors = {'Agriculture' :'b',
                  'Production' : 'k',
                  'Construction' : 'r',
                  'Distribution, transport, hotels and restaurants' : 'g',
                  'Information and communication' : 'c',
                  'Financial and insurance' : 'b--',
                  'Real estate' : 'k--',
                  'Professional and support activities' : 'r--',
                  'Government, health & education' : 'g--',
                  'Other services' : 'c--'}

        signs = {'Positive': 1,
                 'Negative': -1}

        # Hardcoded UK Input/Output data to avoid data dependencies
        IO = [
            [6246,  17458,     0,   618,     0,     2,    65,   0,      80,    63],
            [8409, 367066, 57463, 99070, 16041,  8583,  5463, 26135, 75138, 14615],
            [ 353,   7816, 96507,  5117,   177,  4787, 18953,  3527,  7223,   349],
            [1389,  23737,  1339, 76591,  5697, 17617,   684, 18503, 15250,  2254],
            [ 465,   8272,  2641, 11955, 34882, 20785,  1893, 11425, 11123,  4190],
            [1395,  18151,  4514, 13071,  2866, 39145, 27991,  9098, 10320,  1485],
            [   0,   1875,   395, 17849,  3583,  4050,  1914,  5850,  8587,  1489],
            [1837,  41941, 16618, 55125, 24720, 46022,  8383, 93386, 38383, 14367],
            [  27,   5325,  1189,  6479,   678,  2539,  3124,  5885, 32118,   280],
            [  71,    310,     1,   196,  2385,  1617,    29,   773,  5277,  7897]]

        X = np.asmatrix([[55226],
                         [1599529],
                         [324055],
                         [389993],
                         [247981],
                         [288304],
                         [355744],
                         [519509],
                         [532486],
                         [121469]])

        D = np.asmatrix([[30694],
                        [921546],
                        [179246],
                        [226932],
                        [140350],
                        [160268],
                        [310152],
                        [178727],
                        [474842],
                        [102913]])

        IOmat = np.asmatrix(IO)
        TCmat = IOmat / np.transpose(X)
        I = np.identity(10)
        TCinv = np.linalg.inv(I-TCmat)

        # Apply shocks to sectors
        dD = np.zeros([10,1])
        for sec_choice in shockSector:
            sector = sectorDict[sec_choice]
            dD[sector] = signs[shockSign]*(shockSize/100)*D[sector]

        shockRounds = np.zeros([10,NumRounds])
        dXround = dD
        for shockRound in range(NumRounds):
            dXround = TCmat*dXround
            shockRounds[:,shockRound] = dXround.flatten()

        # Concatenate the initial shock, calculate cumulative effect and 
        # the % captured by the rounds
        shockRoundsCum = np.cumsum(
                np.concatenate((dD,shockRounds), axis = 1), axis = 1)
        shockRounds = np.concatenate((dD,shockRounds),axis = 1)
        dX = TCinv*dD
        cover = float( 100*sum(shockRoundsCum[:,-1])/sum(dX))

        # Normalise if required, get bounds
        if Normalise is True:
            shockRounds /= X
            shockRoundsCum /= sum(X)

        yMax = 1.2*max(shockRounds.flatten())        
        yMin = 1.2*min(shockRounds.flatten())
        yMaxCum = 1.1*sum(shockRoundsCum[:,-1]) 

        # Create figure
        fig, ax = plt.subplots(figsize=(20,10))
        roundsVec = np.arange(0,NumRounds+1)

        # Plot shocks by sector and round
        if plotType == 'Shocks over time':

            for sector in sectorDict:
                ax.plot(roundsVec, shockRounds[sectorDict[sector],:], 
                        colors[sector], linewidth=2, alpha=0.6, label=sector)

            # Add legend and format axes to look nice
            ax.autoscale(enable=True, axis='both', tight=True)
            if shockSign == 'Positive':
                ax.set_ylim(top = yMax, bottom = 0)
                loc = 'upper right'
                ax.tick_params(labelsize=20)
                xLabelYPos = -0.05
                yAxisArrowPos = 1
                yAxisArrow = "^k"
            elif shockSign == 'Negative':
                ax.set_ylim(top = 0, bottom = yMin)
                loc = 'lower right'
                ax.tick_params(labelsize=20,labelbottom=False,labeltop=True)
                xLabelYPos = 1.075
                yAxisArrowPos = 0
                yAxisArrow = "vk"

            ax.set_xlim(right = NumRounds, left = 0)
            ax.legend(loc=loc, frameon=False,prop={'size':20})
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_position('zero')
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Rounds', fontdict = {'fontsize': 25})
            ax.xaxis.set_label_coords(0.5, xLabelYPos)
            ax.set_ylabel('Impact', fontdict = {'fontsize': 25})
            
            ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
            ax.plot(0, yAxisArrowPos, yAxisArrow, transform=ax.get_xaxis_transform(), clip_on=False)


        elif plotType == 'Cumulative shocks':

            bottomVals = np.zeros(NumRounds+1)

            for sector in sectorDict:

                barColor = colors[sector][0]
                if '--' in colors[sector]:
                    if sectorDict[sector] % 2 == 0:
                        barHatch = '/'
                    else:
                        barHatch = '\\'
                else:
                    barHatch = ''

                ax.bar(roundsVec, shockRoundsCum[sectorDict[sector],:], 
                       width = 0.5, bottom = bottomVals, color = barColor, 
                       hatch = barHatch, alpha = 0.6, label=sector)
                bottomVals += shockRoundsCum[sectorDict[sector],:]

            coverStr = '{:d} rounds capture {:.2f}% of the total effect'.format(
                NumRounds,cover)

            ax.autoscale(enable=True, axis='both', tight=True)
            if shockSign == 'Positive':
                ax.set_ylim(top = yMaxCum, bottom = 0)
                legLoc = 'upper center'
                annotLoc1 = 1
                annotLoc2 = 0.90
                ax.tick_params(labelsize=20,labelbottom=True,labeltop=False)
                xLabelYPos = -0.05
                yAxisArrowPos = 1
                yAxisArrow = "^k"
            elif shockSign == 'Negative':
                ax.set_ylim(top = 0, bottom = yMaxCum)
                legLoc = 'lower center'
                annotLoc1 = 0
                annotLoc2 = 0.05                
                ax.tick_params(labelsize=20,labelbottom=False,labeltop=True)
                xLabelYPos = 1.075
                yAxisArrowPos = 0
                yAxisArrow = "vk"

            ax.annotate(coverStr,[0,annotLoc1], xytext = [0.05,annotLoc2], 
                        xycoords ='axes fraction', fontsize = 25, clip_on = True)
                
            ax.set_xlim(right = NumRounds+0.5, left = -0.5)
            ax.legend(loc=legLoc, frameon=False,prop={'size':20},ncol=3, 
                      bbox_to_anchor=(0.5, -0.25))
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Rounds', fontdict = {'fontsize': 25})
            ax.xaxis.set_label_coords(0.5, xLabelYPos)
            ax.set_ylabel('Cumulative Impact', fontdict = {'fontsize': 25})

            ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
            ax.plot(-0.5, yAxisArrowPos, yAxisArrow, transform=ax.get_xaxis_transform(), clip_on=False)



        plt.tight_layout()
    
    out = widgets.interactive_output(uk_IO_plot, {'shockSector': shockSector_list,
                                                   'shockSign': shockSign_list,
                                                   'shockSize': shockSize_slider, 
                                                   'NumRounds': NumRounds_slider,
                                                   'plotType': plotType_list,                                                      
                                                   'Normalise': Normalise_check})
    
    lblStr = 'Pick sector(s) to shock:'
    output = widgets.VBox([out,
                  widgets.HBox([plotType_list,
                                NumRounds_slider,
                                Normalise_check]),
                  widgets.HBox([widgets.Label(value=lblStr)]),      
                  widgets.HBox([shockSector_list,
                                shockSign_list,
                                shockSize_slider])
                          ])
    
    display(output)
    
def total_revenue_widget(Qmax_init = 15, Pmax_init = 30, Qval_init = 7.5, a_d_init = 2, b_d_init = 25):

    # Declare widgets for interactive input
    Qmax_slider = widgets.IntSlider(min= 5,
                                 max=50,
                                 step=1,
                                 description=r'Maximum $Q$:',
                                 value = Qmax_init,
                                 continuous_update =False)
    Pmax_slider = widgets.IntSlider(min=5,
                                 max=50,
                                 step=1,
                                 description=r'Maximum $P$:',
                                 value = Pmax_init,
                                 continuous_update =False)
    Qval_slider = widgets.FloatSlider(min=0,
                                 max=50,
                                 description='Pick a Quantity:',
                                 value = Qval_init,
                                 continuous_update =False)
    a_d_slider = widgets.FloatSlider(min=0,
                                 max=10,
                                 description= r'$a_d$:',
                                 value = a_d_init,
                                 continuous_update =False)
    b_d_slider = widgets.FloatSlider(min=0,
                                 max=50,
                                 description= r'$b_d$:',
                                 value=b_d_init,
                                 continuous_update =False)
    
    # Link widgets as required
    widgets.jslink((Qmax_slider,'value'),(Qval_slider,'max'))

    def total_revenue_plot(Qmax,Pmax, Qval, a_d, b_d):

        # create a quantity vector, calculate demand/Total revenue vectors
        Q = np.arange(0,Qmax,Qmax/500)

        P_d = -a_d*Q + b_d
        TR = -a_d*Q**2 + b_d*Q

        # Calculate prices/revenues for selected value
        Pval_d = -a_d*Qval + b_d
        TRval = -a_d*Qval**2 + b_d*Qval
        if a_d == 0:
            TRmax = 1.25*b_d*Qmax
        else:
            TRmax = 1.25*(b_d**2)/(4*a_d)

        # Create figure
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,10))

        # Plot demand
        ax[0].plot(Q, P_d,'b', linewidth=2, alpha=0.6,
                    label=r'Inverse Demand $\quad P_d = -a_d Q + b_d$')

        # Add markers for the price/quantity points, with dotted lines
        ax[0].fill([0,Qval,Qval,0],[0,0,Pval_d,Pval_d],'b',alpha = 0.2,
                        label = r'Total revenue $P_d \times Q$')
        ax[0].scatter(Qval, Pval_d, s=mrkrSize, c='k', alpha=0.6,
                    label='Selection')
        ax[0].plot([0,Qval],[Pval_d,Pval_d],'k--',linewidth=1)
        ax[0].plot([Qval,Qval],[0,Pval_d],'k--',linewidth=1)
        ax[0].annotate(r'$Q={:.2f}$'.format(Qval),[Qval,0], 
                    xytext = [Qval+0.15,0.25], xycoords ='data', fontsize = 25, 
                    clip_on = True)
        ax[0].annotate(r'$P_d={:.2f}$'.format(Pval_d),[0,Pval_d], 
                    xytext = [0.15,Pval_d+0.25], xycoords ='data', fontsize = 25, 
                    clip_on = True)

        # Add legend and format axes to look nice
        ax[0].legend(loc='upper center', frameon=False,prop={'size':20})
        ax[0].autoscale(enable=True, axis='both', tight=True)
        ax[0].set_ylim(top = Pmax, bottom = 0)
        ax[0].set_xlim(right = Qmax, left = 0)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_xlabel(r'$Q$', fontdict = {'fontsize': 25},position=(1, 0))
        ax[0].set_ylabel(r'$P$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax[0].plot(1, 0, ">k", transform=ax[0].get_yaxis_transform(), clip_on=False)
        ax[0].plot(0, 1, "^k", transform=ax[0].get_xaxis_transform(), clip_on=False)
        ax[0].tick_params(labelsize=20)

        # Plot Total revenue
        ax[1].plot(Q, TR,'b', linewidth=2, alpha=0.6,
                    label=r'Total revenue $\quad TR = -a_d Q^2 + b_d Q$')


        # Add markers for the price/quantity points, with dotted lines
        ax[1].scatter(Qval, TRval, s=mrkrSize, c='k', alpha=0.6,
                    label='Selection')
        ax[1].plot([0,Qval],[TRval,TRval],'k--',linewidth=1)
        ax[1].plot([Qval,Qval],[0,TRval],'k--',linewidth=1)
        ax[1].annotate(r'$Q={:.2f}$'.format(Qval),[Qval,0], 
                       xytext = [Qval+0.15,0.25],
                       xycoords ='data', fontsize = 25, clip_on = True)
        ax[1].annotate(r'$TR={:.2f}$'.format(TRval),[0,TRval], 
                       xytext = [0.15,TRval+0.25],
                       xycoords ='data', fontsize = 25, clip_on = True)

        # Add legend and format axes to look nice
        ax[1].legend(loc='upper center', frameon=False,prop={'size':20})
        ax[1].autoscale(enable=True, axis='both', tight=True)
        ax[1].set_ylim(top = TRmax, bottom = 0)
        ax[1].set_xlim(right = Qmax, left = 0)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_xlabel(r'$Q$', fontdict = {'fontsize': 25},position=(1, 0))
        ax[1].set_ylabel(r'$TR$', fontdict = {'fontsize': 25},position=(0, 1), 
                         rotation=0)
        ax[1].plot(1, 0, ">k", transform=ax[1].get_yaxis_transform(), 
                   clip_on=False)
        ax[1].plot(0, 1, "^k", transform=ax[1].get_xaxis_transform(), 
                   clip_on=False)
        ax[1].tick_params(labelsize=20)  

        plt.tight_layout()
    
    out = widgets.interactive_output(total_revenue_plot, {'Qmax': Qmax_slider,
                                                       'Pmax': Pmax_slider,
                                                       'Qval': Qval_slider, 
                                                       'a_d': a_d_slider,
                                                       'b_d': b_d_slider})

    output = widgets.VBox([out,
                  widgets.HBox([Qmax_slider,
                               Pmax_slider,
                               Qval_slider]),
                  widgets.HBox([a_d_slider, 
                                b_d_slider])])
    display(output)    
    
    
def compound_interest_widget(xMin_init = 0, xMax_init = 1.1, yMin_init = 0, 
                             yMax_init = 3, numTermsStr_init = '1', 
                             expFlag_init = False):
    
    # Declare widgets for interactive input
    xMin_slider = widgets.FloatSlider(min=0,
                                 max=1.1,
                                 description=r'Min $x$:',
                                 value = xMin_init,
                                 continuous_update =False)
    xMax_slider = widgets.FloatSlider(min=0,
                                 max=1.1,
                                 description=r'Max $x$:',
                                 value = xMax_init,
                                 continuous_update =False)
    yMin_slider = widgets.FloatSlider(min=0,
                                 max=3,
                                 description=r'Min $y$:',
                                 value = yMin_init,
                                 continuous_update =False)
    yMax_slider = widgets.FloatSlider(min=0,
                                 max=3,
                                 description=r'Max $y$:',
                                 value = yMax_init,
                                 continuous_update =False)
    numTermsStr_text = widgets.Text(value = numTermsStr_init,
                                placeholder='Enter numbers separated by commas',
                                description='N° terms:',
                                disabled=False)
    expFlag_check = widgets.Checkbox(value = expFlag_init,
                                   description='Include Exponential',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    widgets.jslink((xMin_slider,'value'),(xMax_slider,'min'))
    widgets.jslink((xMax_slider,'value'),(xMin_slider,'max'))
    widgets.jslink((yMin_slider,'value'),(yMax_slider,'min'))
    widgets.jslink((yMax_slider,'value'),(yMin_slider,'max'))

    def compound_interest_plot(xMin, xMax, yMin, yMax, numTermsStr, expFlag):

        numTermsList = numTermsStr.split(',')

        if len(numTermsList) > 6:
            numTermsList = numTermsList[0:6]

        colors = ['b','r','g','m','c','y']

        # Create figure, plot compounded functions
        fig, ax = plt.subplots(figsize=(20,10))
        mrkrSize = 2*rcParams['lines.markersize'] ** 2

        for index, item in enumerate(numTermsList):
            
            # Process entry if item is valid
            if not item == '' :
                numTerms = int(item)

                step = 1/numTerms
                rate = 1 + step
                K,T = 1,0
                ax.scatter(T, K, s=mrkrSize, c='b', alpha=0.6)
                col = colors[index]

                for n in range(numTerms):
                    if n == 0:
                        termLabel = r'N$^\circ$ terms: {:d}'.format(numTerms)
                    else:
                        termLabel = None
                    ax.scatter(T+step, K*rate, s=mrkrSize, c=col, alpha=0.6)
                    ax.plot([T,T+step], [K,K], col, linewidth=2, alpha=0.6, 
                            label=termLabel)
                    ax.plot([T+step,T+step], [0,K*rate], col+'--', linewidth=2,
                            alpha=0.6)
                    T+=step
                    K*=rate

                ax.plot([0,T],[K,K],col+'--',linewidth=1)
                ax.annotate(r'Final value = ${:.4f}$'.format(K),[0,K], 
                            xytext = [0.05,K+0.1], xycoords ='data', 
                            fontsize = 25, clip_on = True)

        # Plot the exponential if requested
        if expFlag is True:
            x = np.arange(0,1,1/500)
            y = np.exp(x)
            ax.plot(x, y,'k', linewidth=2, label=r'$\quad y = e^x$')

        # Add legend and format axes to look nice
        ax.legend(loc='lower center', frameon=False,prop={'size':20},ncol=6,
                   bbox_to_anchor=(0.5, -0.25))

        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = yMax, bottom = yMin)
        ax.set_xlim(right = xMax, left = xMin)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$T$', fontdict = {'fontsize': 25},position=(1, 0))
        ax.set_ylabel(r'Asset value', fontdict = {'fontsize': 25},
                      position=(0, 1.05), rotation=90)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
    
    out = widgets.interactive_output(
                        compound_interest_plot, {'xMin': xMin_slider,
                                                 'xMax': xMax_slider,
                                                 'yMin': yMin_slider,
                                                 'yMax': yMax_slider,
                                                 'numTermsStr': numTermsStr_text,
                                                 'expFlag' : expFlag_check})

    output = widgets.VBox([out,
                  widgets.HBox([xMin_slider,
                               xMax_slider,
                               yMin_slider,
                               yMax_slider]),
                  widgets.HBox([numTermsStr_text, 
                                expFlag_check])])
    display(output)  
    
def log_demand_widget(Pmax_init = 15,Qmax_init = 10, Pval1_init = 5, 
                      Pval2_init = 5, a_d_init = 0.8, b_d_init = 1.5):

    # Declare widgets for interactive input
    Qmax_slider = widgets.IntSlider(min= 5,
                                 max=50,
                                 step=1,
                                 description=r'Maximum $Q$:',
                                 value = Qmax_init,
                                 continuous_update =False)
    Pmax_slider = widgets.IntSlider(min=5,
                                 max=50,
                                 step=1,
                                 description=r'Maximum $P$:',
                                 value = Pmax_init,
                                 continuous_update =False)
    Pval1_slider = widgets.FloatSlider(min=0,
                                 max=50,
                                 description='Pick $P_1$:',
                                 value = Pval1_init,
                                 continuous_update =False)
    Pval2_slider = widgets.FloatSlider(min=0,
                                 max=15,
                                 description='Pick $P_2$:',
                                 value = Pval1_init,
                                 continuous_update =False)
    a_d_slider = widgets.FloatSlider(min=0,
                                 max=4,
                                 description= r'$\alpha$:',
                                 value = a_d_init,
                                 continuous_update =False)
    b_d_slider = widgets.FloatSlider(min=0,
                                 max=4,
                                 description= r'$\beta$:',
                                 value=b_d_init,
                                 continuous_update =False)
    
    # Link widgets as required
    widgets.jslink((Pmax_slider,'value'),(Pval1_slider,'max'))

    def log_demand_plot(Pmax, Qmax, Pval1, Pval2, a_d, b_d):

        # create a quantity vector, calculate demand/Total revenue vectors
        P = np.arange(0.01,Pmax,(Pmax-0.01)/500)
        p = np.log(P)
        pval1 = np.log(Pval1)

        Q = (P**(-a_d)) * (Pval2**b_d)
        q = -a_d*p + b_d*np.log(Pval2)

        # Calculate prices/revenues for selected value
        Qval = (Pval1**(-a_d)) * (Pval2**b_d)
        qval = -a_d*pval1 + b_d*np.log(Pval2)

        # Create figure
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,10))

        # Plot demand
        ax[0].plot(P, Q,'b', linewidth=2, alpha=0.6,
                    label=r'Demand $\quad Q_1 = P_1^{-\alpha} P_2^{\beta}$')

        # Add markers for the price/quantity points, with dotted lines
        ax[0].scatter(Pval1, Qval, s=mrkrSize, c='k', alpha=0.6,
                    label='Selection')
        ax[0].plot([0,Pval1],[Qval,Qval],'k--',linewidth=1)
        ax[0].plot([Pval1,Pval1],[0,Qval],'k--',linewidth=1)
        ax[0].annotate(r'$P_1={:.2f}$'.format(Pval1),[Qval,0], 
                    xytext = [Pval1+0.15,0.25], xycoords ='data', fontsize = 25, 
                    clip_on = True)
        ax[0].annotate(r'$Q_1={:.2f}$'.format(Qval),[0,Qval], 
                    xytext = [0.15,Qval+0.25], xycoords ='data', fontsize = 25, 
                    clip_on = True)

        # Add legend and format axes to look nice
        ax[0].legend(loc='upper center', frameon=False,prop={'size':20})
        ax[0].autoscale(enable=True, axis='both', tight=True)
        ax[0].set_ylim(top = Qmax, bottom = 0)
        ax[0].set_xlim(right = Pmax, left = 0)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_xlabel(r'$P$', fontdict = {'fontsize': 25},position=(1, 0))
        ax[0].set_ylabel(r'$Q$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax[0].plot(1, 0, ">k", transform=ax[0].get_yaxis_transform(), clip_on=False)
        ax[0].plot(0, 1, "^k", transform=ax[0].get_xaxis_transform(), clip_on=False)
        ax[0].tick_params(labelsize=20)

        # Plot Logarithnmic demand
        ax[1].plot(p, q,'b', linewidth=2, alpha=0.6,
                    label=r'Log Demand $\quad q_1 = -\alpha p_1 + \beta p_2$')

        # Add markers for the price/quantity points, with dotted lines
        ax[1].scatter(pval1, qval, s=mrkrSize, c='k', alpha=0.6,
                    label='Selection')
        ax[1].plot([0,pval1],[qval,qval],'k--',linewidth=1)
        ax[1].plot([pval1,pval1],[0,qval],'k--',linewidth=1)
        ax[1].annotate(r'$\log P_1={:.2f}$'.format(pval1),[qval,0], 
                    xytext = [pval1+0.05,0.1], xycoords ='data', fontsize = 25, 
                    clip_on = True)
        ax[1].annotate(r'$\log Q_1={:.2f}$'.format(qval),[0,qval], 
                    xytext = [0.05,qval+0.1], xycoords ='data', fontsize = 25, 
                    clip_on = True)

        # Add legend and format axes to look nice
        ax[1].legend(loc='upper center', frameon=False,prop={'size':20})
        ax[1].autoscale(enable=True, axis='both', tight=True)
        ax[1].set_ylim(top = np.log(Qmax), bottom = 0)
        ax[1].set_xlim(right = np.log(Pmax), left = 0)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_xlabel(r'$\log P$', fontdict = {'fontsize': 25},position=(1, 0))
        ax[1].set_ylabel(r'$\log Q$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax[1].plot(1, 0, ">k", transform=ax[1].get_yaxis_transform(), clip_on=False)
        ax[1].plot(0, 1, "^k", transform=ax[1].get_xaxis_transform(), clip_on=False)
        ax[1].tick_params(labelsize=20)

        plt.tight_layout()
    
    out = widgets.interactive_output(log_demand_plot, {'Qmax': Qmax_slider,
                                                       'Pmax': Pmax_slider,
                                                       'Pval1': Pval1_slider,
                                                       'Pval2': Pval2_slider,
                                                       'a_d': a_d_slider,
                                                       'b_d': b_d_slider})

    output = widgets.VBox([out,
                  widgets.HBox([Qmax_slider,
                               Pmax_slider,
                               Pval1_slider,
                               Pval2_slider]),
                  widgets.HBox([a_d_slider, 
                                b_d_slider])])
    
    display(output)
    
def profit_maximisation_widget(Qmax_init = 8,  a_d_init = 2, b_d_init = 5, c_d_init = 50, 
                   a_s_init = 1, b_s_init = -4, c_s_init = 10, d_s_init = 0, 
                   revFlag_init = True, cstFlag_init = True):

    # Declare widgets for interactive input
    Qmax_slider = widgets.IntSlider(min= 5,
                                 max=50,
                                 step=1,
                                 description=r'Maximum $Q$:',
                                 value = Qmax_init,
                                 continuous_update =False)
    a_d_slider = widgets.FloatSlider(min=0,
                                 max=10,
                                 description= r'$a_d$:',
                                 value = a_d_init,
                                 continuous_update =False)
    b_d_slider = widgets.FloatSlider(min=-50,
                                 max=50,
                                 description= r'$b_d$:',
                                 value=b_d_init,
                                 continuous_update =False)
    c_d_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$c_d$:',
                                 value = c_d_init,
                                 continuous_update =False)
    a_s_slider = widgets.FloatSlider(min=0,
                                 max=10,
                                 description= r'$a_s$:',
                                 value = a_s_init,
                                 continuous_update =False)
    b_s_slider = widgets.FloatSlider(min=-50,
                                 max=50,
                                 description= r'$b_s$:',
                                 value=b_s_init,
                                 continuous_update =False)
    c_s_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$c_s$:',
                                 value = c_s_init,
                                 continuous_update =False)
    d_s_slider = widgets.FloatSlider(min=-100,
                                 max=100,
                                 description= r'$d_s$:',
                                 value = d_s_init,
                                 continuous_update =False)
    revFlag_check = widgets.Checkbox(value = revFlag_init,
                                   description='Show revenue',
                                   disabled=False,
                                   indent=True)
    cstFlag_check = widgets.Checkbox(value = cstFlag_init,
                                   description='Show costs',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required


    def profit_plot(Qmax, a_d, b_d, c_d, a_s, b_s, c_s, d_s, revFlag, cstFlag):

        # create a quantity vector
        Q = np.arange(-1e-3,Qmax,(Qmax-1e-3)/500)

        # Calculate Revenue vectors
        TR = lambda Q: -a_d*Q**3 + b_d*Q**2 + c_d*Q
        AR = lambda Q: -a_d*Q**2 + b_d*Q + c_d
        mR = lambda Q: -3*a_d*Q**2 + 2*b_d*Q + c_d

        # Calculate Cost vectors
        TC = lambda Q: a_s*Q**3 + b_s*Q**2 + c_s*Q + d_s
        AC = lambda Q: a_s*Q**2 + b_s*Q + c_s + d_s/Q
        mC = lambda Q: 3*a_s*Q**2 + 2*b_s*Q + c_s

        # Function labels
        sig = []
        for param in [b_d, c_d, b_s, c_s, d_s]:
            if param >= 0:
                sig.append('+')
            else:
                sig.append('-')

        lblTR = r'$TR = -{:.0f}Q^3 {:s} {:.0f}Q^2 {:s} {:.0f}Q$'
        lblAR = r'$AR = -{:.0f}Q^2 {:s} {:.0f}Q {:s} {:.0f}$'
        lblmR = r'$mR = -{:.0f}Q^2 {:s} {:.0f}Q {:s} {:.0f}$'

        lblTC = r'$TC = {:.0f}Q^3 {:s} {:.0f}Q^2 {:s} {:.0f}Q {:s} {:.0f}$'
        lblAC = r'$AC = {:.0f}Q^2 {:s} {:.0f}Q {:s} {:.0f} {:s} {:.0f}/Q$'
        lblmC = r'$mC = {:.0f}Q^2 {:s} {:.0f}Q {:s} {:.0f}$'

        # find profit-maximising point
        Qval = Q[np.argmax(TR(Q)-TC(Q))]
        PiMax = TR(Qval) - TC(Qval)
        Pmax = max(max(AR(Q)),max(AC(Q)),max(mR(Q)),max(mC(Q)))
        TRmax = max(max(TR(Q)),max(TC(Q)))

        # Create figure
        mrkrSize = 2*rcParams['lines.markersize'] ** 2
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,10))

        # Plot revenue if requested
        if revFlag is True:
            ax[0].plot(Q, AR(Q),'b', linewidth=2, alpha=0.6,
              label=lblAR.format(
                      a_d, sig[0], abs(b_d), sig[1], abs(c_d) ))
            ax[0].plot(Q, mR(Q),'b--', linewidth=2, alpha=0.6,
              label=lblmR.format(
                      3*a_d, sig[0], abs(2*b_d), sig[1], abs(c_d) ))

        # Plot Cost if requested
        if cstFlag is True:
            ax[0].plot(Q, AC(Q),'r', linewidth=2, alpha=0.6,
              label = lblAC.format(
                      a_s, sig[2], abs(b_s), sig[3], abs(c_s), sig[4], abs(d_s)))
            ax[0].plot(Q, mC(Q),'r--', linewidth=2, alpha=0.6,
              label=lblmC.format(
                      3*a_s, sig[2], abs(2*b_s), sig[3], abs(c_s) ))

        if revFlag is True and cstFlag is True:
            # Plot profits
            ax[0].fill([0,Qval,Qval,0],[AC(Qval),AC(Qval),AR(Qval),AR(Qval)],'g',alpha = 0.2,
                        label = r'$\Pi = (AR-AC) \times Q$')
            ax[0].scatter([Qval, Qval, Qval],
                          [AR(Qval),mR(Qval),AC(Qval)],
                          s=mrkrSize, c='k', alpha=0.6, label='Max profits')
            ax[0].plot([Qval,Qval],[0,AR(Qval)],'k--',linewidth=1)    
            ax[0].plot([0,Qval],[AR(Qval),AR(Qval)],'k--',linewidth=1)
            ax[0].plot([0,Qval],[AC(Qval),AC(Qval)],'k--',linewidth=1)

        # Add legend and format axes to look nice
        if revFlag is True or cstFlag is True:
            ax[0].legend(loc='upper center', frameon=False,prop={'size':20})
        ax[0].autoscale(enable=True, axis='both', tight=True)
        ax[0].set_ylim(top = Pmax, bottom = 0)
        ax[0].set_xlim(right = Qmax, left = 0)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_xlabel(r'$Q$', fontdict = {'fontsize': 25},position=(1, 0))
        ax[0].set_ylabel(r'$£$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax[0].plot(1, 0, ">k", transform=ax[0].get_yaxis_transform(), clip_on=False)
        ax[0].plot(0, 1, "^k", transform=ax[0].get_xaxis_transform(), clip_on=False)
        ax[0].tick_params(labelsize=20)

        # Plot Total revenue
        if revFlag is True:
            ax[1].plot(Q, TR(Q),'b', linewidth=2, alpha=0.6,
              label=lblTR.format(
                      a_d, sig[0], abs(b_d), sig[1], abs(c_d) ))

        # Plot Cost if requested
        if cstFlag is True:
            ax[1].plot(Q, TC(Q),'r', linewidth=2, alpha=0.6,
              label = lblTC.format(
                      a_s, sig[2], abs(b_s), sig[3], abs(c_s), sig[4], abs(d_s)))

        if revFlag is True and cstFlag is True:
            # Plot profits
            ax[1].plot(Q, TR(Q) - TC(Q),'g', linewidth=2, alpha=0.6,
                        label=r'$\Pi = TR -TC$')

            # Add markers for the optimal points, with dotted lines
            ax[1].scatter([Qval,Qval,Qval], 
                         [TR(Qval),TC(Qval),PiMax], 
                         s=mrkrSize, c='k', alpha=0.6, label='Max profits')
            ax[1].plot([Qval,Qval],[0,TR(Qval)],'k--',linewidth=1)
            ax[1].plot([Qval,Qval],[0,TR(Qval)],'k--',linewidth=1)

            # Add parallel slope lines
            dQ = Qmax*0.1
            ax[1].plot([Qval-dQ,Qval+dQ],
                      [PiMax,PiMax],
                      'k--',linewidth=1)
            ax[1].plot([Qval-dQ,Qval+dQ],
                      [TC(Qval)-dQ*mC(Qval),TC(Qval)+dQ*mC(Qval)],
                      'k--',linewidth=1)
            ax[1].plot([Qval-dQ,Qval+dQ],
                      [TR(Qval)-dQ*mR(Qval),TR(Qval)+dQ*mR(Qval)],
                      'k--',linewidth=1)

        # Add legend and format axes to look nice
        if revFlag is True or cstFlag is True:
            ax[1].legend(loc='upper center', frameon=False,prop={'size':20})
        ax[1].autoscale(enable=True, axis='both', tight=True)
        ax[1].set_ylim(top = TRmax, bottom = 0)
        ax[1].set_xlim(right = Qmax, left = 0)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_xlabel(r'$Q$', fontdict = {'fontsize': 25},position=(1, 0))
        ax[1].set_ylabel(r'$£$', fontdict = {'fontsize': 25},position=(0, 1), rotation=0)
        ax[1].plot(1, 0, ">k", transform=ax[1].get_yaxis_transform(), clip_on=False)
        ax[1].plot(0, 1, "^k", transform=ax[1].get_xaxis_transform(), clip_on=False)
        ax[1].tick_params(labelsize=20)  

        plt.tight_layout()
    
    out = widgets.interactive_output(profit_plot, {'Qmax': Qmax_slider,
                                                   'a_d': a_d_slider,
                                                   'b_d': b_d_slider,
                                                   'c_d': c_d_slider,
                                                   'a_s': a_s_slider,
                                                   'b_s': b_s_slider,
                                                   'c_s': c_s_slider,
                                                   'd_s': d_s_slider,
                                               'revFlag': revFlag_check,
                                               'cstFlag': cstFlag_check})
    
    output = widgets.VBox([out,
                  widgets.HBox([Qmax_slider,
                               revFlag_check,
                               cstFlag_check]),
                  widgets.HBox([a_d_slider, 
                                b_d_slider,
                                c_d_slider]),
                  widgets.HBox([a_s_slider, 
                                b_s_slider,
                                c_s_slider,
                                d_s_slider])])
    display(output)
    
def plot_income_widget(stats_init = False, max_init = 100, manual_axis = False, 
                       loc = 'assets/income.csv'):

    # Declare widgets for interactive input
    stats_check = widgets.Checkbox(value = stats_init,
                                   description='Show stats',
                                   disabled=False,
                                   indent=True)
    
    manual_check = widgets.Checkbox(value = manual_axis,
                                   description='Manual maximum ',
                                   disabled=False,
                                   indent=True) 
    
    max_box = widgets.widgets.FloatText(value = max_init,
                                description='Maximum:',
                                disabled=False)
    
    # Link widgets as required
    # Not needed here
    
    def plot_income(stats, max_check, max_val):
        
        # Import income data
        load_data = np.array(pd.read_csv(loc, header=None).values)
        obs = load_data.shape[0]
        income = load_data[0:obs-1,0].flatten()
        freq = load_data[0:obs-1,1].flatten()
        income_midpoint = income + 0.5

        # Set maximum of income bins
        if max_check is True:
            max_income = max_val
        else:
            max_income = max(income)
        
        # Format colors (alternating pattern of 5)
        colorbars = []
        for i in range(int(np.ceil(obs/10))):
            colorbars+= 5*['b'] + 5*['g']
        colorbars=colorbars[0:obs-1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20,10))
        ax.bar(income_midpoint, freq, color = colorbars, width = 1, edgecolor = 'black',alpha = 0.5)

        # Calculate and add central tendency stats if needed
        if stats is True:
            N = sum(freq)
            cum_freq = np.cumsum(freq)

            mean = sum(freq*income_midpoint)/N
            mode = income_midpoint[freq == max(freq)][0]
            median_cls = np.argwhere(cum_freq > N/2)[0][0]
            median = income[median_cls - 1] + (N/2 - cum_freq[median_cls - 1])/freq[median_cls]
            
            mean_ind = np.argwhere(income > mean)[0][0]
            y_mean = freq[mean_ind - 1]
            y_mode = max(freq)
            y_median = freq[median_cls - 1]
            ax.plot([mean,mean],[0,y_mean*1.15],'k--',linewidth=1)
            ax.annotate(r'Mean: {:10.4f} '.format(mean), [mean, y_mean*1.15], 
                xytext = [mean + 0.15,y_mean*1.15], xycoords ='data', fontsize = 25, 
                clip_on = True)
            ax.plot([mode,mode],[0,y_mode*1.15],'k--',linewidth=1)
            ax.annotate(r'Mode: {:10.4f} '.format(mode), [mode, y_mode*1.15], 
                xytext = [mode + 0.15, y_mode*1.15], xycoords ='data', fontsize = 25, 
                clip_on = True)
            ax.plot([median,median],[0,y_median*1.15],'k--',linewidth=1)
            ax.annotate(r'Median: {:10.4f} '.format(median), [median, y_median*1.15], 
                xytext = [median + 0.15, y_median*1.15], xycoords ='data', fontsize = 25, 
                clip_on = True)
        
        # Add legend and format axes to look nice
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top=max(freq)*1.25)
        ax.set_xlim(left=0,right= max_income + 2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Income band (£1000)', fontdict = {'fontsize': 25})
        ax.set_ylabel('Number of households (thousands)', fontdict = {'fontsize': 25})
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.tick_params(labelsize=20)

        plt.tight_layout()
    
    out = widgets.interactive_output(plot_income, {'stats': stats_check,
                                                  'max_check':manual_check,
                                                  'max_val':max_box})
    
    output = widgets.VBox([out,
                  widgets.HBox([stats_check, manual_check, max_box])
                          ])
    display(output)
    
#------------------------------------------------------------------------------
# Statistics widgets

def plot_dist_widget(dist_1_init = '-', d1_p1_init = 0, d1_p2_init = 0, d1_frmt_init = '-',
                     dist_2_init = '-', d2_p1_init = 0, d2_p2_init = 0, d2_frmt_init = '-',
                     dist_3_init = '-', d3_p1_init = 0, d3_p2_init = 0, d3_frmt_init = '-',
                     dist_4_init = '-', d4_p1_init = 0, d4_p2_init = 0, d4_frmt_init = '-',
                     min_init = -3, max_init = 3, stats_init = False, manual_axis = False):

    # Declare widgets for interactive input
    dist_list_1 = widgets.Dropdown(options=['-','Binomial','Normal','Lognormal','Student t'],
                                value = dist_1_init,
                                description='Distribution:',
                                disabled=False)
    dist_1_p1_box = widgets.widgets.FloatText(value = d1_p1_init,
                                description='Param 1:',
                                disabled=False)
    dist_1_p2_box = widgets.widgets.FloatText(value = d1_p2_init,
                                description='Param 2:',
                                disabled=False)
    dist_1_formatStr = widgets.Text(value = d1_frmt_init,
                                placeholder = '',
                                description='Format:',
                                disabled=False)
    dist_list_2 = widgets.Dropdown(options=['-','Binomial','Normal','Lognormal','Student t'],
                                value = dist_2_init,
                                description='Distribution:',
                                disabled=False)
    dist_2_p1_box = widgets.widgets.FloatText(value = d2_p1_init,
                                description='Param 1:',
                                disabled=False)
    dist_2_p2_box = widgets.widgets.FloatText(value = d2_p2_init,
                                description='Param 2:',
                                disabled=False)
    dist_2_formatStr = widgets.Text(value = d2_frmt_init,
                                placeholder = '',
                                description='Format:',
                                disabled=False)
    dist_list_3 = widgets.Dropdown(options=['-','Binomial','Normal','Lognormal','Student t'],
                                value = dist_3_init,
                                description='Distribution:',
                                disabled=False)
    dist_3_p1_box = widgets.widgets.FloatText(value = d3_p1_init,
                                description='Param 1:',
                                disabled=False)
    dist_3_p2_box = widgets.widgets.FloatText(value = d3_p2_init,
                                description='Param 2:',
                                disabled=False)
    dist_3_formatStr = widgets.Text(value = d3_frmt_init,
                                placeholder = '',
                                description='Format:',
                                disabled=False)
    dist_list_4 = widgets.Dropdown(options=['-','Binomial','Normal','Lognormal','Student t'],
                                value = dist_4_init,
                                description='Distribution:',
                                disabled=False)
    dist_4_p1_box = widgets.widgets.FloatText(value = d4_p1_init,
                                description='Param 1:',
                                disabled=False)
    dist_4_p2_box = widgets.widgets.FloatText(value = d4_p2_init,
                                description='Param 2:',
                                disabled=False)
    dist_4_formatStr = widgets.Text(value = d4_frmt_init,
                                placeholder = '',
                                description='Format:',
                                disabled=False)
    min_box = widgets.widgets.FloatText(value = min_init,
                                description='Min',
                                disabled=False)
    max_box = widgets.widgets.FloatText(value = max_init,
                                description='Max',
                                disabled=False)
    stats_check = widgets.Checkbox(value = stats_init,
                                   description='Show stats',
                                   disabled=False,
                                   indent=True)
    manual_check = widgets.Checkbox(value = manual_axis,
                                   description='Set limits',
                                   disabled=False,
                                   indent=True) 
    
    # Link widgets as required
    # Not needed here
    
    def plot_dist(dist_1, dist_1_p1, dist_1_p2, dist_1_frmt,
                  dist_2, dist_2_p1, dist_2_p2, dist_2_frmt,
                  dist_3, dist_3_p1, dist_3_p2, dist_3_frmt,
                  dist_4, dist_4_p1, dist_4_p2, dist_4_frmt,
                  min_val,max_val,stats,manual):
        
        param_list = [[dist_1, dist_1_p1, dist_1_p2, dist_1_frmt],
                      [dist_2, dist_2_p1, dist_2_p2, dist_2_frmt],
                      [dist_3, dist_3_p1, dist_3_p2, dist_3_frmt],
                      [dist_4, dist_4_p1, dist_4_p2, dist_4_frmt]]
        
        # Safety check for incorrect inputs
        if max_val < min_val:
            manual = False
        
        if manual:
            x_min = min_val
            x_max = max_val
        else:
            x_min = 0
            x_max = 0
            
        for param in param_list:
            dist = param[0]
            if dist == 'Binomial':
                n = param[1]
                p = param[2]
                if not manual:
                    x_min = min(x_min,binom.ppf(0.001,n,p))
                    x_max = max(x_max,binom.ppf(0.999,n,p))            
            elif dist == 'Normal':
                mu = param[1]
                sig2 = param[2]
                if not manual:
                    x_min = min(x_min,norm.ppf(0.001,mu,sig2**0.5))
                    x_max = max(x_max,norm.ppf(0.999,mu,sig2**0.5))
            elif dist == 'Lognormal':
                mu = param[1]
                sig2 = param[2]
                if not manual:
                    x_min = min(x_min,lognorm.ppf(0.01,sig2**0.5,0,np.exp(mu)))
                    x_max = max(x_max,lognorm.ppf(0.99,sig2**0.5,0,np.exp(mu)))
            elif dist == 'Student t':
                nu = param[1]
                if not manual:
                    x_min = min(x_min,t.ppf(0.01,nu))
                    x_max = max(x_max,t.ppf(0.99,nu))

        x = np.arange(np.floor(x_min),np.ceil(x_max)+1)
        x_N = np.linspace(x_min,x_max, 500)
        x_range = max(x) - min(x)

        # Create figure
        fig, ax = plt.subplots(figsize=(20,10))
        y_max = 0
        y = [0]
        for param in param_list:
            dist = param[0]
            frmt = param[-1]      
            if dist == 'Binomial':        
                n = param[1]
                p = param[2]
                y = binom.pmf(x, n, p)
                ax.plot(x, y, frmt+'o', ms=8, alpha=0.6,
                        label=r'$B('+str(n)+','+str(p)+')$')
                ax.vlines(x, 0, y,colors = frmt, linestyles='-', alpha=0.6)
                if stats is True:
                    mean = n*p
                    mode = np.round(n*p)
                    median = np.round((n+1)*p)
                    ax.plot([mean,mean],[0,max(y)*1.05],'k--',linewidth=1)
                    ax.annotate(r'Mean: {:10.4f} '.format(mean), [mean, max(y)*1.05], 
                        xytext = [mean + 0.15,max(y)*1.05], xycoords ='data', fontsize = 25, 
                        clip_on = True)
                    ax.plot([mode,mode],[0,max(y)*1.05],'k--',linewidth=1)
                    ax.annotate(r'Mode: {:10.4f} '.format(mode), [mode, max(y)*1.05], 
                        xytext = [mode + 0.15,max(y)*1.05], xycoords ='data', fontsize = 25, 
                        clip_on = True)
                    ax.plot([median,median],[0,max(y)*1.05],'k--',linewidth=1)
                    ax.annotate(r'Median: {:10.4f} '.format(median), [median, max(y)*1.05], 
                        xytext = [median + 0.15,max(y)*1.05], xycoords ='data', fontsize = 25, 
                        clip_on = True)
                
            elif dist == 'Normal':
                mu = param[1]
                sig2 = param[2]
                y = norm.pdf(x_N,mu,sig2**0.5)
                ax.plot(x_N, y,frmt, linewidth=2, 
                        alpha=0.6, label=r'$N('+str(mu)+','+str(sig2)+')$')
                if stats is True:
                    ax.plot([mu,mu],[0,max(y)*1.05],'k--',linewidth=1)
                    ax.annotate(r'Mean, Median, Mode: {:10.4f} '.format(mu), [mu, max(y)*1.05], 
                        xytext = [mu + 0.15,max(y)*1.05], xycoords ='data', fontsize = 25, 
                        clip_on = True)

            elif dist == 'Lognormal':
                mu = param[1]
                sig2 = param[2]
                y = lognorm.pdf(x_N,sig2**0.5,0,np.exp(mu))
                ax.plot(x_N, y,frmt, linewidth=2, alpha=0.6, label=r'$L('+str(mu)+','+str(sig2)+')$')
                if stats is True:
                    mean = np.exp(mu+0.5*sig2)
                    mode = np.exp(mu-sig2)
                    median = np.exp(mu)
                    y_mean = lognorm.pdf(mean,sig2**0.5,0,np.exp(mu))
                    y_mode = lognorm.pdf(mode,sig2**0.5,0,np.exp(mu))
                    y_median = lognorm.pdf(median,sig2**0.5,0,np.exp(mu))
                    ax.plot([mean,mean],[0,y_mean*1.15],'k--',linewidth=1)
                    ax.annotate(r'Mean: {:10.4f} '.format(mean), [mean, y_mean*1.15], 
                        xytext = [mean + 0.15,y_mean*1.15], xycoords ='data', fontsize = 25, 
                        clip_on = True)
                    ax.plot([mode,mode],[0,y_mode*1.1],'k--',linewidth=1)
                    ax.annotate(r'Mode: {:10.4f} '.format(mode), [mode, y_mode*1.1], 
                        xytext = [mode + 0.15, y_mode*1.1], xycoords ='data', fontsize = 25, 
                        clip_on = True)
                    ax.plot([median,median],[0,y_median*1.1],'k--',linewidth=1)
                    ax.annotate(r'Median: {:10.4f} '.format(median), [median, y_median*1.1], 
                        xytext = [median + 0.15, y_median*1.1], xycoords ='data', fontsize = 25, 
                        clip_on = True)

            elif dist == 'Student t':  
                nu = param[1]
                y = t.pdf(x_N,nu)
                ax.plot(x_N, y,frmt, linewidth=2, alpha=0.6, label=r'$t^{'+str(nu)+'}$')
                if stats is True:
                    ax.plot([0,0],[0,max(y)*1.1],'k--',linewidth=1)
                    ax.annotate(r'Mean, Median, Mode: {:10.4f} '.format(0), [0, max(y)*1.1], 
                        xytext = [0 + 0.15,max(y)*1.1], xycoords ='data', fontsize = 25, 
                        clip_on = True)

            y_max = max(y_max,max(y))   
        
        # Add legend and format axes to look nice
        ylim_top = y_max*1.25
        xlim_left = min(x) - x_range*0.025
        xlim_right = max(x) + x_range*0.025

        ax.legend(loc='best', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = ylim_top)
        ax.set_xlim(left = xlim_left,right = xlim_right) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xlim_right, 0, ">k", clip_on=False)
        ax.plot(xlim_left, ylim_top, "^k", clip_on=False)
        plt.tick_params(labelsize=20)

        plt.tight_layout()
    
    out = widgets.interactive_output(plot_dist, {'dist_1': dist_list_1,
                                                'dist_1_p1': dist_1_p1_box,
                                                'dist_1_p2': dist_1_p2_box,
                                                'dist_1_frmt': dist_1_formatStr,
                                                'dist_2': dist_list_2,
                                                'dist_2_p1': dist_2_p1_box,
                                                'dist_2_p2': dist_2_p2_box,
                                                'dist_2_frmt': dist_2_formatStr,
                                                'dist_3': dist_list_3,
                                                'dist_3_p1': dist_3_p1_box,
                                                'dist_3_p2': dist_3_p2_box,
                                                'dist_3_frmt': dist_3_formatStr,
                                                'dist_4': dist_list_4,
                                                'dist_4_p1': dist_4_p1_box,
                                                'dist_4_p2': dist_4_p2_box,
                                                'dist_4_frmt': dist_4_formatStr,
                                                'min_val':min_box,
                                                'max_val' :max_box,
                                                'stats': stats_check,
                                                'manual':manual_check})
    
    output = widgets.VBox([out,
                  widgets.HBox([
                    widgets.VBox([dist_list_1,
                                dist_1_p1_box,
                                dist_1_p2_box,
                                dist_1_formatStr]),
                    widgets.VBox([dist_list_2,
                                dist_2_p1_box,
                                dist_2_p2_box,
                                dist_2_formatStr]),
                    widgets.VBox([dist_list_3,
                                dist_3_p1_box,
                                dist_3_p2_box,
                                dist_3_formatStr]),
                    widgets.VBox([dist_list_4,
                                dist_4_p1_box,
                                dist_4_p2_box,
                                dist_4_formatStr]),
                    widgets.VBox([min_box,
                                max_box])
                              ]),
                  widgets.HBox([stats_check, manual_check])
                          ])
    display(output)
    
def plot_hist_widget(data_loc, class_init, x_label = 'Support',  
                     y_label = 'Density', title = 'Histogram'):

    # Declare widgets for interactive input
    class_formatStr = widgets.Text(value = class_init,
                                placeholder = '',
                                description='Classes:',
                                disabled=False,
                                continuous_update=False)
        
    # Link widgets as required
    # Not needed here
    
    def plot_hist(class_str):

        # Specify class boundaries from string
        classes_strList = class_str.split(",")
        classes = [float(i) for i in classes_strList]
        
        # load frequency data
        data = np.array(pd.read_csv(data_loc).values).flatten()
        freq = []
        for i in range(len(classes)-1):
            f_i = data[(data>= classes[i])*(data<classes[i+1])]
            freq.append(len(f_i))

        M = []
        f = []
        w = []
        index = []
        tick_lab = []
        cls_lag = classes.pop(0)
        index.append(cls_lag)
        tick_lab.append(str(cls_lag))
        for cls_curr in classes:
            i_curr = cls_curr - cls_lag
            M_curr = cls_lag + i_curr/2
            M.append(M_curr)
            w.append(i_curr)
            f.append(freq.pop(0)/i_curr)
            index.append(cls_curr)
            tick_lab.append(str(cls_curr))
            cls_lag = cls_curr

        fig, ax = plt.subplots(figsize=(20,10))
        ax.bar(M, f, width = w, color = 'b', edgecolor = 'black', alpha =0.6)

        # Add legend and format axes to look nice
        ax.set_xlabel(x_label, fontdict = {'fontsize': 25})
        ax.set_ylabel(y_label, fontdict = {'fontsize': 25})
        ax.set_title(title, fontdict = {'fontsize': 25})
        ax.set_xticks(index)
        ax.set_xticklabels(tick_lab, fontdict = {'fontsize': 20})   
        ax.tick_params(labelsize=25)

        # Set axes limts and arrows (allow for the half-integer class width effect)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(ax.get_xlim()[0]-0.5, ax.get_xlim()[1]+0.5) 
        ax.set_ylim(0, max(f)*1.25) 

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.plot(xmax, ymin, ">k", clip_on=False)
        ax.plot(xmin, ymax, "^k", clip_on=False)

        plt.tight_layout()
        
    out = widgets.interactive_output(plot_hist, {'class_str':class_formatStr})    

    output = widgets.VBox([out,
                  widgets.HBox([class_formatStr])
                          ])
    display(output)
    
def throw_dice_widget(dice_init = 1, N_init = 1):

    # Declare widgets for interactive input
    dice_list = widgets.Dropdown(options=[1,2,3],
                                value = dice_init,
                                description='Pick a dice:',
                                disabled=False)
    N_box = widgets.widgets.IntText(value = N_init,
                                description='No of throws:',
                                disabled=False)

    def throw_dice(dice, N):
        dice_list = [0,0.01,0.05]
        p_base = np.ones(6)/6

        # Set dice probabilities (fixed RNG)
        np.random.seed(0)
        p_diff = np.random.normal(0,1,6)
        p_diff = p_diff - np.mean(p_diff)
        p = p_base + dice_list[dice-1]*p_diff
        p_cdf = np.tile(np.cumsum(p)[None,:],[N,1])

        # Throw the dice N times
        np.random.seed()
        X = np.tile(np.random.uniform(0,1,N)[:,None],[6])
        draw = np.argmax(X-p_cdf<0,1)+1

        # Get frequencies for each face
        freq = np.histogram(draw, bins = 6, range = (0.5,6.5), density=True)[0]

        # Plot frequencies, with uniform for comparison
        fig, ax = plt.subplots(figsize=(20,10))
        ax.bar(np.arange(1,7), freq, color = 'b', edgecolor = 'black', alpha =0.6,
               label=r'Empirical frequency' )
        ax.plot(np.arange(1,7), p_base, 'r', linewidth=2, alpha=0.6,
                label=r'Uniform: $p = 1/6$' )

        # Add legend, titles and axis labels
        ax.legend(loc='best', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('Value', fontdict = {'fontsize': 25})
        ax.set_ylabel('Frequency/Probability', fontdict = {'fontsize': 25})
        ax.set_title(str(N) + ' throws of dice ' + str(dice), 
                     fontdict = {'fontsize': 25})

        # Set axes limts and arrows,format axes to look nice
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1]) 
        ax.set_ylim(0, max(freq)*1.5) 

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.plot(xmax, ymin, ">k", clip_on=False)
        ax.plot(xmin, ymax, "^k", clip_on=False)
        plt.tick_params(labelsize=20)

        plt.tight_layout()
    
    out = widgets.interactive_output(throw_dice, {'dice': dice_list,
                                                  'N':N_box})    
    output = widgets.VBox([out,
                          widgets.HBox([dice_list,
                                       N_box])
                          ])
    display(output)
    
def tree_widget(tree_init, popSize_init = 1000, popMode_init = False):

    # Declare widgets for interactive input - single format string here
    tree_formatStr = widgets.Text(value = tree_init,
                                placeholder = '',
                                description='Tree format:',
                                disabled=False,
                                layout = widgets.Layout(width='500px'),
                                continuous_update=False)
    popSize_Int = widgets.widgets.IntText(value = popSize_init,
                                description='Size of population:',
                                disabled=False)
    popMode_check = widgets.Checkbox(value = popMode_init,
                                   description='Pop. Mode',
                                   disabled=False,
                                   indent=True)
    
    def plot_tree(event_listStr, popSize, popMode):

        # get format
        event_list = eval(event_listStr)
        
        # Set up graph
        D = len(event_list)
        nr_vertices = 2**(D+1) - 1

        # Build node information
        Xn = []
        Yn = []
        N_labels = ['$\emptyset$']
        p_vals = [1]
        for k in range(nr_vertices):
            d = int(np.floor(np.log2(k+1)))
            a = int((2**(D+1))/(2**(d)))
            b = int((2**D)/(2**d))
            x = int(k - 2**(d) + 1)
            Xn += [d]
            Yn += [a*x+b]
            if d > 0:
                N_labels.append(event_list[d-1][x%2])
                p_event = event_list[d-1][2]
                if type(p_event) == list:
                    i = int(np.floor(x/2**(d-1)))
                    if x%2 == 0:
                        p_vals.append(p_event[i])
                    else:
                        p_vals.append(1-p_event[i])
                else:
                    if x%2 == 0:
                        p_vals.append(p_event)
                    else:
                        p_vals.append(1-p_event)

        #  Build edge information
        XedgeList = []
        YedgeList = []
        edgeLabels = []
        eventCum = [' ']
        pCum = [1]
        pad = 0.1

        for k in range(nr_vertices-1):

            parent = int(np.floor((k+2)/2))-1
            XedgeList.append([Xn[k+1]-1 + pad, Xn[k+1] - pad])
            YedgeList.append([Yn[parent], Yn[k+1]])
            if k > 1:
                p_str = N_labels[k+1] + '|' + eventCum[parent]

            else:
                p_str = N_labels[k+1]

            eventCum.append(eventCum[parent] + N_labels[k+1]) 
            pCum.append(pCum[parent]*p_vals[k+1])

            p_val = round(p_vals[k+1],5)
            if p_val == 1 or p_val == 0:
                p_val_str = '{:5.4e}'.format(p_vals[k+1])
            else:
                p_val_str = str(p_val)
            edgeLabels += ['$ P({:s}) = {:s}$'.format(p_str,p_val_str)]

        # Create figure
        fig, ax = plt.subplots(figsize=(20,10))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-0.5, D+1.5) 

        # add nodes
        for X,Y,Label in zip(Xn,Yn,N_labels):
            ax.annotate(Label, 
                [X -0.05, Y - 0.15], 
                xytext = [X - 0.05, Y-0.15],
                xycoords ='data',
                fontsize = 25,
                clip_on = True)

        # add edges, with annotations
        for Xedge, Yedge, edgeLabel in zip(XedgeList,YedgeList, edgeLabels):
            ax.plot(Xedge, Yedge, 'r', alpha = 0.6, linewidth = 2)
            Xel = 0.99*Xedge[0] + 0.01*Xedge[1]
            Yel = 0.01*Yedge[0] + 0.99*Yedge[1]
            ax.annotate(edgeLabel, 
                        [Xel , Yel], 
                        xytext = [Xel , Yel],
                        xycoords ='data',
                        fontsize = 25,
                        clip_on = True)

        # add total annotations
        if popMode:
            eventLbl = 'n'
        else:
            eventLbl = 'P'
        
        for k in range(nr_vertices-2**D, nr_vertices):
            if popMode:
                p_val_str = str(int(round(popSize*pCum[k])))
            else:
                p_val_str = str(pCum[k])
                if len(p_val_str) > 6:
                    p_val_str = '{:5.4e}'.format(pCum[k])
                
            ax.annotate(r'${:s}({:s}) = {:s}$'.format(eventLbl,eventCum[k],p_val_str), 
                        [Xn[k] , Yn[k]], 
                        xytext = [Xn[k] + 0.2 , Yn[k]],
                        xycoords ='data',
                        fontsize = 25,
                        clip_on = True)
    
        plt.tight_layout()

    out = widgets.interactive_output(plot_tree, {'event_listStr':tree_formatStr,
                                                'popSize':popSize_Int,
                                                'popMode':popMode_check}) 
    output = widgets.VBox([out,
                  widgets.HBox([tree_formatStr, popSize_Int, popMode_check])
                          ])
    display(output)
    
def birthday_widget(k_init=23):

    # Declare widgets for interactive input - single integer input here
    pick_k = widgets.widgets.IntText(value = k_init,
                                description='Size of group:',
                                disabled=False)
    
    def birthday_plot(k):
        N = float(365)
        x = np.arange(101)
        
        prob_all = ss.binom(N,x)*ss.gamma(x+1)/(N**x)
        num_k = ss.binom(N,k)*ss.gamma(k+1)
        denom_k = N**k
        prob_k = num_k/denom_k
        
        # Create figure, plot overall probabilities
        fig, ax = plt.subplots(figsize=(20,10))
        ax.step(x, prob_all, 'r', linewidth=2, alpha=0.6)
        
        # Add propbability for required k
        ax.plot([k, k], [0, prob_k], 'k--', linewidth=2, alpha=0.6)
        ax.plot([0, k], [prob_k, prob_k], 'k--', linewidth=2, alpha=0.6)

        # Add annotations
        kstr = '{:d}'.format(k)
        Nstr = '{:d}'.format(int(N))
        permStr = r'$P^{{{:s}}}_{{{:s}}} = {:5.4e}$'.format(Nstr,kstr,num_k)
        tupleStr = r'${:s}^{{{:s}}} = {:5.4e}$'.format(Nstr,kstr,denom_k)
        
        ax.annotate(r'$P =  {:5.4f}$'.format(prob_k), [0, prob_k + 0.05], 
                        xytext = [0.5,prob_k + 0.05], xycoords ='data', fontsize = 25, 
                        clip_on = True)
        ax.annotate(r'$k =  {:d}$'.format(k), [k, 0], 
                        xytext = [k + 0.5,0.05], xycoords ='data', fontsize = 25, 
                        clip_on = True)
        ax.annotate(permStr, [80, 0.9], 
                        xytext = [80,0.9], xycoords ='data', fontsize = 25, 
                        clip_on = True)
        ax.annotate(tupleStr, [80, 0.8], 
                        xytext = [80,0.8], xycoords ='data', fontsize = 25, 
                        clip_on = True)
        
        # Format axes to look nice
        ylim_top = 1.05
        xlim_left = 0
        xlim_right = 101

        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = ylim_top)
        ax.set_xlim(left = xlim_left,right = xlim_right) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xlim_right, 0, ">k", clip_on=False)
        ax.plot(xlim_left, ylim_top, "^k", clip_on=False)
        plt.tick_params(labelsize=20)

        plt.tight_layout()

    out = widgets.interactive_output(birthday_plot, {'k':pick_k})    
    output = widgets.VBox([out,
                  widgets.HBox([pick_k])
                          ])
    display(output)
    
def norm_area_widget(mu_init=0, sig2_init=1, frmt_init='r-', X_cond_init = 'X < 1.5'):
    
    mu_box = widgets.widgets.FloatText(value = mu_init,
                                description='Mean:',
                                disabled=False,
                                continuous_update=False)
    sig2_box = widgets.widgets.FloatText(value = sig2_init,
                                description='Variance:',
                                disabled=False,
                                continuous_update=False)
    formatStr = widgets.Text(value = frmt_init,
                                placeholder = '',
                                description='Format:',
                                disabled=False,
                                continuous_update=False)
    X_condStr = widgets.Text(value = X_cond_init,
                                placeholder = '',
                                description='Condition:',
                                disabled=False,
                                continuous_update=False)
    
    def plot_norm_area(mu, sig2, frmt, X_cond):

        x_min = norm.ppf(0.001,mu,sig2**0.5)
        x_max = norm.ppf(0.999,mu,sig2**0.5)

        x = np.linspace(x_min,x_max, 500)
        y = norm.pdf(x,mu,sig2**0.5)
        y_max = 1.25*max(y)

        if '>' in X_cond:
            ineq = '>'
        elif '<' in X_cond:
            ineq = '<'

        cond_list = re.split(ineq,X_cond)
        conds = []
        for foo in cond_list:
            try:
                cond = np.double(foo)
                conds.append(cond)
            except:
                pass

        if len(conds) == 1:
            if ineq == '>':
                x_1 = conds[0]
                x_2 = x_max
                p_val = 1-norm.cdf(conds[0],mu,sig2**0.5)
            elif ineq == '<':
                x_1 = x_min
                x_2 = conds[0]
                p_val = norm.cdf(conds[0],mu,sig2**0.5)

            Pstr = 'P(X' + ineq + str(conds[0]) + ')='

        elif len(conds) == 2:
            x_1 = min(conds)
            x_2 = max(conds)
            Pstr = 'P(' + str(x_1) + ineq + 'X' + ineq + str(x_2) + ')='
            p_val = norm.cdf(x_2,mu,sig2**0.5) - \
                norm.cdf(x_1,mu,sig2**0.5)

        x_cut = np.linspace(x_1,x_2, 500)
        y_cut = norm.pdf(x_cut,mu,sig2**0.5)

        x_fill = np.concatenate((x_cut,np.flip(x_cut,axis=0)))
        y_fill = np.concatenate((y_cut,0*y_cut))
        p_val_str = '{:5.4f}'.format(p_val)

        fig, ax = plt.subplots(figsize=(20,10))
        ax.fill(x_fill, y_fill, frmt[0], alpha=0.3, 
                label='$'+ Pstr + str(p_val_str) + '$')
        ax.plot(x, y,frmt, linewidth=2, alpha=0.6, 
                label=r'$N('+str(mu)+','+str(sig2)+')$')
        frmts = ['k','k--']
        for crit in conds:
            ax.plot([crit,crit], [0,y_max],frmts.pop(0), linewidth=1.5, alpha=0.6, 
                label=r'$X=' + str(crit)+'$')

        # Add legend and format axes to look nice
        ylim_top = y_max*1.05
        x_range = x_max - x_min
        xlim_left = x_min - x_range*0.025
        xlim_right = x_max + x_range*0.025
            
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = ylim_top)
        ax.set_xlim(left = xlim_left,right = xlim_right) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xlim_right, 0, ">k", clip_on=False)
        ax.plot(xlim_left, ylim_top, "^k", clip_on=False)
        plt.tick_params(labelsize=20)
        
        plt.tight_layout()

    out = widgets.interactive_output(plot_norm_area, {'mu': mu_box,
                                        'sig2': sig2_box,
                                        'frmt': formatStr,
                                        'X_cond': X_condStr})

    output = widgets.VBox([out,
                  widgets.HBox([mu_box,
                                sig2_box,
                                X_condStr,
                                formatStr])
                          ])
    display(output)
    
def norm_binom_approx_widget(n_init = 10, p_init = 0.9, show_norm_init = False):

    # Declare widgets for interactive input
    n_box = widgets.widgets.IntText(value = n_init,
                                description='$n$:',
                                disabled=False)
    p_box = widgets.widgets.FloatText(value = p_init,
                                description='$p$:',
                                disabled=False)
    show_norm_check = widgets.Checkbox(value = show_norm_init,
                                   description='Show Normal',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    # Not needed here
    
    def norm_binom_approx_plot(n, p, show_norm):
        
        x_min = binom.ppf(0.001,n,p)
        x_max = binom.ppf(0.999,n,p)        

        mu = float(n)*p
        sig2 = float(n)*p*(1-p)
        x_min = min(x_min,norm.ppf(0.001,mu,sig2**0.5))
        x_max = max(x_max,norm.ppf(0.999,mu,sig2**0.5))

        x = np.arange(np.floor(x_min),np.ceil(x_max)+1)
        x_N = np.linspace(x_min,x_max, 500)
        x_range = max(x) - min(x)

        # Create figure
        fig, ax = plt.subplots(figsize=(20,10))

        # Plot binomial
        y = binom.pmf(x, n, p)
        y_max = max(y)
        ax.plot(x, y, 'bo', ms=8, alpha=0.6,
                label=r'$B('+str(n)+','+str(p)+')$')
        ax.vlines(x, 0, y,colors = 'b', linestyles='-', alpha=0.6)

        # Calculate normal, plot it if required
        y = norm.pdf(x_N,mu,sig2**0.5)       
        y_max = max(y_max,max(y))   
        if show_norm is True:
            ax.plot(x_N, y,'r-', linewidth=2, 
                    alpha=0.6, label=r'$N('+'{:5.3f}'.format(mu) +','+'{:5.3f}'.format(sig2)+')$')

            y_max = max(y_max,max(y))   
        
        # Add legend and format axes to look nice
        ylim_bottom = 0
        ylim_top = y_max*1.25
        xlim_left = min(x) - x_range*0.025
        xlim_right = max(x) + x_range*0.025

        ax.legend(loc='best', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(bottom = ylim_bottom, top = ylim_top)
        ax.set_xlim(left = xlim_left,right = xlim_right) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xlim_right, 0, ">k", clip_on=False)
        ax.plot(xlim_left, ylim_top, "^k", clip_on=False)
        plt.tick_params(labelsize=20)

        plt.tight_layout()
    
    out = widgets.interactive_output(norm_binom_approx_plot, {'n': n_box,
                                                'p': p_box,
                                                'show_norm': show_norm_check})
    
    output = widgets.VBox([out,
                          widgets.HBox([n_box,
                                        p_box,
                                        show_norm_check])
                          ])
    display(output)
    
def clt_illustration_widget(dist_init = 'Normal', p1_init = 0, p2_init = 1, 
                            n_init = 10, N_init = 100, normalise_init = False):

    # Declare widgets for interactive input
    dist_list = widgets.Dropdown(options=['Uniform','Bernoulli','Exponential',
                                          'Normal','Lognormal'],
                                value = dist_init,
                                description='Distribution:',
                                disabled=False)
    p1_box = widgets.widgets.FloatText(value = p1_init,
                                description='Param 1:',
                                disabled=False)
    p2_box = widgets.widgets.FloatText(value = p2_init,
                                description='Param 2:',
                                disabled=False)
    n_box = widgets.widgets.IntText(value = n_init,
                                description='$n$:',
                                disabled=False)
    N_box = widgets.widgets.IntText(value = N_init,
                                description='$N$:',
                                disabled=False)
    normalise_check = widgets.Checkbox(value = normalise_init,
                                   description='Normalise',
                                   disabled=False,
                                   indent=True)
    
    # Link widgets as required
    # Not needed here
    
#     def clt_illustration(dist, p1, p2, n, N, normalise):
    def clt_illustration(dist_base, p1, p2, n, N, normalise):

        # Combine distribution choice and parameters (relic from non-widget code)
        dist = [dist_base, p1, p2]

        # Generate underlying distribution    
        if dist[0] == 'Uniform':

            a = dist[1]
            b = dist[2]
            mu = (a+b)/2
            sig2 = ((b-a+1)**2 - 1)/12
            distLabel = 'Uniform ({:.2f},{:.2f})'.format(a,b)
            x = np.arange(a,b+1)
            pmf = np.ones([len(x)])/len(x)

            # Draw sum of n samples N times
            S = np.zeros(N)
            for i in range(N):
                X = np.random.randint(a,b+1,n)
                S[i] = sum(X)

        elif dist[0] == 'Bernoulli':

            p = dist[1]

            mu = p
            sig2 = p*(1-p)
            distLabel = 'Bernoulli ({:2f})'.format(p)
            x = np.arange(0,2)
            pmf = np.asarray([1-p,p])

            # Draw sum of n samples N times
            S = np.zeros(N)
            for i in range(N):
                X = np.random.rand(n) > (1-p)
                S[i] = sum(X.astype(int))

        elif dist[0] == 'Exponential':

            L = dist[1]
            mu = 1/L
            sig2 = 1/(L**2)
            distLabel = 'Exponential ({:.2f})'.format(L)

            x = np.linspace(0,100,500)
            pmf = np.exp(-x/mu)/mu

            # Draw sum of n samples N times
            S = np.zeros(N)
            for i in range(N):
                p = np.random.uniform(0,1,n)[:,None]
                X = -np.log(1-p)*mu
                S[i] = sum(X)

        elif dist[0] == 'Normal':

            mu = dist[1]
            sig2 = dist[2]
            distLabel = 'Normal ({:.2f},{:.2f})'.format(mu,sig2)
            x = np.linspace(norm.ppf(0.001,mu,sig2**0.5),
                            norm.ppf(0.999,mu,sig2**0.5), 500)

            pmf = norm.pdf(x,mu,sig2**0.5)

            # Draw sum of n samples N times
            S = np.zeros(N)
            for i in range(N):
                p = np.random.uniform(0,1,n)[:,None]            
                X = norm.isf(p,mu,sig2**0.5)
                S[i] = sum(X)

        elif dist[0] == 'Lognormal':

            m = dist[1]
            s = dist[2]
            mu = np.exp(m + s/2)
            sig2 = (np.exp(s)-1)*np.exp(2*m+s)
            distLabel = 'Lognormal ({:.2f},{:.2f})'.format(m,s)
            x = np.linspace(0,100,500)
            pmf = lognorm.pdf(x,s**0.5,0,np.exp(m))

            # Draw sum of n samples N times
            S = np.zeros(N)
            for i in range(N):
                p = np.random.uniform(0,1,n)[:,None]
                X = lognorm.isf(p, s**0.5, 0, np.exp(m))
                S[i] = sum(X)

        plotLabel = distLabel + '\n$\mu$ = {:.2f} \n$\sigma^2$ = {:.4f}'.format(mu, sig2)

        # Get normalised distribution if requested
        if normalise:
            tag1 = r'\bar X'
            tag2 = 'average'
            muAlt = mu
            sig2Alt = sig2/n
            S /= n
        else:
            tag1 = 'S'
            tag2 = 'sum'
            muAlt = mu*n
            sig2Alt = sig2*n

        # Build histogram of S, using scott's rule for bin width
        freq = []
        classes = []
        x_min = np.floor(min(S))
        x_max = np.ceil(max(S))
        h = 3.5*((sig2Alt)**0.5)*(N**(-1/3)) 
        if normalise:                       # Bounded to avoid artefacts
            h = max([h,1/n])
        if not normalise:                   # integer to avoid artefacts  
            h = np.ceil(h)
        nbins = int(np.ceil((x_max-x_min)/h))
        x_s = np.linspace(x_min,x_max,nbins)
        for i in range(len(x_s)-1):
            f_i = S[(S>= x_s[i])*(S<x_s[i+1])]
            freq.append(len(f_i))
            classes.append(x_s[i])

        M = []
        f = []
        w = []
        cls_lag = classes.pop(0)
        for cls in classes:
            i_curr = cls - cls_lag
            M_curr = cls_lag + i_curr/2
            M.append(M_curr)
            w.append(i_curr)
            f.append(freq.pop(0)/(i_curr*N))
            cls_lag = cls

        # Get theoretical limit distribution        
        x_lim = np.linspace(x_min,x_max,500)
        y = norm.pdf(x_lim,muAlt,sig2Alt**0.5)

        # Plot underlying distribution, checking if discrete or continuous
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,8))
        if dist[0] == 'Uniform' or dist[0] == 'Bernoulli':
            ax[0].plot(x, pmf, 'ko', ms=8, alpha=0.6, label=plotLabel)
            ax[0].vlines(x, 0,pmf,colors = 'k', linestyles='-', alpha=0.6)
        else:        
            ax[0].plot(x,pmf, 'k', label = plotLabel)

        ax[0].set_xlabel(r'$X$', fontdict = {'fontsize': 20})
        ax[0].set_ylabel(r'$f(X)$', fontdict = {'fontsize': 20})
        ax[0].set_title('Plot of underlying distribution', 
                      fontdict = {'fontsize': 20})

        # Add legend and format axes to look nice
        x_min0 = min(x)
        x_max0 = max(x)
        ylim0_top = 1.5*max(pmf)
        x_range0= x_max0 - x_min0
        xlim0_left = x_min0 - x_range0*0.025
        xlim0_right = x_max0 + x_range0*0.025
        
        ax[0].legend(loc='best', frameon=False, prop={'size':20})
        ax[0].autoscale(enable=True, axis='both', tight=True)
        ax[0].set_ylim(top = ylim0_top, bottom = 0)
        ax[0].set_xlim(left = xlim0_left,right = xlim0_right) 
        
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].plot(xlim0_right, 0, ">k", clip_on=False)
        ax[0].plot(xlim0_left, ylim0_top, "^k", clip_on=False)
        ax[0].tick_params(axis='x', labelsize=20)    
        ax[0].tick_params(axis='y', labelsize=20)

        # Plot histogram and limit distribution
        ax[1].bar(M, f, width = w, edgecolor = 'black', label=r'Histogram')    
        ax[1].plot(x_lim, y,'r', linewidth=2, alpha=0.6, 
                   label=r'$N({:.3f},{:.3f})$'.format(muAlt,sig2Alt))

        ax[1].set_xlabel(r'${:s}$'.format(tag1), fontdict = {'fontsize': 20})
        ax[1].set_ylabel(r'$f({:s})$'.format(tag1), fontdict = {'fontsize': 20})
        ax[1].set_title('Histogram of {:s} from {:} samples of {:} observations'.format(tag2,N,n), 
                      fontdict = {'fontsize': 20})

        # Add legend and format axes to look nice
        ylim1_top = 1.25*max(y)
        x_range1 = x_max - x_min
        xlim1_left = x_min - x_range1*0.025
        xlim1_right = x_max + x_range1*0.025
        
        ax[1].legend(loc='best', frameon=False, prop={'size':20})
        ax[1].autoscale(enable=True, axis='both', tight=True)
        ax[1].set_ylim(top = ylim1_top, bottom = 0)
        ax[1].set_xlim(left = xlim1_left,right = xlim1_right) 

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].plot(xlim1_right, 0, ">k", clip_on=False)
        ax[1].plot(xlim1_left, ylim1_top, "^k", clip_on=False)
        ax[1].tick_params(axis='x', labelsize=20)    
        ax[1].tick_params(axis='y', labelsize=20)    

        fig.subplots_adjust(hspace=0.4, wspace=0.6)
        plt.tight_layout()
    
    out = widgets.interactive_output(clt_illustration, {'dist_base': dist_list, 
                                                        'p1': p1_box, 
                                                        'p2': p2_box,
                                                        'n': n_box,
                                                        'N': N_box,
                                                        'normalise': normalise_check})
    
    output = widgets.VBox([out,
                          widgets.HBox([dist_list,
                                        p1_box,
                                        p2_box]),
                          widgets.HBox([n_box,
                                        N_box,
                                        normalise_check])
                          ])
    display(output)
    
def bias_variance_widget(dist_init = 'Normal', p1_init = 0, p2_init = 1, 
                            n_init = 10, N_init = 100):

    # Declare widgets for interactive input
    dist_list = widgets.Dropdown(options=['Uniform','Bernoulli','Exponential',
                                          'Normal','Lognormal'],
                                value = dist_init,
                                description='Distribution:',
                                disabled=False)
    p1_box = widgets.widgets.FloatText(value = p1_init,
                                description='Param 1:',
                                disabled=False)
    p2_box = widgets.widgets.FloatText(value = p2_init,
                                description='Param 2:',
                                disabled=False)
    n_box = widgets.widgets.IntText(value = n_init,
                                description='$n$:',
                                disabled=False)
    N_box = widgets.widgets.IntText(value = N_init,
                                description='$N$:',
                                disabled=False)

    # Link widgets as required
    # Not needed here
    
    def bias_variance(dist_base, p1, p2, n, N):

        # Combine distribution choice and parameters (relic from non-widget code)
        dist = [dist_base, p1, p2]

        # Generate underlying distribution    
        if dist[0] == 'Uniform':

            a = dist[1]
            b = dist[2]
            mu = (a+b)/2
            sig2 = ((b-a+1)**2 - 1)/12
            distLabel = 'Uniform ({:.2f},{:.2f})'.format(a,b)
            x = np.arange(a,b+1)
            pmf = np.ones([len(x)])/len(x)

            # Draw sum of n samples N times
            S = np.zeros(N)
            med = np.zeros(N)
            for i in range(N):
                X = np.random.randint(a,b+1,n)
                S[i] = sum(X)
                med[i] = np.median(X) 

        elif dist[0] == 'Bernoulli':

            p = dist[1]

            mu = p
            sig2 = p*(1-p)
            distLabel = 'Bernoulli ({:2f})'.format(p)
            x = np.arange(0,2)
            pmf = np.asarray([1-p,p])

            # Draw sum of n samples N times
            S = np.zeros(N)
            med = np.zeros(N)
            for i in range(N):
                X = np.random.rand(n) > (1-p)
                S[i] = sum(X.astype(int))
                med[i] = np.median(X.astype(int)) 

        elif dist[0] == 'Exponential':

            L = dist[1]
            mu = 1/L
            sig2 = 1/(L**2)
            distLabel = 'Exponential ({:.2f})'.format(L)

            x = np.linspace(0,100,500)
            pmf = np.exp(-x/mu)/mu

            # Draw sum of n samples N times
            S = np.zeros(N)
            med = np.zeros(N)
            for i in range(N):
                p = np.random.uniform(0,1,n)[:,None]
                X = -np.log(1-p)*mu
                S[i] = sum(X)
                med[i] = np.median(X)

        elif dist[0] == 'Normal':

            mu = dist[1]
            sig2 = dist[2]
            distLabel = 'Normal ({:.2f},{:.2f})'.format(mu,sig2)
            x = np.linspace(norm.ppf(0.001,mu,sig2**0.5),
                            norm.ppf(0.999,mu,sig2**0.5), 500)

            pmf = norm.pdf(x,mu,sig2**0.5)

            # Draw sum of n samples N times
            S = np.zeros(N)
            med = np.zeros(N)
            for i in range(N):
                p = np.random.uniform(0,1,n)[:,None]            
                X = norm.isf(p,mu,sig2**0.5)
                S[i] = sum(X)
                med[i] = np.median(X)

        elif dist[0] == 'Lognormal':

            m = dist[1]
            s = dist[2]
            mu = np.exp(m + s/2)
            sig2 = (np.exp(s)-1)*np.exp(2*m+s)
            distLabel = 'Lognormal ({:.2f},{:.2f})'.format(m,s)
            x = np.linspace(0,100,500)
            pmf = lognorm.pdf(x,s**0.5,0,np.exp(m))

            # Draw sum of n samples N times
            S = np.zeros(N)
            med = np.zeros(N)
            for i in range(N):
                p = np.random.uniform(0,1,n)[:,None]
                X = lognorm.isf(p, s**0.5, 0, np.exp(m))
                S[i] = sum(X)
                med[i] = np.median(X)

        plotLabel = distLabel + '\n$\mu$ = {:.2f} \n$\sigma^2$ = {:.4f}'.format(mu,sig2)

        # Normalise distribution
        tag1 = r'\bar X'
        tag2 = 'average'
        muAlt = mu
        sig2Alt = sig2/n
        S /= n

        # Build histogram of S, using scott's rule for bin width
        freqS = []
        freqM = []
        classes = []

        x_min = np.floor(min(min(S),min(med)))
        x_max = np.ceil(max(max(S),max(med)))

        h = 3.5*((sig2Alt)**0.5)*(N**(-1/3))
        nbins = int(np.ceil((x_max-x_min)/h))
        x_s = np.linspace(x_min,x_max,nbins)
        for i in range(len(x_s)-1):
            f_i = S[(S>= x_s[i])*(S<x_s[i+1])]
            m_i = med[(med>= x_s[i])*(med<x_s[i+1])]
            freqS.append(len(f_i))
            freqM.append(len(m_i))
            classes.append(x_s[i])

        M = []
        f = []
        m = []
        w = []
        cls_lag = classes.pop(0)
        for cls in classes:
            i_curr = cls - cls_lag
            M_curr = cls_lag + i_curr/2
            M.append(M_curr)
            w.append(i_curr)
            f.append(freqS.pop(0)/(i_curr*N))
            m.append(freqM.pop(0)/(i_curr*N))
            cls_lag = cls

        # Get theoretical limit distribution        
        x_lim = np.linspace(x_min,x_max,500)
        y = norm.pdf(x_lim,muAlt,sig2Alt**0.5)

        # Plot underlying distribution
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,8))
        if dist[0] == 'Uniform' or dist[0] == 'Bernoulli':
            ax[0].plot(x, pmf, 'ko', ms=8, alpha=0.6, label=plotLabel)
            ax[0].vlines(x, 0,pmf,colors = 'k', linestyles='-', alpha=0.6)
        else:        
            ax[0].plot(x,pmf, 'k', label = plotLabel)

        ax[0].set_xlabel(r'$X$', fontdict = {'fontsize': 20})
        ax[0].set_ylabel(r'$f(X)$', fontdict = {'fontsize': 20})
        ax[0].set_title('Plot of underlying distribution', 
                      fontdict = {'fontsize': 20})

        # Add legend and format axes to look nice
        x_min0 = min(x)
        x_max0 = max(x)
        ylim0_top = 1.5*max(pmf)
        x_range0= x_max0 - x_min0
        xlim0_left = x_min0 - x_range0*0.025
        xlim0_right = x_max0 + x_range0*0.025
        
        ax[0].legend(loc='best', frameon=False, prop={'size':20})
        ax[0].autoscale(enable=True, axis='both', tight=True)
        ax[0].set_ylim(top = ylim0_top, bottom = 0)
        ax[0].set_xlim(left = xlim0_left,right = xlim0_right) 
        
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].plot(xlim0_right, 0, ">k", clip_on=False)
        ax[0].plot(xlim0_left, ylim0_top, "^k", clip_on=False)
        ax[0].tick_params(axis='x', labelsize=20)    
        ax[0].tick_params(axis='y', labelsize=20)

        # Plot histogram and limit distribution
        ax[1].bar(M, m, width = w, facecolor = 'red', edgecolor = 'black', 
              alpha=0.6, label=r'Median')    
        ax[1].bar(M, f, width = w, facecolor = 'blue', edgecolor = 'black',  
              alpha=0.3, label=r'Mean')      
        ax[1].set_xlabel(r'${:s}$'.format(tag1), fontdict = {'fontsize': 20})
        ax[1].set_ylabel(r'$f({:s})$'.format(tag1), fontdict = {'fontsize': 20})
        ax[1].set_title('Histogram of {:s} from {:} samples of {:} observations'.format(tag2,N,n), 
                      fontdict = {'fontsize': 20})

        # Add legend and format axes to look nice
        ylim1_top = 1.25*max(y)
        x_range1 = x_max - x_min
        xlim1_left = x_min - x_range1*0.025
        xlim1_right = x_max + x_range1*0.025
        
        ax[1].legend(loc='best', frameon=False, prop={'size':20})
        ax[1].autoscale(enable=True, axis='both', tight=True)
        ax[1].set_ylim(top = ylim1_top, bottom = 0)
        ax[1].set_xlim(left = xlim1_left,right = xlim1_right) 

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].plot(xlim1_right, 0, ">k", clip_on=False)
        ax[1].plot(xlim1_left, ylim1_top, "^k", clip_on=False)
        ax[1].tick_params(axis='x', labelsize=20)    
        ax[1].tick_params(axis='y', labelsize=20)    

        fig.subplots_adjust(hspace=0.4, wspace=0.6)
        plt.tight_layout()
    
    out = widgets.interactive_output(bias_variance, {'dist_base': dist_list, 
                                                        'p1': p1_box, 
                                                        'p2': p2_box,
                                                        'n': n_box,
                                                        'N': N_box})
    
    output = widgets.VBox([out,
                          widgets.HBox([dist_list,
                                        p1_box,
                                        p2_box]),
                          widgets.HBox([n_box,
                                        N_box])
                          ])
    display(output)
    
def t_area_widget(nu_init=0, frmt_init='b-', X_cond_init = 'X < 1.5'):
    
    nu_box = widgets.widgets.FloatText(value = nu_init,
                                description='D.O.F. :',
                                disabled=False,
                                continuous_update=False)
    formatStr = widgets.Text(value = frmt_init,
                                placeholder = '',
                                description='Format:',
                                disabled=False,
                                continuous_update=False)
    X_condStr = widgets.Text(value = X_cond_init,
                                placeholder = '',
                                description='Condition:',
                                disabled=False,
                                continuous_update=False)
    
    def plot_t_area(nu, frmt, X_cond):

        x_min = t.ppf(0.001,nu)
        x_max = t.ppf(0.999,nu)

        #Fix parameters for standard normal
        mu = 0
        sig2 = 1

        x = np.linspace(x_min,x_max, 500)

        y_t = t.pdf(x,nu)
        y = norm.pdf(x,mu,sig2**0.5)
        y_max = 1.25*max(y)

        if '>' in X_cond:
            ineq = '>'
        elif '<' in X_cond:
            ineq = '<'

        cond_list = re.split(ineq,X_cond)
        conds = []
        for foo in cond_list:
            try:
                cond = np.double(foo)
                conds.append(cond)
            except:
                pass

        if len(conds) == 1:
            if ineq == '>':
                x_1 = conds[0]
                x_2 = x_max
                p_val = 1-norm.cdf(conds[0],mu,sig2**0.5)
                p_val_t = 1-t.cdf(conds[0],nu)
            elif ineq == '<':
                x_1 = x_min
                x_2 = conds[0]
                p_val = norm.cdf(conds[0],mu,sig2**0.5)
                p_val_t = t.cdf(conds[0],nu)

            Pstr = 'P(X' + ineq + str(conds[0]) + ')='

        elif len(conds) == 2:
            x_1 = min(conds)
            x_2 = max(conds)
            Pstr = 'P(' + str(x_1) + ineq + 'X' + ineq + str(x_2) + ')='
            p_val = norm.cdf(x_2,mu,sig2**0.5) - \
                norm.cdf(x_1,mu,sig2**0.5)
            p_val_t = t.cdf(x_2,nu) - t.cdf(x_1,nu)

        x_cut = np.linspace(x_1,x_2, 500)
        y_cut = norm.pdf(x_cut,mu,sig2**0.5)
        y_cut_t = t.pdf(x_cut,nu)

        x_fill = np.concatenate((x_cut,np.flip(x_cut,axis=0)))
        y_fill = np.concatenate((y_cut,0*y_cut))
        y_fill_t = np.concatenate((y_cut_t,0*y_cut_t))

        p_val_str = '{:5.4f}'.format(p_val)    
        p_val_str_t = '{:5.4f}'.format(p_val_t)

        fig, ax = plt.subplots(figsize=(20,10))
        ax.fill(x_fill, y_fill_t, frmt[0], alpha=0.3, 
                label='$'+ Pstr + str(p_val_str_t) + '$')    
        ax.fill(x_fill, y_fill, 'r', alpha=0.3, 
                label='$'+ Pstr + str(p_val_str) + '$')
        ax.plot(x, y_t,frmt, linewidth=2, alpha=0.6, 
                label=r'$t^{'+str(nu)+'}$')    
        ax.plot(x, y,'r-', linewidth=2, alpha=0.6, 
                label=r'$N('+str(mu)+','+str(sig2)+')$')
        frmts = ['k','k--']
        for crit in conds:
            ax.plot([crit,crit], [0,y_max],frmts.pop(0), linewidth=1.5, alpha=0.6, 
                label=r'$X=' + str(crit)+'$')

        # Add legend and format axes to look nice
        ylim_top = y_max*1.05
        x_range = x_max - x_min
        xlim_left = x_min - x_range*0.025
        xlim_right = x_max + x_range*0.025
            
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = ylim_top)
        ax.set_xlim(left = xlim_left,right = xlim_right) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xlim_right, 0, ">k", clip_on=False)
        ax.plot(xlim_left, ylim_top, "^k", clip_on=False)
        plt.tick_params(labelsize=20)
        
        plt.tight_layout()

    out = widgets.interactive_output(plot_t_area, {'nu': nu_box,
                                        'frmt': formatStr,
                                        'X_cond': X_condStr})

    output = widgets.VBox([out,
                  widgets.HBox([nu_box,
                                X_condStr,
                                formatStr])
                          ])
    display(output)
    
def hypothesis_test_widget(n_init = 10, H0_init=0, Xbar_init=0, sig2_init=1, 
                           tails_init='~=', alpha_init = 0.95, t_init=False):
    
    n_box = widgets.widgets.IntText(value = n_init,
                                description='n : ',
                                disabled=False)
    H0_box = widgets.widgets.FloatText(value = H0_init,
                                description=r'$H_0$:',
                                disabled=False,
                                continuous_update=False)
    Xbar_box = widgets.widgets.FloatText(value = Xbar_init,
                                description=r'$\bar X$:',
                                disabled=False,
                                continuous_update=False)
    sig2_box = widgets.widgets.FloatText(value = sig2_init,
                                description=r'$s^2$:',
                                disabled=False,
                                continuous_update=False)
    tails_box = widgets.Text(value = tails_init,
                                placeholder = '',
                                description=r'$H_1$:',
                                disabled=False,
                                continuous_update=False)
    alpha_box = widgets.widgets.FloatText(value = alpha_init,
                                description=r'$\alpha$:',
                                disabled=False,
                                continuous_update=False)
    t_check = widgets.Checkbox(value = t_init,
                               description=r'Use $t$ dist.',
                               disabled=False,
                               indent=True)
    
    def hypothesis_test(n, H0, Xbar, sigma2, tails, alpha, test):

        zTest = (Xbar-H0)*(sigma2/n)**-0.5

        if '>' in tails:
            ineq = '>'
            p_crit = [alpha]
        elif '<' in tails:
            ineq = '<'
            p_crit = [1-alpha]
        else:
            ineq = '~='
            p_crit = [(1-alpha)/2, (1+alpha)/2]

        conds = []
        if n < 30 or test is True:

            # Fix parameters, generate t-distribution    
            nu = n-1
            x_min = t.ppf(0.001,nu)
            x_max = t.ppf(0.999,nu)    
            x = np.linspace(x_min,x_max, 500)
            y = t.pdf(x,nu)
            testlabel = 't'
            xlabel = r't^{'+str(nu)+'}'
            ylabel = r'$t^{'+str(nu)+'}$'

            for p_val in p_crit:
                conds.append(t.ppf(p_val,nu))

        else:

            # Fix parameters, generate standard normal
            mu = 0
            sig2 = 1
            x_min = norm.ppf(0.001,mu,sig2**0.5)
            x_max = norm.ppf(0.999,mu,sig2**0.5)
            x = np.linspace(x_min,x_max, 500)
            y = norm.pdf(x,mu,sig2**0.5)
            testlabel = 'z'
            xlabel = 'z'
            ylabel = r'$N('+str(mu)+','+str(sig2)+')$'
            for p_val in p_crit:
                conds.append(norm.ppf(p_val,mu,sig2**0.5))

        y_max = 1.25*max(y)

        if len(conds) == 1:
            if ineq == '>':
                x_1 = x_min
                x_2 = conds[0]
            elif ineq == '<':
                x_1 = conds[0]
                x_2 = x_max


        elif len(conds) == 2:
            x_1 = min(conds)
            x_2 = max(conds)


        x_cut = np.linspace(x_1,x_2, 500)
        if n < 30 or test is True:
            y_cut = t.pdf(x_cut,nu)
        else:
            y_cut = norm.pdf(x_cut,mu,sig2**0.5)

        x_fill = np.concatenate((x_cut,np.flip(x_cut,axis=0)))
        y_fill = np.concatenate((y_cut,0*y_cut))

        fig, ax = plt.subplots(figsize=(20,10))
        ax.fill(x_fill, y_fill, 'r', alpha=0.3, 
                label='Do not reject at {:.0f}%'.format(alpha*100))
        ax.plot(x, y,'r-', linewidth=2, alpha=0.6,label=ylabel)
        ax.plot([zTest,zTest], [0,y_max*0.66],'k', linewidth=2, alpha=1, 
                label=r'$'+ testlabel + '= {:.3f}$'.format(zTest))    
        frmts = ['k','k--']
        for crit in conds:
            ax.plot([crit,crit], [0,y_max*0.66],frmts.pop(0), linewidth=1.5, alpha=0.6, 
                label='$'+ xlabel + r'_{' + '{:.3f}'.format(p_crit.pop(0)) + \
                '} = '+'{:.3f}$'.format(crit))

        # Add legend and format axes to look nice
        ylim_top = y_max*1.05
        x_range = x_max - x_min
        xlim_left = x_min - x_range*0.025
        xlim_right = x_max + x_range*0.025
            
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = ylim_top)
        ax.set_xlim(left = xlim_left,right = xlim_right) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xlim_right, 0, ">k", clip_on=False)
        ax.plot(xlim_left, ylim_top, "^k", clip_on=False)
        plt.tick_params(labelsize=20)

        plt.tight_layout()

    out = widgets.interactive_output(hypothesis_test, {'n': n_box,
                                                   'H0': H0_box,
                                                   'Xbar': Xbar_box, 
                                                   'sigma2': sig2_box,
                                                   'tails': tails_box,
                                                   'alpha': alpha_box,
                                                   'test': t_check})

    output = widgets.VBox([out,
                  widgets.HBox([n_box,
                                Xbar_box,
                                sig2_box]),
                  widgets.HBox([H0_box,
                                tails_box,
                                alpha_box,
                                t_check])
                          ])
    display(output)
    
def error_type_widget(noise_init = 0.5, gap_init = 0.5, threshold_init = 0,
                      N_init = 5000):
    
    noise_box = widgets.widgets.FloatText(value = noise_init,
                                description='Noise:',
                                disabled=False)
    gap_box = widgets.widgets.FloatText(value = gap_init,
                                description='Gap: ',
                                disabled=False)
    threshold_box = widgets.widgets.FloatText(value = threshold_init,
                                description='Thres.: ',
                                disabled=False)
    N_box = widgets.widgets.IntText(value = N_init,
                                description='N : ',
                                disabled=False)
    
    def error_types(noise, gap, threshold, N):

        # fix RNG
        np.random.seed(0)

        # Generate coordinates
        pos_x = gap + np.random.rand(N)
        neg_x = -gap - np.random.rand(N)

        pos_y = pos_x + noise*np.random.randn(N)
        neg_y = neg_x + noise*np.random.randn(N)
        y_min = min(neg_y)
        y_max = max(pos_y)

        # Find true/false positives
        tp_x = pos_x[pos_y > threshold]    
        tp_y = pos_y[pos_y > threshold]
        fp_x = neg_x[neg_y > threshold]    
        fp_y = neg_y[neg_y > threshold]    

        # Find true/false negatives
        tn_x = neg_x[neg_y < threshold]    
        tn_y = neg_y[neg_y < threshold]
        fn_x = pos_x[pos_y < threshold]    
        fn_y = pos_y[pos_y < threshold]

        # Populate table data
        colNames = (r'$H_0$ true', r'$H_1$ true')
        rowNames = [r'Accept $H_0$', r'Reject $H_0$']
        freq = [['{:d}'.format(len(tn_x)),'{:d}'.format(len(fn_x))],
                ['{:d}'.format(len(fp_x)),'{:d}'.format(len(tp_x))]]
        colors = [['#8080FF','#C0C0FF'],    # blues
                  ['#FFC0C0','#FF8080']]    # reds

        # Scatter plot all four categories
        fig, ax = plt.subplots(figsize=(20,10))
        ax.scatter(tp_x, tp_y, c='#FF8080', label='True postives')
        ax.scatter(fp_x, fp_y, c='#FFC0C0', label='False positives')    
        ax.scatter(tn_x, tn_y, c='#8080FF', label='True negatives')
        ax.scatter(fn_x, fn_y, c='#C0C0FF', label='False negatives')

        ax.plot([0,0],[y_min,y_max] ,'k--', linewidth=2, alpha=0.6)
        ax.plot([-2*gap-1,2*gap+1],[0,0] ,'k--', linewidth=2, alpha=0.6)
        ax.plot([-2*gap-1,2*gap+1],[threshold,threshold] ,'g', linewidth=2, 
                alpha=0.6, label = 'Threshold: {:.2f}'.format(threshold))

        # Add table to plot
        the_table = ax.table(cellText=freq,
                              rowLabels=rowNames,
                              cellColours=colors,
                              colLabels=colNames,
                              bbox = [0.15,0.75,0.2,0.2])
        the_table.set_fontsize(20)
        
        # Add legend and format axes to look nice
        ylim_top = y_max*1.05
        ylim_bottom = y_min*1.05
        x_min = min(neg_x)
        x_max = max(pos_x)
        x_range = x_max - x_min
        xlim_left = x_min - x_range*0.025
        xlim_right = x_max + x_range*0.025
            
        ax.legend(loc='lower right', frameon=False,prop={'size':20})
        ax.set_xlabel('Real (unknown) state', fontdict = {'fontsize': 20})
        ax.set_ylabel('Measured state', fontdict = {'fontsize': 20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(bottom = ylim_bottom, top = ylim_top)
        ax.set_xlim(left = xlim_left,right = xlim_right) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xlim_right, ylim_bottom, ">k", clip_on=False)
        ax.plot(xlim_left, ylim_top, "^k", clip_on=False)
        plt.tick_params(labelsize=20)
        
        plt.tight_layout()
        
    out = widgets.interactive_output(error_types, {'noise': noise_box,
                                                'gap': gap_box,
                                                'threshold': threshold_box,
                                                'N': N_box})    
    
    output = widgets.VBox([out,
                  widgets.HBox([noise_box, gap_box, threshold_box, N_box])
                          ])
    display(output)
    
def error_tradeoff_widget(muH0_init = 0, muH1_init=3, muH2_init=None, sig2_init=1, 
                           Zcrit_init = 1.96, useH2_init=False, twotail_init=False):
    
    muH0_box = widgets.widgets.FloatText(value = muH0_init,
                                description=r'$H_0$:',
                                disabled=False,
                                continuous_update=False)
    muH1_box = widgets.widgets.FloatText(value = muH1_init,
                                description=r'$H_1$:',
                                disabled=False,
                                continuous_update=False)
    muH2_box = widgets.widgets.FloatText(value = muH2_init,
                                description=r'$H_2$:',
                                disabled=False,
                                continuous_update=False)
    sig2_box = widgets.widgets.FloatText(value = sig2_init,
                                description='Variance:',
                                disabled=False,
                                continuous_update=False)
    Zcrit_box = widgets.widgets.FloatText(value = Zcrit_init,
                                description=r'$z$:',
                                disabled=False,
                                continuous_update=False)
    useH2_check = widgets.Checkbox(value = useH2_init,
                               description=r'2 alternates?',
                               disabled=False,
                               indent=True)
    twotail_check = widgets.Checkbox(value = twotail_init,
                               description=r'2-tailed test?',
                               disabled=False,
                               indent=True)
    
    def error_tradeoff(muH0, muH1, sigma2, Zcrit, muH2, useH2, twotail):

        # Fix parameters, generate standard normal
        x_min = norm.ppf(0.001,muH0,sigma2**0.5)
        x_max = norm.ppf(0.999,muH0,sigma2**0.5)

        x_min = min([x_min,norm.ppf(0.001,muH1,sigma2**0.5)])
        x_max = max([x_max,norm.ppf(0.999,muH1,sigma2**0.5)])

        if useH2:
            x_min = min([x_min,norm.ppf(0.001,muH2,sigma2**0.5)])
            x_max = max([x_max,norm.ppf(0.999,muH2,sigma2**0.5)])

        x = np.linspace(x_min,x_max, 500)
        yH0 = norm.pdf(x,muH0,sigma2**0.5)
        yH1 = norm.pdf(x,muH1,sigma2**0.5)
        if useH2:
            yH2 = norm.pdf(x,muH2,sigma2**0.5)

        y_max = 1.25*max(yH0)

        if muH0 < muH1:
            xH0_cut1 = np.linspace(Zcrit,x_max, 500)
            xH1_cut = np.linspace(x_min,Zcrit, 500)   
            alpha1 = 1-norm.cdf(Zcrit,muH0,sigma2**0.5)
            beta1 = norm.cdf(Zcrit,muH1,sigma2**0.5)
        else:
            xH0_cut1 = np.linspace(x_min,Zcrit, 500)
            xH1_cut = np.linspace(Zcrit,x_max, 500)
            alpha1 = norm.cdf(Zcrit,muH0,sigma2**0.5)
            beta1 = 1-norm.cdf(Zcrit,muH1,sigma2**0.5)

        if useH2:
            if twotail is True:
                Zcrit2 = -Zcrit
            else:
                if muH0 < muH2:
                    Zcrit2 = x_max
                else:
                    Zcrit2 = x_min

            if muH0 < muH2:
                xH0_cut2 = np.linspace(Zcrit2,x_max, 500)
                xH2_cut = np.linspace(x_min,Zcrit2, 500)   
                alpha2 = 1-norm.cdf(Zcrit2,muH0,sigma2**0.5)
                beta2 = norm.cdf(Zcrit2,muH1,sigma2**0.5)
            else:
                xH0_cut2 = np.linspace(x_min,Zcrit2, 500)
                xH2_cut = np.linspace(Zcrit2,x_max, 500)
                alpha2 = norm.cdf(Zcrit2,muH0,sigma2**0.5)
                beta2 = 1-norm.cdf(Zcrit2,muH2,sigma2**0.5)
        else:
            alpha2 = 0

        yH0_cut1 = norm.pdf(xH0_cut1,muH0,sigma2**0.5)
        xH0_fill1 = np.concatenate((xH0_cut1,np.flip(xH0_cut1,axis=0)))
        yH0_fill1 = np.concatenate((yH0_cut1,0*yH0_cut1))

        yH1_cut = norm.pdf(xH1_cut,muH1,sigma2**0.5)
        xH1_fill = np.concatenate((xH1_cut,np.flip(xH1_cut,axis=0)))
        yH1_fill = np.concatenate((yH1_cut,0*yH1_cut))

        if useH2:
            yH0_cut2 = norm.pdf(xH0_cut2,muH0,sigma2**0.5)
            xH0_fill2 = np.concatenate((xH0_cut2,np.flip(xH0_cut2,axis=0)))
            yH0_fill2 = np.concatenate((yH0_cut2,0*yH0_cut2))

            yH2_cut = norm.pdf(xH2_cut,muH2,sigma2**0.5)
            xH2_fill = np.concatenate((xH2_cut,np.flip(xH2_cut,axis=0)))
            yH2_fill = np.concatenate((yH2_cut,0*yH2_cut))

        fig, ax = plt.subplots(figsize=(20,10))

        ax.fill(xH0_fill1, yH0_fill1, 'r', alpha=0.3, 
                label=r'Type I rate $\alpha = $ {:.2f}%'.format((alpha1+alpha2)*100))
        ax.plot(x, yH0,'r-', linewidth=2, alpha=0.6,
                label=r'Sampling distribution under $H_0$')

        ax.fill(xH1_fill, yH1_fill, 'b', alpha=0.3, 
                label=r'Type II rate $\beta = ${:.2f}%'.format(beta1*100))
        ax.plot(x, yH1,'b--', linewidth=2, alpha=0.6,
                label=r'Sampling distribution under $H_1$')

        if useH2:
            ax.fill(xH0_fill2, yH0_fill2, 'r', alpha=0.3)
            ax.fill(xH2_fill, yH2_fill, 'g', alpha=0.3, 
                label=r'Type II rate $\beta = ${:.2f}%'.format(beta2*100))
            ax.plot(x, yH2,'g--', linewidth=2, alpha=0.6,
                label=r'Sampling distribution under $H_2$')

        if twotail is False:
            tag = ''
        else:
            ax.plot([-Zcrit,-Zcrit], [0,y_max*0.66],'k', linewidth=2, alpha=1)  
            tag = r'\pm'

        ax.plot([Zcrit,Zcrit], [0,y_max*0.66],'k', linewidth=2, alpha=1, 
                label=r'$Z_{\alpha}' + ' = ' + tag + '{:.3f}$'.format(Zcrit))   

        # Add legend and format axes to look nice
        ylim_top = y_max*1.05
        x_range = x_max - x_min
        xlim_left = x_min - x_range*0.025
        xlim_right = x_max + x_range*0.025
            
        ax.legend(loc='upper left', frameon=False,prop={'size':20})
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(top = ylim_top)
        ax.set_xlim(left = xlim_left,right = xlim_right) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xlim_right, 0, ">k", clip_on=False)
        ax.plot(xlim_left, ylim_top, "^k", clip_on=False)
        plt.tick_params(labelsize=20)
    
        plt.tight_layout()

    out = widgets.interactive_output(error_tradeoff, {'muH0': muH0_box,
                                                   'muH1': muH1_box, 
                                                   'sigma2': sig2_box,
                                                   'muH2': muH2_box,
                                                   'Zcrit': Zcrit_box,
                                                   'useH2': useH2_check,
                                                   'twotail': twotail_check})

    output = widgets.VBox([out,
                  widgets.HBox([muH0_box,
                                muH1_box,
                                useH2_check,
                                muH2_box]),
                  widgets.HBox([sig2_box,
                                Zcrit_box,
                                twotail_check])
                          ])
    display(output)
    
def scatter_plot_widget(data, x_label, y_label, b0_init = 0, b1_init = 0, 
                        add_noise_init = 0, rho_init = False, regr_init=False, 
                        ols_init = False, leg_loc_init = 'upper right'):
    
    b0_box = widgets.widgets.FloatText(value = b0_init,
                                description=r'$\beta_0$: ',
                                disabled=False)
    b1_box = widgets.widgets.FloatText(value = b1_init,
                                description=r'$\beta_1$:',
                                disabled=False,
                                continuous_update=False)
    noise_box = widgets.widgets.FloatText(value = add_noise_init,
                                description=r'Extra noise:',
                                disabled=False,
                                continuous_update=False)
    rho_check = widgets.Checkbox(value = rho_init,
                                description=r'show $\rho$:',
                                disabled=False,
                                indent=True)
    regr_check = widgets.widgets.Checkbox(value = regr_init,
                                description='Show regr.:',
                                disabled=False,
                                indent=True)
    ols_check = widgets.Checkbox(value = ols_init,
                                description='Use OLS:',
                                disabled=False,
                                indent=True)
    loc_list = widgets.Dropdown(options=['upper right','lower right','lower left'],
                                value = leg_loc_init,
                                description='Legend:',
                                disabled=False)
    
    def scatter_plot(b0, b1, noise, rho, regr, ols, leg_loc):
        
       # Variables data, x_label, y_label are implicitly passed

        if type(data) == str:  # 'data' contains a path to a csv file
            load_data = np.array(pd.read_csv(data, header=None).values)
            x = load_data[:,0].flatten()
            y_base = load_data[:,1].flatten()
        elif type(data) == list:
            x = data[0]
            y_base = data[1]
            
        y = y_base + noise*np.random.randn(len(y_base))

        x_min = min(x) - (max(x) - min(x))*0.125
        x_max = max(x) + (max(x) - min(x))*0.125
        y_min = min(y) - (max(y) - min(y))*0.125
        y_max = max(y) + (max(y) - min(y))*0.125     

        fig, ax = plt.subplots(figsize=(20,10))
        ax.scatter(x, y, c='b', label='Data points')

        if regr:

            if ols:
                n = len(x)
                X = np.concatenate((np.ones(n)[:,None],
                                    np.asarray(x)[:,None]),axis = 1)
                Y = np.asarray(y)[:,None]
                gramian = np.dot(X.transpose(),X)
                moments = np.dot(X.transpose(),Y)
                betas = np.dot(np.linalg.inv(gramian),moments).flatten()
                b0 = betas[0]
                b1 = betas[1]


            ax.plot([min(x),max(x)],[b0 + b1*min(x),b0 + b1*max(x)],
                     'r--', linewidth=2, alpha=0.6, 
                     label = r'$\hat y = {:.3f} {:+.3f}x$'.format(b0,b1))

        # Add legend and format axes to look nice
        ax.legend(loc = leg_loc, frameon=True, framealpha = 0.1, 
                  facecolor = 'k', prop={'size':20})
        ax.set_xlabel(x_label, fontdict = {'fontsize': 20})
        ax.set_ylabel(y_label, fontdict = {'fontsize': 20})

        ax.autoscale(enable=True, axis='both', tight=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(left = x_min, right = x_max)
        ax.set_ylim(bottom = y_min, top = y_max)
        ax.plot(x_max, y_min, ">k", clip_on=False)
        ax.plot(x_min, y_max, "^k", clip_on=False)
        plt.tick_params(labelsize=20)
        
        if rho:
            r, pval = pearsonr(x,y)

            # Add table to plot
            text = [[r'$r = {:.3f}$'.format(r)],
                     [r'$P(\rho = 0) = {:.3f}$'.format(pval)]]
            the_table = ax.table(cellText=text,
                                  rowLabels=None,
                                  colLabels=None,
                                  bbox = [0.05,0.85,0.2,0.15])
            the_table.set_fontsize(24)
            for ind in range(len(text)):
                the_table[(ind, 0)].set_facecolor("#E5E5E5")
                the_table[(ind, 0)].set_edgecolor("#E5E5E5")                     

        plt.tight_layout()

    out = widgets.interactive_output(scatter_plot, {'b0': b0_box,
                                                    'b1': b1_box,
                                                    'noise': noise_box,
                                                    'rho': rho_check,
                                                    'regr': regr_check,
                                                    'ols': ols_check,
                                                    'leg_loc': loc_list})

    output = widgets.VBox([out,
                  widgets.HBox([noise_box, 
                                loc_list,
                                rho_check]),
                  widgets.HBox([b0_box,
                                b1_box,
                                regr_check,
                                ols_check])
                          ])
    display(output)
