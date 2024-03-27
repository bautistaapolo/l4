# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time

import numpy as np
import pyvisa as visa
#from instrumental import TDS1002B, AFG3021B
import matplotlib.pyplot as plt

#%%
rm = visa.ResourceManager()
rm.list_resources()

fungen =  rm.open_resource('USB0::0x0699::0x0346::C034167::INSTR')
osci =  rm.open_resource('USB0::0x0699::0x0363::C065087::INSTR')
#p=fungen.query("*IDN?")
#print(p)
xze, xin, yze, ymu, yoff = osci.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFF?;', separator=';')

osci.write('DAT:ENC RPB')
osci.write('DAT:WID 1')

data = osci.query_binary_values('CURV?', datatype='B', container=np.array)
print(data)

voltaje =(data-yoff)*ymu+yze;
tiempo = xze + np.arange(len(data)) * xin
#%%
# Vamos a generar valores de frecuencias con una separación logarítmica
# vamos de 10^1 a 10^3 , con 20 pasos
for freq in np.logspace(1, 3, 20):
    fungen.write('FREQ {:f}'.format(freq) )
    print('Comando enviado: ' + 'FREQ {:f}'.format(freq)  )
    time.sleep(0.1)  # tiempo de espera de 0.1 segundos
    #osci.write('AUTOS EXEC')
#%%    
# Rampa lineal de amplitudes
# Vamos a tener 10 pasos que van de 0 V a 1 V
for amplitude in np.linspace(0, 1, 10):
    fungen.write('VOLT {:f}'.format(amplitude) )
    print('Comando enviado: ' + 'VOLT {:f}'.format(amplitude)  )
    time.sleep(0.5)  # tiempo de espera de 0.1 segundos
    #osci.write('AUTOS EXEC')
np.savetxt('datos.txt', datos.T , delimiter=',', newline='\n', header='', footer='', comments='# ')

#%%


#fijar los límites y el paso para el barrido en frecuencias
frec = 0.75#fija el tiempo entre mediciones #float(input("Introduzca la frecuencia de muestreo en Hz: "))
frec1 = 1000.0 #float(input("Introduzca la frecuencia minima en Hz: "))
frec2 = 10000.0 #float(input("Introduzca la frecuencia maxima en Hz: "))
#paso = float(input("Introduzca el paso en Hz: "))
N=5 #int(input("Introduzca el número de mediciones: "))

datos=[]

hscale= [50.0,25.0,10.0,5.0,2.5,1.0,0.5,0.25,0.1,0.05,0.025,0.01,0.005,0.00025,0.001,0.0005,0.00025,0.0001,
         0.00005,0.000025,0.00001,5e-6,2.5e-6,1e-6,5e-7,2.5e-7,1e-7,5e-8,2.5e-8,1e-8,5e-9]#escala temporal tiempo/div
vscale=[5.0,2.0,1.0,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002]#escalas en Volt/div
#Rampa logaritmica de frecuencias
frecuencias=np.geomspace(frec1,frec2,N)
print(frecuencias)

datos=[]
escalav1=float(osci.query('CH1:SCAle?') )
osci.write('MEASU:MEAS3:SOURCE1')
# Rampa logaritmica de frequencias
frecuencias = np.geomspace(frec1, frec2, N)
for freq in frecuencias:
    fungen.write('SOURCE1:FREQ %f' % freq)
    time.sleep(1/frec)
    for x in hscale:
        if 1/freq < 2*x:
            hs=x
            print(hs)
    
    osci.write('HORizontal:MAIn:SCALE %f' %hs)
    osci.write('MEASUrement:IMMed:SOURCE CH1; TYP CRMs')
    VMRSCh1=float(osci.query( 'MEASU:IMMed:VALue?'))
    vs = None
    for y in vscale:
        if VMRSCh1 < 3*y:
            vs=y
    if vs is None:
        print("No se pudo determinar la escala y para el canal 1")
        vs=5
    time.sleep(3/frec)
    osci.write('CH1:SCAle %f'%vs)        
    osci.write('MEASUrement:IMMed:SOURCE CH1; TYPe CRMs')
    time.sleep(1/frec)	
    osci.write('MEASUrement:IMMed:SOURCE CH1; TYPe PK2pk')
    amplitudCh1=float(osci.query('MEASU:IMMed:VALue?'))
    errampCh1=0.03*amplitudCh1+escalav1*10/255
    time.sleep(1)
    frCh1Ch2Tfa= [freq, amplitudCh1]
    datos.append(frCh1Ch2Tfa)
    print(freq)
	

print(datos)

A = np.array(datos)


osci.close() #cerramos la comunicación   
fungen.close()
del(osci) #borramos el objeto
del(fungen)


#%%
nombre = input("Indique el nombre del archivo a guardar: ")

np.savetxt(str(nombre), A)

import matplotlib.pyplot as plt

plt.close("all")
fig, ax1 = plt.subplots()
ax1.plot(A[:,0], A[:,1],'.-g')
ax1.set_ylabel('Tension (V)')
ax1.set_xlabel('frecuencia (Hz)');



plt.show()





