# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:04:27 2024

@author: Publico
"""

# NI-DAQmx Python Documentation: https://nidaqmx-python.readthedocs.io/en/latest/index.html
# NI USB-621x User Manual: https://www.ni.com/pdf/manuals/371931f.pdf
import matplotlib.pyplot as plt
import numpy as np
import nidaqmx
import math
import time
import pyvisa as visa

#%%
#para saber el ID de la placa conectada (DevX)
system = nidaqmx.system.System.local()
for device in system.devices:
    print(device)

rm= visa.ResourceManager()

print(rm.list_resources())

#multi = visa.ResourceManager().open_resource('GPIB0::22::INSTR')

#para setear (y preguntar) el modo y rango de un canal analógico
with nidaqmx.Task() as task:
    ai1_channel = task.ai_channels.add_ai_voltage_chan("Dev1/ai1",max_val=10,min_val=-10)
    ai2_channel = task.ai_channels.add_ai_voltage_chan("Dev1/ai2",max_val=10,min_val=-10)
    ai4_channel = task.ai_channels.add_ai_voltage_chan("Dev1/ai4",max_val=10,min_val=-10)
    #v1 = 
    # print(ai1_channel.ai_term_cfg, ai2_channel.ai_term_cfg)    
    # print(ai1_channel.ai_max, ai2_channel.ai_max)
    # print(ai1_channel.ai_min, ai2_channel.ai_min)	
	
#%%
## Medicion por tiempo/samples de una sola vez
def medicion_una_vez1(duracion, fs):
    cant_puntos = int(duracion*fs)
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFF
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config = modo,max_val=10,min_val=-10)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2", terminal_config = modo,max_val=10,min_val=-10)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai4", terminal_config = modo,max_val=10,min_val=-10)
        
               
        task.timing.cfg_samp_clk_timing(fs,samps_per_chan = cant_puntos,
                                        sample_mode = nidaqmx.constants.AcquisitionType.FINITE)
        
        datos1 = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE, timeout=duracion+0.1)           
    
    datos = np.asarray(datos1)
    temp = PT100_res2temp_interp(datos[2,:]/1e-3)
    return datos,temp

#%%
def PT100_res2temp_interp(R): #en Ohm
    data = np.loadtxt('C:/Users/publico/Desktop/Grupo Miércoles/Pt100_resistencia_temperatura.csv',delimiter=',') 
    temperature_vals = data[:,0] # en Celsius
    resistance_vals = data[:,1] #en Ohm
    return np.interp(R, resistance_vals, temperature_vals)




#%%
duracion = 1/25 * 25 *60 *3
fs = 5000 #Frecuencia de muestreo   
[y1, y2, vr], temp = medicion_una_vez1(duracion, fs)
t1 = np.arange(len(y1))/fs
np.savez('datos_nitrogeno2.npz', tiempo = t1 , tensiones = [y1, y2], R = vr, temp=temp)
I = 0.001 #mA

#%%
import numpy as np
placa = np.load('datos_nitrogeno2.npz',allow_pickle=True)

Vin, Vout = placa['tensiones']
R = placa['R']
t = placa['tiempo']
temp = placa['temp']


from scipy.signal import savgol_filter as savgol


temp_suave = savgol(temp,151,3)


plt.figure(1)
plt.plot(t, temp, '.')
plt.plot(t, temp_suave, '-')
plt.show()
plt.figure(2)
plt.plot(Vin, Vout)
plt.show()

#%%

for jj in range(0,2000*10,2000):
    plt.plot(Vin[jj+0:jj+200],Vout[jj+0:jj+200],'.')
    plt.gca().axhline(0,ls='--',color='gray')
    plt.gca().axvline(0,ls='--',color='gray')
#%%
for j in range(0, 20000):    
    if Vin[j]*Vin[j+1]<0:
          print(Vin[j])
          print(Vout[j])
          print(j)
#%%

plt.plot(Vin[0:200],Vout[0:200])
plt.plot(Vin[0:200],'.-')
plt.plot(Vout[0:200],'.-')
plt.gca().axhline(0,ls='--',color='gray')

#%%

plt.figure(1)
plt.plot(t1,y1)
plt.grid()
plt.show()

plt.plot(t1,y2)
plt.grid()
plt.show()
plt.figure(2)
plt.plot(y1,y2)
plt.grid()
plt.show()


#%%
## Medicion continua
def medicion_continua(duracion, fs):
    cant_puntos = int(duracion*fs)
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFF
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config = modo,max_val=10,min_val=-10)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2", terminal_config = modo,max_val=10,min_val=-10)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai4", terminal_config = modo,max_val=10,min_val=-10)
        task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
        task.start()
        t0 = time.time()
        total = 0
        data =[]
        while total<cant_puntos:
            time.sleep(0.1)
            datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)           
            data.extend(datos)
            total = total + len(datos)
            t1 = time.time()
            print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos), total, total/(t1-t0)))            
        return data
fs1 = 250 #Frecuencia de muestreo
duracion1 = 0.5 #segundos
vin, vout, vr = medicion_continua(duracion1, fs1)


#%%
## Modo conteo de flancos 
# Obtengo el nombre de la primera placa y el primer canal de conteo (ci)
cDaq = system.devices[0].name
ci_chan1 = system.devices[0].ci_physical_chans[0].name
print(cDaq)
print(ci_chan1 )

# Pinout: 
# NiUSB6212 
# gnd: 5 or 37
# src: 33


def daq_conteo(duracion):
    with nidaqmx.Task() as task:

        # Configuro la task para edge counting
        task.ci_channels.add_ci_count_edges_chan(counter=ci_chan1,
            name_to_assign_to_channel="",
            edge=nidaqmx.constants.Edge.RISING,
            initial_count=0)
        
        # arranco la task
        task.start()
        counts = [0]
        t0 = time.time()
        try:
            while time.time() - t0 < duracion:
                count = task.ci_channels[0].ci_count
                print(f"{time.time()-t0:.2f}s {count-counts[-1]} {count}")
                counts.append(count)
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            pass
        
        finally:
            task.stop()
            
    return counts  

duracion = 1 # segundos
y = daq_conteo(duracion)
t = np.arange(len(y))/fs
plt.plot(t, y)
plt.grid()
plt.show()