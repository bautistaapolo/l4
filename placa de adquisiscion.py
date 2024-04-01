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

multi = visa.ResourceManager().open_resource('GPIB0::22::INSTR')

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
    return datos
    

#%%
duracion = 5 #segundos
fs = 25000 #Frecuencia de muestreo
y1, y2, Vr = medicion_una_vez1(duracion, fs)
t1 = np.arange(len(y1))/fs
np.savez('grupo5ferro.npz', varX= t1 , varY=[y1, y2, Vr])
I = 0.001 #mA
R = Vr / I
print(R)
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
plt.figure(3)
plt.plot(t1,R)
plt.grid()
plt.show()

#%%
## Medicion continua
def medicion_continua(duracion, fs):
    cant_puntos = int(duracion*fs)
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFF
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config = modo)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2", terminal_config = modo)
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

fs = 2500 #Frecuencia de muestreo
duracion = 1 #segundos
y1, y2 = medicion_continua(duracion, fs)
t1 = np.arange(len(y1))/fs
plt.plot(t1, y1)
plt.plot(t1, y2)
plt.grid()
plt.show()

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