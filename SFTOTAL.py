import glob
from netCDF4 import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy import stats

######
pps=10 # MAX ORDER OF THE STRUCTURE FUNCTION
nn=249 #MAX INDEX VALUE OF DISTANCE BETWEEN INDEXS
Numfile=360 # TOTAL NUMBER OF NETCDF4 files

#RANGE OF DEPTH'S ( NOT ALL OF THEM HAVE DATA SO WE CUT IT TO ONLY THE FIRST 450m  DEPTH )
zeta0=5
zeta1=90

clat=39*np.pi/180 # TO EXCLUDE THE CONTINENTAL SHELF, WHICH HAS ALSO POOR DATA VALUES )
def ip(lat,vlon,indlat,inh):
	if lat[indlat] > clat:
		return(True)
	else:
		return(vlon.mask[inh][indlat])

#PROJECTION OF  MERIDIONAL AND ZONAL VELOCITY COMPONENTS ONTO THE LONGITUDINAL AND PERPENDICULAR VELOCITY COMPONENTNS ( ALONG THE SHIP'S TRACK)
def projection(lat, lon, u, v,h,x):
	slope, intercept, r_value, p_value, std_err=stats.linregress(lat,lon)
	angle=np.arctan(slope)
	vlon=ma.array(data=np.zeros((len(h), len(x))), mask=np.zeros((len(h),len(x))))
	vtt=ma.array(data=np.zeros((len(h), len(x))), mask=np.zeros((len(h),len(x))))
	vlon.data[:]=[u[indexz]*(np.cos(angle)**2 -np.sin(angle))**2 +2*v[indexz]*np.sin(angle)*np.cos(angle) for indexz in range(len(h))]
	vtt.data[:]=[v[indexz]*(np.cos(angle)**2 -np.sin(angle))**2 -2*u[indexz]*np.sin(angle)*np.cos(angle) for indexz in range(len(h))]
	vlon.mask=u.mask|v.mask # TO EXPAND THE MASK WHEN INTERACTING 2 DIFERENTLY-MASKED ARRAYS
	for inh in range(len(h)):
		for indlat in range(len(x)):
			vlon.mask[inh][indlat]=ip(lat,vlon,indlat,inh)
	vtt.mask=np.copy(vlon.mask)
	return(vlon, vtt)
	
########
#EXTRACTING ALL DATA FROM FILES AND STORING IT IN MULTIPLE NUMPY MASKED-ARRAYS. THEY ARE ALSO THEN STORED IN A GENERAL ARRAY NAMED "ALL"
uls=[]
uts=[]
xs=[]
hs=[]
lats=[]
lons=[]
for filename in glob.glob('PATH_TO_FILES'+'*.nc'): # PATH TO THE FILES, IT DEPENDS ON WHERE YOU HAVE THEM STORED
	print(11)
	file=Dataset(filename,'r')
	u=file.variables['u'][:]
	v=file.variables['v'][:]
	x=file.variables['x'][:]
	h=file.variables['depth'][:]
	lat=file.variables['lat'][:]*np.pi/180
	lon=file.variables['lon'][:]*np.pi/180
	file.close()
	ul,vl=projection(lat,lon,u,v,h,x)
	uls.append(ul)
	uts.append(vl)
	xs.append(x)
	hs.append(h)
	lats.append(lat)
	lons.append(lon)

ALL=[uls,uts,xs,hs]
#np.save('PATH_TO_FILE'+'ALL',ALL)

########
#CALCULATING THE MEAN DISTANCE VALUE AND, ALSO, THE STANDARD DEVIATION 

def DX_fnj(indexfl,n): #FUNCTION FOR THE DISTANCE
    xf=ALL[2][indexfl] # THIS IS THE ARRAY OF X'S FOR EACH .NET FILE 
    DX=[np.abs(xf[j]-xf[j+n]) for j in range(len(xf)-n)] # ARRAY OF THE DIFERENCE'S BETWEEN POINTS ALONG THE INDEX VALUE n
    return(DX)

def DU_fnz_0(i,indexz,indexfl,n,p): #FUNCTION FOR THE VELOCITY ( IT DOES THE SAME AS THE ONE FOR DISTANCE BUT WITH EXTRA STEPS BECAUSE VELOCITY'S MASKED CONDITION
    ui0=ALL[i][indexfl][indexz]
    DUfnz=np.ma.array(data=np.zeros((len(ui0)-n)), mask=np.zeros((len(ui0)-n)))
    DUfnz.data[:]=[np.abs(ui0.data[j+n]-ui0.data[j])**p  for j in range(len(ui0)-n)]
    DUfnz.mask[:]=[ui0.mask[j]|ui0.mask[j+n] for j in range(len(ui0)-n)]
    return(DUfnz)

def DX_fn(n): #MEAN, FIRST, FOR ALL THE FILES WITH n fixed
    DX_fn=[np.mean(DX_fnj(indexfl,n)) for indexfl in range(Numfile)]
    return(DX_fn)
    
#NOW CHANGE n AND DO IT AGAIN... (GOES FROM 1 TO NN BECAUSE 0 IS NOT TAKEN ( IT WILL BE ALWAYS 0 )
DX=[np.mean(DX_fn(n)) for n in range(1,nn)] #ARRAY OF MEAN DISTANCE VALUES 
SDX=[np.std(DX_fn(n)) for n in range(1,nn)] #ARRAY FOR STANDARD DEVIATION


#######
# REGULARIZATION AND DATA BINNING 

I=1 # VALUE FOR THE REGULARIZATION, TAKEN RANDOMLY LOW

def MADX_nf_I(indexfl,n): #CREATING A MASK TO EXCLUDE ALL "OUT-RANGE" DATA VALUES 
    xi0=ALL[2][indexfl]
    DX01=np.ma.array(data=np.zeros((len(xi0)-n)), mask=np.zeros((len(xi0)-n), dtype=bool))
    DX01.data[:]=DX_fnj(indexfl,n)[:]
    DX01.mask[:]=(DX01.data[:]<(DX[n-1]-I*SDX[n-1])) | (DX01.data[:] > (DX[n-1]+I*SDX[n-1]))
    return(DX01.mask)

def DU_fnz_1(i,indexz,indexfl,n,p): # IMPOSING, OR COMBINING, THE MASK CREATED IN LAST FUNCTION TO THE, ALREADY EXSISTING, MASK OF VELOCITY VALUES
    DX01=MADX_nf_I(indexfl,n)
    DUfnz=DU_fnz_0(i,indexz,indexfl,n,p)  
    DUfnz.mask=DX01|DUfnz.mask
    return(DUfnz)

def DeltaU_nz(i,indexz,n,p): # MEAN OF IT FOR ALL .NET FILES
    DUnz=np.ma.array(data=np.zeros(Numfile), mask=np.zeros((Numfile)))
    DUnz.data[:]=[np.mean(DU_fnz_1(i,indexz,indexfl,n,p)) for indexfl in range(Numfile)]
    DUnz.mask=np.isnan(DUnz.data)
    return(DUnz)

def DeltaU_z(i,indexz,p): # MEAN FOR ALL n VALUES
    DUz=np.ma.array(data=np.zeros((nn-1)), mask=np.zeros((nn-1)))
    DUz.data[:]=[(np.mean(DeltaU_nz(i,indexz,n,p))) for n in range(1,nn)]#O[nn*numfile*(j-n)] o #O[nn*numfile*(j-n)^2
    DUz.mask=np.isnan(DUz.data)
    return(DUz)

#STRUCTURE FUNCTIONS FOR ALL DEPTHS ( IN THE ESTABLISHED RANG ), FOR ALL ORDERS (p from 0 to pps=10), FOR ALL THE VELOCITY COMPONENTS ( i=0 is longitudinal and i=1 is perpendicular )
So=[[[DeltaU_z(i, indexz, p) for indexz in range(zeta0, zeta1)] for p in range(0,pps)] for i in range(2)] 
#np.save('path',So) #INSERT PATH TO SAVE THE STRUCTURE FUNCTIONS

       


