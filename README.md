# Ocean-Turbulence-Structure-Functions
Calculating structure functions of several orders with experimental velocity components of the North-West Atlantic Ocean
The data used here has been collected by the Oleander Project. This project consists in a Ship of oportunity, a vessel in which they have putted a mesurement instrument based on acustic waves and the doppler effect to obtaing vertical profiles of horitzontal velocity components ( zonal and meridional ),  doing the trajectory between the ports of New Jersey and Hamilton, Bahames. The velocity components are projected onto the longitudinal and perpendicular in the direction of the ship's track by a linear adjustment of the latitud and longitud data. 
The original URL of the project is http://www.po.gso.uri.edu/rafos/research/ole/index.html; there one can find all the data i used. It is in a NETCDF4 format and each arxive has information about one transect.

The main purpose of this project is to improve the existing code i 've made for my "final degree project", some of which can be found in the file "SFTOTAL.py". In that code one enters the data and process it till one arrives at the structure functions of the velocity components. The method usually used in such works is the well known "data binning" method.
The problem here comes when one has:
  1) Non regular grid of data points ( Diferent travels imply, usually, diferent measurement locations althought within some límit ) and 
  2)Data with missing values, which is no problem if one uses the numpy_mask enviroment. 
In the existing code i solve the first problem by doing some sort of "regularization" of the data by first calculating the mean distance between all points separated by 1 index ( in the numpy array this will be the mean of all A[i]-A[i+1] ), then by 2 indexs (A[i]-A[i+2]), then by 3 and so on and second imposing, by logical operators, a mask in those values of the data that fall in between some close range to the mean value that was calculated. By doing so i lose many data points as well as i also lose the track to keep the uncertainity values. 

The specific purpose of this project is then to upgrade the code so that this "regularization" doesn't have to be done and one can bin de data directly.

**The data i'm binning is the diference of velocity between 2 points and the distance between those 2 points **

