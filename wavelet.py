"""
Continuous Wavelet Transform tool
31 Oct 2017

A continuous wavelet is a well-known fundamental tool that allows to filter data-sets such as to enhance localised features of a given shape (or periodicity) for a given scale, whilst diminishing features with scales far removed.
The module contains two Continuous Wavelet Transform (CWT) functions, a one-dimensional CWT wavelet 1d and a two-dimensional CWT wavelet 2d. Each has associated mother wavelet functions. Additionally, wavelet 1d has a plot function. There are two demo functions added to see the CWTs in action.

Prerequisites
This module has been designed for Python 2.7 and uses external standard Python modules: numpy, scipy, matplotlib

Using this module in publications
When using this module is published work, please cite the following papers: 
For the 1d CWT:
- Verwichte, E., Nakariakov, V. M., Ofman, L., & Deluca, E. E. 2004, Sol. Phys., 223, 77
- Torrence, C. & Compo, G. P.: 1998, Bull. Amer. Meteor. Soc. 79, 61.
For the 2d CWT:
- White, R. S., Verwichte, E., & Foullon, C. 2012, AA, 545, A129
- Witkin, A. P. 1983, in Proc. Int. Joint Conf. Artifcial Intell., IJCAI’83, San Francisco, CA, USA, 1019

Also, please add to the Acknowledgments:
The wavelet transform has been performed using the Python wavelet module by 
Erwin Verwichte (University of Warwick) and was supported by a UK STFC grant ST/L006324/1.

This software falls under Licences:
- GNU AGPLv3 license
- CRAPL license
"""

__authors__ = ["Erwin Verwichte"]
__email__ = "erwin.verwichte@warwick.ac.uk"

import numpy as np
import scipy.interpolate
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import gridspec

__all__ = ['wavelet_1d','wavelet_2d','plot1','wvl_func_morlet1','wvl_func_mexicanhat2','wvl_func_gaussian2']

class cwt1d_class(object):
	def __init__(self,x,y):
		self.x0 = x
		self.y0 = y

def wavelet_1d(*args,**kwargs):
	"""
	Description:
		One-dimensional Continuous Wavelet Transform of a time-series
	Input:
		x: 1d array of independent variable
		y: 1d array of dependent variable, same dimensions as y
		a0,a1,a2: 1d arrays of scale-space parameters
	Output:
		C: object of class cwt1d_class, containing all info on the transform
			C.x0: original independent variable
			C.y0: original dependent variable
			C.x: independent variable used in transform
			C.y: dependent variable used in transform
			C.dxx: interval in C.x
			C.cwt: complex CWT
			C.cwt_real: real part of CWT
			C.cwt_imag: imaginary part of CWT
			C.cwt_abs: absolute part of CWT
			C.scale: array of scales, either bases on input or
				Choice of scales s = s0 2^(j dj) with j=0..ns-1, ns = log2(Ndt/s0) / dj
				dj is chosen such that sigma of reciprocal wavelet is larger than scale sampling difference		
			C.period: array of period scales
			C.coi: cone of influence
			C.signif: array of significance value as a function of scale
			C.siglvl: significance level
			C.array_missing: array indicating if data is missing
			C.fft_theor:
			C.time_max: array of x-values of maxima in CWT
			C.period_max: array of y-values of maxima in CWT
			C.count_max: number of maxima present
			C.mother: string name of mother wavelet used
			C.fit: order of polynomial fit (if set)
			C.yfit: array of polynomial fit		
	Keywords:
		fit: if given will subtract a polynomial of order fit from y (default = average substracted)
		kk: 1d array of reciprocal space variable as input (default = None)
		lag1: (default = 0)
		missing: Boolean stating whether data is missing (default = False)
		mother: string of name of mother wavelet: 'morlet' (default = 'morlet')
		regular: Boolean stating whether the intervals in x are regular (default = True)
			if not regular the data will be interpolated onto a regular grid
		resolv: Scalar of resolution of scales (default = 1.0)
		siglvl: Scalar of significance level between 0 and 1 (defaul = 0.99)
		variance: Scalar of variance to use in determining significance (default = variance(y))
			Note that if fit is given, the default variance is that of the residual y where the polynomial has been substracted 
	"""
	
	# input
	x,y = args[0:2]
	x,y = np.array(x),np.array(y)
	
	# init
	dof_min = 1
	C = cwt1d_class(x,y)
	
	# Data resampling
	regular = kwargs.pop('regular',False)
	if not regular:
		sample = kwargs.pop('sample',1)
		maxm = kwargs.pop('maxm',1e5)
		if 'dx' in kwargs.keys():
			dx0 = kwargs['dx']
		else:
			dx0 = x - np.roll(x,1)
			dx0 = dx0[1:].min()
		m = np.ceil((x.max() - x.min()) / dx0) * sample
		if m > maxm:
			print 'Too many points, ' + str(maxm) + ' taken!'
			m = maxm * 1
		xx = x.min() + np.arange(m)/(m-1.) * (x.max() - x.min())
		nx = len(xx)
		tck = scipy.interpolate.splrep(x,y,s=0,k=3)
	 	yy = scipy.interpolate.splev(xx,tck,der=0,ext=0)	
	else:
		xx = x * 1.
		nx = len(xx)
		yy = y * 1.
	dxx = xx[1] - xx[0]
		
	# missing
	array_missing = np.repeat(1,nx)
	missing = kwargs.pop('missing',False)
	if missing:
		delta_x = xx - np.roll(xx)
		delta_x[0] = 0.
		ss = np.where(delta_x > (missing*dxx))[0]
		count = len(ss)
		if count > 0:
			for s in ss:
				tt = np.where((xx > xx[s-1]) & (xx < xx[s]))[0]
				if len(tt) > 0: array_missing[tt] = 0
		
	time = xx * 1.
	xx = xx - xx.min()
	
	# subtract polynomial fit or average value
	if 'fit' in kwargs.keys():
		fit = kwargs['fit']
		c = np.polyfit(xx,yy,fit)
		yy = yy - np.polyval(c,xx)
		C.fit = fit; C.yfit = np.polycal(c,xx)
	else:
		yy = yy - np.sum(yy) / (nx * 1.)
		
	# amount of wavelet parameters
	n_scale = len(args) - 2
	m_scale = []
	d_scale = []
	for a in args[2:]:
		dims = np.shape(np.array(a))
		d_scale.append(len(dims))
		if len(dims) == 0:
			m_scale.append(1)
		else:
			m_scale.append(len(a))
	nr_scale = len(m_scale)

	# Choice of scales s = s0 2^(j dj) with j=0..ns-1, ns = log2(Ndt/s0) / dj
	# dj is chosen such that sigma of reciprocal wavelet is larger than scale sampling difference
		
	if len(args) == 2:
		resolv = kwargs.pop('resolv',1.0)
		s0 = 2. * dxx
		dj = 0.125 / resolv
		ns = int((np.log(nx * dxx / s0) / np.log(2.)) / dj) + 1
		scale = s0 * 2.**(np.arange(ns) * dj)
		args = args + (scale,)
		n_scale = 1
		m_scale = [ns]
		d_scale = [1]
		nr_scale = 1
	
	# checking whether reciprocal coordinates are set or not
	knew = True
	if 'kk' in kwargs.keys():
		kk = kwargs['kk']
		if len(kk) == nx: knew = False	
	#knew = False
	#if 'kk' in kwargs.keys():
	#	kk = kwargs['kk']
	#	if len(kk) != nx: knew = True
	#else:
	#	knew = True
		
	# reciprocal space
	of = int(nx/2.) if (nx % 2) == 0 else int((nx-1)/2.)		
	yy_k = np.roll(np.fft.fft(yy),of)
	kx = np.roll(np.fft.fftfreq(nx) * 2. * np.pi,of)			
	#yy_k,kx = evsp.fft(yy)
	if knew:
		kk = kx * 1.
		kk = kk / (dxx * 1.)
						
	# wavelet choice
	mother = kwargs.pop('mother','morlet')
	mother = 'wvl_func_' + mother + '1'
	
	# loop through scales
	n_ss = 1
	for ms in m_scale: n_ss = n_ss * ms
	ss = np.arange(n_ss)
	index = where2indices(ss,m_scale)
	
	txt = 'cwt = np.zeros((nx,'
	for j in np.arange(nr_scale): txt = txt + 'm_scale[j],'
	txt = txt[:-1] + '),dtype=complex)'
	exec txt

	# wavelet transform
	lag1 = kwargs.pop('lag1',0.)
	fft_theor_k = (1.-lag1**2) / (1. - 2.*lag1*np.cos(kk*dxx) + lag1**2)

	txt = ' = np.zeros('
	for j in np.arange(nr_scale): txt = txt + 'm_scale[j],'
	txt = txt[:-1] + ')'
	exec 'fft_theor ' + txt
	exec 'period ' + txt
	
	for i in ss:
		parms = []
		parms.append(kk)
		txt = ''
		for j in np.arange(nr_scale): 
			a = args[j+2]
			if d_scale[j] == 0: a = [a]
			parms.append(a[index[j][i]])
			txt = txt + str(index[j][i]) + ','
		txt = txt[:-1] + ']'
		exec 'mother_i,p,coi,dofmin = ' + mother + '(*parms,**kwargs)' 
		#cwt_i,xxx = evsp.fft(mother_i * yy_k,inverse=True)
		cwt_i = np.fft.ifft(np.roll(mother_i * yy_k,-of))		
		exec 'cwt[:,' + txt + ' = cwt_i'
		exec 'period[' + txt + ' = p'
		v = np.sum(np.abs(mother_i)**2 * fft_theor_k) / (nx * 1.)
		exec 'fft_theor[' + txt + ' = v'
		
	# Cone of influence
	if (nx % 2) == 0:
		n2 = (nx-2)/2
		coiv = xx[0:n2+1]
		coiv = np.append(coiv,(xx[0:n2+1])[::-1])
	else:
		n2 = (nx-1)/2
		coiv = xx[0:n2+1]
		coiv = np.append(coiv,(xx[0:n2])[::-1])
	coi = coi * coiv
		
	# Significance levels
	siglvl = kwargs.pop('siglvl',0.95)
	variance = kwargs.pop('variance',np.var(yy))
	fft_theor = variance * fft_theor
	dof = dofmin
	#signif = fft_theor * chisqr_cvf(1.0-siglvl,dof) / (dof * 1.)
	signif = fft_theor * scipy.stats.chi2.isf(1.0-siglvl,dof) / (dof * 1.)
	
	# maxima in wavelet power
	f = np.abs(cwt**2) / signif
	dimsf = np.shape(f)
	df1 = f * 1.
	for j in np.arange(dimsf[0]): df1[j,:] = np.roll(f[j,:],1)						
	df2 = f * 1.
	for j in np.arange(dimsf[0]): df2[j,:] = np.roll(f[j,:],-1)						
	ss = np.where(((f - df1) > 0) & ((f- df2) > 0) & (f >= 1))
	time_max = time[ss[0]]
	period_max = period[ss[1]]
	coi_max = coi[ss[0]]
	tt = np.where(period_max < coi_max)[0]
	count_max = len(tt)
	if count_max > 0:
		time_max = time_max[tt]
		period_max = period_max[tt]
	else:
		time_max = -1
		period_max = -1

	# Output into class
	C.y = yy; C.dxx = dxx; C.cwt = cwt; C.x = time
	C.cwt_real = cwt.real; C.cwt_imag = cwt.imag; C.cwt_abs = np.abs(cwt)
	C.scale = scale; C.period = period; C.coi = coi
	C.signif = signif; C.siglvl = siglvl
	C.array_missing = array_missing; C.fft_theor = fft_theor
	C.siglvl = siglvl; C.time_max = time_max
	C.period_max = period_max; C.count_max = count_max
	C.mother = mother
	
	return C

def wvl_func_morlet1(*args,**kwargs):
	"""
	1d Morlet CWT
	Input:
		kx: reciprocal space array
		a: Scalar of scale
	Output:
		wavelet: FFT of mother wavelet for given scale as a function of kx
		period: Scalar of period associated with a
		coi: cone of influence
		dofmin: degree of freedom
	Keywords:
		norm: Scalar of norm used (default = 1)
		epsilon: Scalar (default = 1.)
		k0: Scalar (default = 5.6)
	"""	
	# Arguments
	kx,a = args
	# Keywords
	norm = kwargs.pop('norm',1)
	epsilon = kwargs.pop('epsilon',1.)
	k0 = kwargs.pop('k0',5.6)
	
	dims = np.shape(a)
	if len(dims) > 0: a = a[0]
	n = len(kx)
	dt = 2 * np.pi / (n * (kx[1]-kx[0]))	

	fourier_factor = 4 * np.pi / (k0 + np.sqrt(2. + (k0*epsilon)**2)/epsilon)
	period = fourier_factor * a
	coi = fourier_factor / np.sqrt(2.0)
	dofmin = 2.

	wavelet = np.sqrt(2.0*np.pi*epsilon/(dt*np.sqrt(np.pi))) * np.sqrt(a) * np.exp(-0.5 * epsilon**2 * (k0 - kx * a) * (k0 - kx * a))
	wavelet = wavelet * ((kx > 0)*1.)
	return wavelet,period,coi,dofmin

def plot1(C,**kwargs):
	"""
	Routine to plot the result of the 1D wavelet transform,for typically Morlet transform,
	which includes x-scale-space, the original time series and a power spectrum.
	Input:
		C: cwt1d_class type class object
	Keywords:
		CWT:
			real: Boolean, plot real part of CWT (default = False)
			imaginary: Boolean, plot imaginary part of CWT (default = False)
			absolute: Boolean, plot absolute part of CWT (default = True)
			zrange: 2-element array of min-max of CWT to plot (default = [min,max])
		Scales:
			freq: Boolean, plot versus frequency = 2 pi / scale (default = False)
			yrange: 2-element array of scale-range (default = [min,max])
			ylog: Boolean, plot logarithmic scale (defaull = True)
		COI:
			coi: Boolean, whether to plot COI (default = True)
		Colors:
			theme: theme of color scheme: 'blue','red','' (default = '')
			cmap: colormap used for x-scale plot (default given by theme)
			color: color of lines (default given by theme)
			coi_color: color of COI hatches (default givem by theme)
			background_color: color of background in time series and power spectrum plots (default given by theme)
		Titles:
			title: title of x-scale plot (default = 'CWT')
			title_scale: title of y-label in x-scale plot (default = 'period' or 'frequency')
			title_time: title of x-label of time series plot (default = 'time)
			title_signal: title of y-label of time series plot (default = 'signal) 
	Examples:
		plot1(C,ylog=False,yrange=[0,4])	
		plot1(C,freq=False,theme='Blue')			
	"""

	# Titles
	title = kwargs.pop('title','CWT')
	title_signal = kwargs.pop('title_signal','signal')
	title_time = kwargs.pop('title_time','time')
	
	theme = kwargs.pop('theme','Blue').lower()
	if theme == 'blue':
		cmap = 'Blues'
		color = 'Blue'
		coi_color = 'Red'
		background_color = '#F5F5FF'
	elif theme == 'red':
		cmap = 'Reds'
		color = '#AA0000'
		coi_color = 'Black'
		background_color = '#FFF5F5'
	else:
		cmap = 'Greys'
		color = '#000000'
		coi_color = '#000000'
		background_color = '#FFFFFF'
	#background_color = hex2color(background_color)
	
	if 'coi_color' in kwargs.keys(): coi_color = kwargs['coi_color']
	if 'cmap' in kwargs.keys(): cmap = kwargs['cmap']
	
	# Plot real, imaginary or absolute value
	real = kwargs.pop('real',False)
	imaginary = kwargs.pop('imaginary',False)
	absolute = kwargs.pop('absolute',True)	
	if real:
		cwt = C.cwt_real
	elif imaginary:
		cwt = C.cwt_imag
	else:
		cwt = C.cwt_abs
	
	# Create plotting environment
	multi = gridspec.GridSpec(2,2,
		left=0.125, bottom=0.1, right=0.9, top=0.9, 
		wspace=0.1, hspace=0.1,
		width_ratios=[3,1],height_ratios=[3,1])
	
		
	# Main plot
	plt.subplot(multi[0])
	# x
	x = C.x
	nx = len(x)
	# y
	freq = kwargs.pop('freq',False)
	y = 2.*np.pi/C.period if freq else C.period
	yr = np.array([y.min(),y.max()])
	yrange = kwargs.pop('yrange',yr)
	ystyle = kwargs.pop('ystyle',1)
	if 'title_scale' in kwargs.keys():
		title_scale = kwargs['title_scale']
	else:
		title_scale = 'frequency' if freq else 'period'	
	# contours
	if 'zrange' in kwargs.keys():
		vvmin,vvmax = kwargs['zrange']
	else:
		vvmin,vvmax = np.min(cwt),np.max(cwt)
	
	# Filled contours of CWT
	ax = plt.gca()
	ax.axes.xaxis.set_ticklabels([])
	if kwargs.pop('ylog',True): plt.yscale('log')
	plt.ylim(yrange[0],yrange[1])
	plt.title(title)
	plt.ylabel(title_scale,fontsize=11)
	Y,X = np.meshgrid(y,x)
	CS = plt.contourf(X,Y,cwt,kwargs.pop('nlevels',40),vmin=vvmin,vmax=vvmax,cmap=cmap,origin='lower')			
	
	# Significance level
	signif = C.signif; siglvl = C.siglvl
	ns = len(signif)
	sgn = np.zeros((nx,ns),dtype='float')
	for i in np.arange(nx): sgn[i:] = signif
	CS2 = plt.contour(X,Y,np.abs(cwt*cwt)/sgn,[1.0],colors='Black',origin='lower')
	plt.clabel(CS2,CS2.levels,inline=1,fontsize=11,fmt=str(siglvl),linestyle='-',linewidth=1)	

	# COI
	coi = kwargs.pop('coi',True)	
	if coi:
		coi = C.coi
		ss = np.where(coi >= yr[0])[0]
		count = len(ss)
		if count > 0:
			if freq:
				plt.plot(x,2*np.pi/coi,linestyle='-',color=coi_color)
				#evp.plot(x,2.*np.pi/coi,line=0,overplot=True,color='Red')
				coi_x = x #[ss]	
				coi_y = 2.0*np.pi / coi #[ss]
				coi_y_tip = coi_y.min()
				if x[ss[-1]] < x.max():
					arr = np.array([x.max(),x.max(),x.min(),x.min()])
					for a in arr: coi_x = np.append(coi_x,a)
					arr = np.array([y.max(),y.min(),y.min(),y.max()])
					for a in arr: coi_y = np.append(coi_y,a)
				else:
					arr = np.array([x.max(),x.min()])
					for a in arr: coi_x = np.append(coi_x,a)
					arr = np.array([y.min(),y.min()])
					for a in arr: coi_y = np.append(coi_y,a)
			else:
				plt.plot(x,coi,linestyle='-',color=coi_color)
				coi_x = x[1:-1] #[ss]	
				coi_y = coi[1:-1] #[ss]
				coi_y_tip = coi_y.max()
				if x[ss[-1]] < x.max():
					arr = np.array([x.max(),x.max(),x.min(),x.min()])
					for a in arr: coi_x = np.append(coi_x,a)
					arr = np.array([y.min(),y.max(),y.max(),y.min()])
					for a in arr: coi_y = np.append(coi_y,a)
				else:
					arr = np.array([x.max(),x.min()])
					for a in arr: coi_x = np.append(coi_x,a)
					arr = np.array([y.max(),y.max()])
					for a in arr: coi_y = np.append(coi_y,a)
			
			plt.fill(coi_x,coi_y,coi_color,hatch='\\',fill=False,color=coi_color)	
			plt.fill(coi_x,coi_y,coi_color,hatch='/',fill=False,color=coi_color)	
							
	# maxima
	if C.count_max > 0:
		dt = C.x[1] - C.x[0]
		if freq:
			for i in np.arange(C.count_max):
				plt.plot(C.time_max[i] + 0.5*dt*np.array([-1,1]),2.*np.pi/C.period_max[i]*np.array([1,1]),linewidth=1,color=color)
		else:
			for i in np.arange(C.count_max):
				plt.plot(C.time_max[i] + 0.5*dt*np.array([-1,1]),C.period_max[i]*np.array([1,1]),linewidth=1,color=color)


	# Time series plot
	try:
		plt.subplot(multi[2],facecolor=background_color)
	except:	
		plt.subplot(multi[2],axisbg=background_color)
	f = C.y
	#ax.set_axis_bgcolor(background_color)
	plt.xlabel(title_time,fontsize=11)
	plt.ylabel(title_signal,fontsize=11)
	plt.ylim(f.min() - 0.1*(f.max()-f.min()),f.max() + 0.1*(f.max()-f.min()))
	plt.plot(x,f,color=color,linewidth=1.5)
	
	# Spectrum plot
	try:
		plt.subplot(multi[1],facecolor=background_color)
	except:	
		plt.subplot(multi[1],axisbg=background_color)
	g = np.sum(cwt,axis=0) / (nx * 1.)
	g = g**2
	gmax = g.max()
	g = g / gmax
	gs = signif / gmax

	g_x = g
	g_x = np.append(g_x,0)
	g_x = np.append(g_x,0)
	g_y = y
	if freq:
		g_y = np.append(g_y,y.min()) 
		g_y = np.append(g_y,y.max())
	else:
		g_y = np.append(g_y,y.max())
		g_y = np.append(g_y,y.min()) 

	ax = plt.gca()
	plt.xlabel('power',fontsize=11)
	dg = (max(g)-min(g))
	dg_tiny = 1e-10 * dg
	dg = dg / 2.
	plt.xticks(np.arange(min(g)-dg_tiny, max(g)+dg_tiny, dg))
	ax.set_xticklabels(['0','0.5','0.1'])	
	plt.xlim(0,1)
	
	plt.ylim(yrange[0],yrange[1])
	ax.axes.yaxis.set_ticklabels([])
	ax.tick_params(labelleft='off',labelright='on')
	if kwargs.pop('ylog',True): plt.yscale('log')
	
	#ax.set_axis_bgcolor(background_color)
	plt.plot(g,y,linewidth=1,color=color)
	plt.fill(g_x,g_y,color=color,alpha=0.5)
	ss = np.where(y >= coi_y_tip)[0] if freq else np.where(y <= coi_y_tip)[0]
	plt.plot(gs[ss],y[ss],linestyle='--',color='Black',linewidth=1.5)
	u = np.array([0,1,1,0])
	vv = 1 - freq*1
	v = np.array([coi_y_tip,coi_y_tip,yr[vv],yr[vv]])
	plt.fill(u,v,hatch='\\',color=coi_color,fill=False)
	plt.fill(u,v,hatch='/',color=coi_color,fill=False)	
	
	return





def wavelet_2d(*args,**kwargs):
	"""
	Description:
		Two-dimensional Continuous Wavelet Transform (CWT) of an image
	Input:
		img: 2d-array of image
		scale1: 1d array of parameter values related to first parameter taken by mother wavelet
		[scale2: 1d array of parameter values related to second parameter taken by mother wavelet]
		[...]
	Output:
		cwt: 2d or 3d array of CWT
	Keywords:
		absolute: boolean if set outputs absolute part of CWT (default: False)
		edge: scalar of introducing extra edge pixels (default: 0)
		imaginary: boolean if set outputs imaginary part of CWT (default: False)
		kk: 2d array of reciprocal space variable as input (default = None)
		mother: string of name of mother wavelet: 'mexicanhat' (default = 'mexicanhat')
		norm: scalar of norm used in transform, i.e. 1, 2 or np.inf (default: 1)
		real: boolean if set outputs real part of CWT (default: False)
	Dependencies:
	History:
		Created by Erwin Verwichte
	"""
	
	
	# input	
	img_in = args[0]
	img_in = np.array(img_in)
	
	# keywords
	mother = kwargs.pop('mother','mexicanhat')
	edge = kwargs.pop('edge',0)
	real = kwargs.pop('real',False)
	imaginary = kwargs.pop('imaginary',False)
	absolute = kwargs.pop('absolute',False)
	
	# amount of wavelet parameters
	n_scale = len(args) - 1
	m_scale = []
	d_scale = []
	for a in args[1:]:
		dims = np.shape(np.array(a))
		d_scale.append(len(dims))
		if len(dims) == 0:
			m_scale.append(1)
		else:
			m_scale.append(len(a))
	nr_scale = len(m_scale)
		
	nx,ny = np.shape(img_in)
	#windows_scale = kwargs.pop('windows_scale',[nx,ny])
	
	# edge control
	if edge > 0:
		mscale = int(edge) * max(args[1])
		nnx = nx * 2 * mscale
		nny = ny * 2 * mscale
		img = np.zeros((nnx,nny),dtype=np.complex)
		img[0:nx,0:ny] = img_in
		for i in np.arange(ny): img[:,i] = np.roll(img[:,i],mscale)			
		for j in np.arange(nx): img[j,:] = np.roll(img[j,:],mscale)								
		for i in np.arange(mscale): img[i,:] = img[mscale,:]
		for i in np.arange(mscale): img[nnx-mscale+i,:] = img[nnx-1-mscale,:]
		for i in np.arange(mscale): img[:,i] = img[:,mscale]
		for i in np.arange(mscale): img[:,nny-mscale+i] = img[:,nny-1-mscale]
		img[0:mscale,0:mscale] = img[mscale,mscale]
		img[0:mscale,nny-mscale:] = img[mscale,nny-1-mscale]
		img[nnx-mscale:,0:mscale] = img[nnx-1-mscale,mscale]
		img[nnx-mscale:,nny-mscale:] = img[nnx-1-mscale,nny-1-mscale]
		nx_old = nx * 1
		ny_old = ny * 1
		nx = nnx * 1
		ny = nny * 1
	else:
		img = img_in * 1.
		
	# checking whether reciprocal coordinates are set or not
	knew = False
	if 'kk' in kwargs.keys():
		kk = kwargs['kk']
		dims = np.shape(kk)
		if len(dims) != 3:
			knew = True
		else:
			if (dims[0] != nx) or (dims[1] != ny) or (dims[2] != 2):
				knew = True
	else:
		knew = True
		
	# Fourier transform of image and building reciprocal space
	#fft_img,kx,ky = evsp.fft(img)
	ofx = int(nx/2.) if (nx % 2) == 0 else int((nx-1)/2.)
	ofy = int(ny/2.) if (ny % 2) == 0 else int((ny-1)/2.)		
	kx = np.roll(np.fft.fftfreq(nx) * 2.*np.pi,ofx)
	ky = np.roll(np.fft.fftfreq(ny) * 2.*np.pi,ofy)	
	fft_img = np.fft.fft2(img)
	for i in np.arange(ny): fft_img[:,i] = np.roll(fft_img[:,i],ofx)			
	for j in np.arange(nx): fft_img[j,:] = np.roll(fft_img[j,:],ofy)						
	
	if not knew: 
		kx,ky = kk[:,:,0],kk[:,:,1]
	else:
		ky,kx = np.meshgrid(ky,kx)

	# wavelet choice
	mother = 'wvl_func_' + mother + '2'
	
	# loop through scales
	n_ss = 1
	for ms in m_scale: n_ss = n_ss * ms
	ss = np.arange(n_ss)
	index = where2indices(ss,m_scale)
	
	txt = 'cwt = np.zeros((nx,ny,'
	for j in np.arange(nr_scale): txt = txt + 'm_scale[j],'
	txt = txt[:-1] + '),dtype=complex)'
	exec txt
	for i in ss:
		parms = []
		parms.append(kx)
		parms.append(ky)
		txt = 'cwt[:,:,'
		for j in np.arange(nr_scale): 
			a = args[j+1]
			if d_scale[j] == 0: a = [a]
			parms.append(a[index[j][i]])
			txt = txt + str(index[j][i]) + ','
		txt = txt[:-1] + '] = cwt_i'
		exec 'mother_i = ' + mother + '(*parms,**kwargs)' 
		#cwt_i,x,y = evsp.fft(mother_i * fft_img,inverse=True)
		g = mother_i * fft_img
		for i in np.arange(ny): g[:,i] = np.roll(g[:,i],-ofx)			
		for j in np.arange(nx): g[j,:] = np.roll(g[j,:],-ofy)						
		cwt_i = np.fft.ifft2(g)
		
		exec txt
		
	if edge:
		txt = 'cwt = cwt[mscale:mscale+nx_old,mscale:mscale+ny_old,'
		for a in args[1:]: txt = txt + ':,'
		txt = txt[-1] + ']'	
		
	if real: 
		cwt = cwt.real
	elif imaginary:
		cwt = cwt.imag
	elif absolute:
		cwt = np.abs(cwt)
		
	dims = np.shape(cwt)
	if dims[2] == 1: cwt = cwt[:,:,0]
				
	return cwt
	
	
def wvl_func_mexicanhat2(*args,**kwargs):
	"""
	2D Mexican Hat continuous mother wavelet
	"""
	kx,ky,a = args
	norm = kwargs.pop('norm',1)
	dims = np.shape(a)
	if len(dims) > 0: a = a[0]
	ka2 = a * a * (kx*kx + ky*ky)
	wavelet = 2. * np.pi * a**2 * ka2 * np.exp(-0.5 * ka2)
	if norm != np.inf: wavelet = a**(-2/(norm*1.)) * wavelet
	return wavelet

def wvl_func_gaussian2(*args,**kwargs):
	"""
	A Gaussian, plain blurring, not really a CWT
	"""	
	kx,ky,a = args
	norm = kwargs.pop('norm',1)
	dims = np.shape(a)
	if len(dims) > 0: a = a[0]
	ka2 = a * a * (kx*kx + ky*ky)
	wavelet = np.exp(-0.5 * ka2)
	if norm != np.inf: wavelet = a**(-2./(norm*1.)) * wavelet
	return wavelet
	

def where2indices(ss,ass):
	"""
	convert search index array into dimensions indices of original array
		ss: 1d search index array
		ass: 1d array of dimensions of arr, e.g. ass = np.shape(arr)
	"""
	dims = np.shape(ass)
	m = 1 if len(dims) == 0 else len(ass)
	dims = np.shape(ss)
	nr = 1 if len(dims) == 0 else len(ss)
	index = np.zeros((nr,m),dtype=np.int)
	tt = ss * 1
	ni = 1
	for i in np.arange(m-1): ni = ni * int(ass[i])
	
	i = m - 1
	while i > 0:
		index[:,i] = int(tt / (ni * 1.0))
		tt = tt % ni
		ni = int(ni / (ass[i-1] * 1.0))
		i = i - 1
	index[:,0] = tt

	txt = 'tt = ['
	for l in np.arange(m): txt = txt + 'index[:,' + str(l) + '],'
	txt = txt[:-1] + ']'
	exec txt
	
	return tt


def demo1():
	"""
	Example of calculating and plotting Morlet wavelet transform of a time series
	"""
	nx = 128
	t = np.arange(nx)/(nx-1.) * 10.
	w1 = 2.*np.pi/1.9
	w2 = 0.5*w1 + t/t.max()*(2.5*w1)
	#w2 = 2.*np.pi/0.5
	y = np.cos(w1*t) + 2*np.cos(w2*t)
	import random
	r = np.zeros(nx)
	for i in np.arange(nx): r[i] = random.random() * 1.5
	y = y + r
	C = wavelet_1d(t,y,mother='morlet',siglvl=0.99)
	fig = plt.figure(num=0,figsize=[8,6],dpi=100,facecolor='White',edgecolor='White')
	#plot1(C,ylog=False,yrange=[0,4])	
	plot1(C,freq=False,theme='Blue')	
		
def demo2():
	"""
	2d wavelet transform, using a Mexican Hat wavelet, of an image with a Gaussian shape, 
	and showing the scale-dependency of the CWT value at the Gaussian position wrt norm
	"""
	#import ev_img.plot as evp
	# Create Image
	nx = 64
	ny = 32
	ii = np.arange(nx)
	jj = np.arange(ny)
	JJ,II = np.meshgrid(jj,ii)
	i0,j0,sigma = 32,16,3.
	img = np.exp(-0.5*((II-i0)**2 + (JJ-j0)**2)/sigma**2)
	
	# Perform Mexican Hat wavelet
	a = 1e-5 + np.arange(30)/29.*(10.-1e-5)
	cwt = wavelet_2d(img,a)
	
	# Show CWT transform
	fig = plt.figure(num=1,figsize=[8,6],dpi=100,facecolor='White',edgecolor='White')
	multi = [0,2,2]	
	plt.subplot(2,2,1)
	plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.1)
	plt.title(r'$\mathrm{original}\,,\,\sigma_0\,=\,3$')
	plt.xlim(0,nx); plt.ylim(0,ny)
	im = plt.imshow(np.clip(img,-1,1).T, interpolation='bilinear', cmap='coolwarm',
		origin='lower',filterrad=40,extent=[0,nx,0,ny],vmax=1, vmin=-1)
	plt.plot(np.arange(nx),img[:,j0]*10+15,linewidth=1,color='Black')
	v = [1,9,25]
	for i in np.arange(3):
		plt.subplot(2,2,i+2)
		w = cwt[:,:,v[i]].real/np.abs(cwt).max()
		plt.title('scale = ' + str(round(a[v[i]]*100)/100.)[0:4])
		plt.xlim(0,nx); plt.ylim(0,ny)
		im = plt.imshow(np.clip(w,-1,1).T, interpolation='bilinear', cmap='coolwarm',
			origin='lower',filterrad=40,extent=[0,nx,0,ny],vmax=1, vmin=-1)
		plt.plot(np.arange(nx),w[:,j0]*10+15,linewidth=1,color='Black')
	
	# Show norm dependence of CWT
	a = 1e-5 + np.arange(60)/59.*(10.-1e-5)
	cwt = wavelet_2d(img,a)
	fig = plt.figure(num=2,figsize=[8,6],dpi=100,facecolor='White',edgecolor='White')
	plt.xlim(0,10); plt.ylim(0,1.1)
	plt.xlabel('scale'); plt.ylabel('Re(CWT) [a.u.]')
	plt.title('CWT at (32,16)')
	plt.plot(a,cwt[i0,j0,:].real/np.abs(cwt).max(),color='Blue',linewidth=1.5)
	plt.plot(np.array([1,1])*sigma,np.array([0,1]),linestyle='--',color='Blue',linewidth=1)
	cwt2 = wavelet_2d(img,a,norm=2)
	plt.plot(a,cwt2[i0,j0,:].real/np.abs(cwt2).max(),color='Red',linewidth=1.5)
	plt.plot(np.array([1,1])*sigma*np.sqrt(3.0),np.array([0,1]),linestyle='--',color='Red',linewidth=1)
	cwt3 = wavelet_2d(img,a,norm=np.inf)
	plt.plot(a,cwt3[i0,j0,:].real/np.abs(cwt3).max(),color='Black',linewidth=1.5)

	ax = plt.gca()
	ax.annotate(r'$\mathrm{scale}\,=\,\sigma_0$', xy=(3.05, 0.05),xycoords='data',color='Blue',rotation=90,verticalalignment='right')
	ax.annotate(r'$\mathrm{scale}\,=\,\sqrt{3}\sigma_0$', xy=(np.sqrt(3)*3.05, 0.05),xycoords='data',color='Red',rotation=90,verticalalignment='right')

	ax.annotate(r'$\mathrm{norm}\,=\,1$', xy=(6, 0.3),xycoords='data',color='Blue')
	ax.annotate(r'$\mathrm{norm}\,=\,2$', xy=(6, 0.24),xycoords='data',color='Red')
	ax.annotate(r'$\mathrm{norm}\,=\,\infty$', xy=(6, 0.18),xycoords='data',color='Black')

def main():
	demo1()
	demo2()

if __name__ == '__main__':
	main()
