"""
Nonlinear Diffusion tool
31 Oct 2017

Linear, nonlinear and anisotropic nonlinear diffusion tools for image analysis. This is a Python version of an equivalent Matlab Nonlinear Diffusion Toolbox.

Prerequisites
This module has been designed for Python 2.7 and uses external standard Python modules: numpy, scipy

Using this module in publications
When using this module is published work, please cite the following paper: 
- 

Also, please add to the Acknowledgments:
The nonlinear diffusion has been performed using the Python nldif module by 
Erwin Verwichte (University of Warwick) and was supported by a UK STFC grant ST/L006324/1.

This software falls under Licences:
- GNU AGPLv3 license
- CRAPL license
"""

__authors__ = ["Erwin Verwichte"]
__email__ = "erwin.verwichte@warwick.ac.uk"

import os,sys
import numpy as np
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
from PIL import Image
import time


__all__ = ['nldif','eedfun','cedif','isodifstep','aosiso','anidifstep','thomas','gsderiv','orideriv','oriedgezero','cm_func','performance','demo']

def ldif(*args,**kwargs):
	"""
	Linear diffusion
	"""
	y = np.array(args[0])	
	dif_time = 0

	nosteps = kwargs.pop('nosteps',10)
	stepsize0 = kwargs.pop('stepsize0',0.1)
	if len(np.shape(stepsize0)) == 0: stepsize0 = [stepsize0]
	stepsize = stepsize0 if len(stepsize0) == nosteps else np.repeat(stepsize0[0],nosteps)

	for i in np.arange(nosteps):
		# convolution kernel
		sd = np.sqrt(2. * stepsize[i])
		kernel_size = np.ceil(3*sd)
		kernel_index = np.arange(2.*kernel_size+1) - kernel_size
		kernel = np.exp(-0.5*(kernel_index/sd)**2) / (np.sqrt(2.*np.pi) * sd)
		kernel = kernel / np.sum(kernel)		
		y = convolv(y,kernel)		
		dif_time = dif_time + stepsize[i]
	return y

def nldif(*args,**kwargs):
	
	"""
	NAME :
		nldif.pro	
	PURPOSE :
		Nonlinear diffusion of an 2d image
		returns the 
	    image y as the result of the application of the Non Linear diffusion
	    to the image u.
	
	    dy/dt = div( d(grad(y)).grad(y) ),  y(0) = u,  y(t+T) = y(t) + T*dy/dt
	
	    The diffusivity function (d) is given by:
	
	       d(g) = 1 - exp( -Cm / (g/lambda)^m ), g > 0
	              1                            , g <=0
	
	    -  The constant Cm is calculated to make the flux (g*d(g)) ascending for g < lambda 
	       and descending for g >= lambda.
			
	CALLING SEQUENCE :
		result = nldif(u,lambda0,rho0,m,stepsize0,nosteps,p)	
	INPUTS
		u: 2d array
		lambda: scalar or array of size nosteps, coherence parameter (if the coherence is inferior to lambda the 
	       flux is increasing with the coherence and if the coherence is larger then lambda
	       the flux decreases as the coherence grows. 
			For time-variable lambda use lambda as a row vector and length(lambda)=number of steps.
		sigma: scalar or array of size nosteps, the space regularization parameter (the stardard deviation of the 
	       gaussian which will be convolved with the image before the gradient is calculated.
	       For time-variable sigma use sigma as a row vector and length(sigma)=number of steps.
		rho: scalar or array of size nosteps, the space regularization parameter for structure S(G). A rho std.dev. gaussian will be 
	       convolved with the elements of the structure tensor before the diffusion tensor is calculated
	       For time-variable rho use rho as a row vector and length(rho)=number of steps.
		m: scalar defines the speed the difusivity parallel to the structure changes for a variation in 
	       the coherence. Big values of 'm' make the diffusivity change quickly. 'm' must be bigger than 1, ideally between 8 < m < 16
		stepsize: scalar or array of size nosteps. For stability use stepsize < 0.25.
		nosteps: scalar of number of integration steps
		p: performs an coherence enhancing diffusion 
	         after every p steps using the given parameters.
	OUTPUT :
	       
	SIDE EFFECTS :
		      
	KEYWORDS :
	 	dfstep: if set only recalculates the diffusivities after dfstep steps (increase speed)  (default = 1)
		aos: uses the AOS scheme to update the image. This allows the steptime to be arbitrarially big (t<inf)
				diffusion is nonlinear but isotropic
		ch: 
		eed: if set will use Coherence enhancing diffusion after p steps
				if ori set diffusion is nonlinear but isotropic, if ori is not set diffusion is nonlinear and anistropic	    
	RESTRICTIONS :
	       
	NOTES :
		Based on Matlab Nonlinear Diffusion Toolbox ()	
	PROCEDURES USED :
		
	MODIFICATION HISTORY :
		Erwin Verwichte (07/05/2009)       
	"""
	
	y = args[0]
	
	dif_time = 0
	accuracy = np.core.getlimits.finfo(float).eps
	
	# Methods
	aos = kwargs.pop('aos',False)
	ch = kwargs.pop('ch',False)
	eed = kwargs.pop('eed',False)
	ori = kwargs.pop('ori',False)

	dfstep = kwargs.pop('dfstep',1)
	nosteps = kwargs.pop('nosteps',10)
	stepsize0 = kwargs.pop('stepsize0',0.1)
	if len(np.shape(stepsize0)) == 0: stepsize0 = [stepsize0]
	stepsize = stepsize0 if len(stepsize0) == nosteps else np.repeat(stepsize0[0],nosteps)
	
	# Determine default values 
	percentile = kwargs.pop('percentile',70)
	if 'lam' in kwargs.keys():
		lambda0 = kwargs['lam']
	else:
		dims = np.shape(percentile)
		if len(dims) > 0:
			lambda0 = []
			for p in percentile: lambda0.append(np.percentile(grad(y,abs=True),p))
			lambda0 = np.array(lambda0)
		else:
			lambda0 = np.percentile(grad(y,abs=True),percentile)
	if len(np.shape(lambda0)) == 0: lambda0 = [lambda0]
	lam = lambda0 if len(lambda0) == nosteps else np.repeat(lambda0[0],nosteps)
	sigma0 = kwargs.pop('sigma',2)
	if len(np.shape(sigma0)) == 0: sigma0 = [sigma0]
	sig = sigma0 if len(sigma0) == nosteps else np.repeat(sigma0[0],nosteps)
	rho0 = kwargs.pop('rho',3.)
	if len(np.shape(rho0)) == 0: rho0 = [rho0]
	rho = rho0 if len(rho0) == nosteps else np.repeat(rho0[0],nosteps)
	m = kwargs.pop('m',10.)
	p = kwargs.pop('p',nosteps+1)
 
	# Cm constant
	Cm = Cmcalc(m)
	
	# Loop through time
	for i in np.arange(nosteps):
		# use coherence enhancing diffusion
		eestep = ((i % p) == 0) & (i > 0) if eed else False
		if eestep:
			kwargs['high'] = True
			kwargs['p'] = p
			kwargs['lam'] = lam[i]
			kwargs['sigma'] = sig[i]
			kwargs['rho'] = rho[i]
			kwargs['m'] = m
			kwargs['stepsize'] = stepsize[i]
			kwargs['steps'] = 1
			kwargs['eedtime'] = eedtime
			kwargs['dfstep'] = dfstep
			kwargs['ori'] = ori
			eedpar,eedtime = eedfun(y,**kwargs)
		else:
			# Diffusivity recalc step
			if (i % dfstep) == 0:
				# gradient
				grd = gsderiv(y,sigma=sig[i],order=1,modulus=True)
				grd = np.clip(grd,accuracy,np.max(grd))
				# Diffusivity
				garg = -Cm/(grd/lam[i])**m
				garg = np.clip(garg,-100,100)
				g = 1. - np.exp(garg) 
			# dydt
			if aos:
				y = aosiso(y,g,stepsize[i])
			else:
				dy = isodifstep(y,g)
				y = y + stepsize[i] * dy
			# diffusion time
			if eestep:
				dif_time = dif_time + eedtime
			else:
				dif_time = dif_time + stepsize[i]
	
	return y

def isodifstep(x,d):
	"""
	Isotropic diffusion step
	 calculates de isotropic (scalar) diffusion
	 step "y" based on the image "x" and on the diffusivity "d". If
	 "d" is constant the diffusion will be linear, if "d" is
	 a matrix the same size as "x" the diffusion will be nonlinear.
	
	 The diffused image is calculated as:
	   xd = x + T*isodifstep(x,d)  , T = step size
	"""		
	d_dpo = d + roll(d,[1,0])
	d_dmo = roll(d_dpo,[-1,0])
	d_dop = d + roll(d,[0,1])
	d_dom = roll(d_dop,[0,-1])

	# Translations of x
	x_xpo = x - roll(x,[1,0])
	x_xmo = roll(x_xpo,[-1,0])
	x_xop = x - roll(x,[0,1])
	x_xom = roll(x_xop,[0,-1])

	# Calculate y = dx/dt
	y = 0.5 * (d_dmo*x_xmo - d_dpo*x_xpo + d_dom*x_xom - d_dop*x_xop)
	return y

def aosiso(x,d,t):
	"""
	Additive Operator Splitting Isotropic Interation
	"""
	y = x * 0.
	p = d * 0. 
	
	# operating on rows
	q = d[:-1,:] + d[1:,:]
	p[0,:] = q[0,:]
	p[-1,:] = q[-1,:]
	p[1:-1,:] = q[:-1,:] + q[1:,:]

	a = 1. + t*p
	b = -t*q
	y = thomas(a,b,b,x)

	# operating on columns
	q = d[:,:-1] + d[:,1:]
	p[:,0] = q[:,0]
	p[:,-1] = q[:,-1]
	p[:,1:-1] = q[:,:-1] + q[:,1:]

	a = 1. + t*p.T
	b = -t*q.T
	y = y + (thomas(a,b,b,x.T)).T
	y = 0.5 * y
	
	return y
	
		
def eedfun(y,**kwargs):
	eedtime = kwargs['stepsize'] * kwargs['steps'] if len(kwargs['stepsize']) == 1 else np.sum(kwargs['stepsize'])
	y = cedif(y,**kwargs)
	return y,eedtime
	
def cedif(y,**kwargs):
	"""
		Coherence Enhancing Diffusion
				
		 dy/dt = div( D(S(grad(us))).grad(y) ),  y(0) = u,  y(t+T) = y(t) + T*dy/dt
	
	    The diffusion tensor (D) is given by:
	       D(S) =  / c1+c2+[(c2-c1)*(s11-s22)/a]         (c2-c1)*s12/a         \
	               \      (c2-c1)*s12/a            c1+c2-[(c2-c1)*(s11-s22)/a] /
	
	         where: a = sqrt{ (s11-s22)^2 + 4*(s12)^2 },
	                c1 and c2 are the diffusivities perpendicular and parallel to the structure orientation,
	                c1 = gamma                                             
	                c2 = / gamma+(1-gamma)*exp(-lambda/(mi1-mi2)^m ) , mi ~= mi2   
	                     \ gamma                                    , mi1 = mi2   
	 
	    The structure tensor (S) is given by:
	       S(G) =  / R(Gx*Gx)  R(Gx*Gy) \    , where G = [Gx Gy]' is the gradient of us
	               \ R(Gx*Gy)  R(Gy*Gy) /      R(.) is smoothing with a 2D rho std.dev. gaussian.
	    INPUTS
		u: 2d array
		lambda: scalar or array of size nosteps, coherence parameter (if the coherence is inferior to lambda the 
	       flux is increasing with the coherence and if the coherence is larger then lambda
	       the flux decreases as the coherence grows. 
			For time-variable lambda use lambda as a row vector and length(lambda)=number of steps.
		sigma: scalar or array of size nosteps, the space regularization parameter (the stardard deviation of the 
	       gaussian which will be convolved with the image before the gradient is calculated.
	       For time-variable sigma use sigma as a row vector and length(sigma)=number of steps.
		rho: scalar or array of size nosteps, the space regularization parameter for structure S(G). A rho std.dev. gaussian will be 
	       convolved with the elements of the structure tensor before the diffusion tensor is calculated
	       For time-variable rho use rho as a row vector and length(rho)=number of steps.
		m: scalar defines the speed the difusivity parallel to the structure changes for a variation in 
	       the coherence. Big values of 'm' make the diffusivity change quickly. 'm' must be bigger than 1.
		stepsize: scalar or array of size nosteps. For stability use stepsize < 0.25.
		nosteps: scalar of number of integration steps	
	"""
	dif_time = 0
	accuracy = np.core.getlimits.finfo(float).eps

	ori = kwargs.pop('ori',False)
	gamma = kwargs.pop('gamma',0.01)
	high = kwargs.pop('high',False)
	
	dfstep = kwargs.pop('dfstep',1)
	nosteps = kwargs.pop('nosteps',10)
	stepsize0 = kwargs.pop('stepsize0',0.1)
	if len(np.shape(stepsize0)) == 0: stepsize0 = [stepsize0]
	stepsize = stepsize0 if len(stepsize0) == nosteps else np.repeat(stepsize0[0],nosteps)
	if np.array(stepsize).max() >= 0.25:
		print 'Warning: stepsize too large, should be less than 0.25!'
	
	# Determine default values 
	percentile = kwargs.pop('percentile',70)
	if 'lam' in kwargs.keys():
		lambda0 = kwargs['lam']
	else:
		dims = np.shape(percentile)
		if len(dims) > 0:
			lambda0 = []
			for p in percentile: lambda0.append(np.percentile(grad(y,abs=True),p))
			lambda0 = np.array(lambda0)
		else:
			lambda0 = np.percentile(grad(y,abs=True),percentile)
	if len(np.shape(lambda0)) == 0: lambda0 = [lambda0]
	lam = lambda0 if len(lambda0) == nosteps else np.repeat(lambda0[0],nosteps)
	sigma0 = kwargs.pop('sigma',2.)
	if len(np.shape(sigma0)) == 0: sigma0 = [sigma0]
	sig = sigma0 if len(sigma0) == nosteps else np.repeat(sigma0[0],nosteps)
	rho0 = kwargs.pop('rho',3.)
	if len(np.shape(rho0)) == 0: rho0 = [rho0]
	rho = rho0 if len(rho0) == nosteps else np.repeat(rho0[0],nosteps)
	m = kwargs.pop('m',10.)
	
	# Cm constant
	Cm = Cmcalc(m)

	for i in np.arange(nosteps):
		# Diffusivity recalc step
		if (i % dfstep) == 0:
			if ori:
				gradx,grady = orideriv(gsderiv(y,sigma=sig[i],order=0))
			else:
				gradx,grady = gsderiv(y,sigma=sig[i],order=1)
			# Structure tensor
			s11 = gsderiv(gradx**2,sigma=rho[i],order=0)
			s12 = gsderiv(gradx*grady,sigma=rho[i],order=0)
			s22 = gsderiv(grady**2,sigma=rho[i],order=0)
			s1_m_s2 = s11 - s22
			# Structure tensor autovalues, mi1 - mi2 coherence
			alfa = np.sqrt(s1_m_s2**2 + 4.*s12**2)
			alfa= np.clip(alfa,accuracy,np.max(alfa))
			# Diffusion tensor autovalues
			g0arg = -Cm / (alfa/lam[i])**m
			g0arg = np.clip(g0arg,-100,100)
			g0 = np.exp(g0arg)
			c1 = gamma * 1.
			c2 = gamma + (1.-gamma) * (g0 * (not high)*1 + (1.-g0) * (high)*1)
			# Diffusion tensor components
			c1_p_c2 = c1 + c2
			c1_m_c2 = c1 - c2
			dd = c1_m_c2 * s1_m_s2 / alfa
			d11 = 0.5 * (c1_p_c2 + dd)
			d12 = -c1_m_c2 * s12 / alfa
			d22 = 0.5 * (c1_p_c2 - dd)
		# dydt
		if ori:
			dx,dy = orideriv(y)
			j1 = d22 * dx + d12 * dy
			j2 = d12 * dx + d11 * dy
			dy = orideriv(j1,order=1) + orideriv(j2,order=2)
		else:
			dy = anidifstep(y,d11,d12,d22)
		# difusion time
		dif_time = dif_time + stepsize[i]
		# image update
		y = y + stepsize[i] * dy
		
	vx = 2. * s12
	vy = -s1_m_s2 + alfa
	ang_st = 180./np.pi * np.arctan2(vx,vy)
	info = kwargs.pop('info',False)
	if info:
		return y,alfa,vx,vy,ang_st
	else:
		return y
		
def anidifstep(x,a,b,c):
	"""
	Anisotropic diffusion step, calculates de anisotropic (tensor) 
	diffusion step "y" based on the image "x" and on the diffusion tensor "D".
	   D = / a  b \
	       \ b  c /
	The diffused image is calculated as:
	  xd = x + T*anidifstep(x,d)  , T = step size	
	"""
	# Translation of a
	a_apo = a + roll(a,[1,0])
	a_amo = roll(a_apo,[-1,0])
	# Translation of b
	bop = roll(b,[0,1])		
	bom = roll(b,[0,-1])		
	bpo = roll(b,[1,0])		
	bmo = roll(b,[-1,0])
	# Translations of c
	c_cop = c + roll(c,[0,1])		
	c_com = roll(c_cop,[0,-1])
	# Translation of x
	xop = roll(b,[0,1])		
	xom = roll(b,[0,-1])		
	xpo = roll(b,[1,0])		
	xmo = roll(b,[-1,0])		
	xpp = roll(b,[1,1])		
	xpm = roll(b,[1,-1])		
	xmp = roll(b,[-1,1])		
	xmm = roll(b,[-1,-1])
	# dydt
	y = 0.5 * ( c_cop * xop + a_amo * xmo - (a_amo + a_apo + c_com + c_cop) * x + a_apo * xpo + c_com * xom)
	y = y + 0.25 * ( -( (bmo+bop) * xmp + (bpo+bom) * xpm ) + (bpo+bop) * xpp + (bmo+bom) * xmm )
	return y	


def thomas(a,b,c,d):
	"""
	Solves a tridiagonal linear system using the very efficient
	 Thomas Algorith. The vector x is the returned answer.
	
	    A*x = d;    /  a1  b1   0   0   0   ...   0  \   / x1 \    / d1 \
	                |  c1  a2  b2   0   0   ...   0  |   | x2 |    | d2 |
	                |   0  c2  a3  b3   0   ...   0  | x | x3 | =  | d3 |
	                |   :   :   :   :   :    :    :  |   | x4 |    | d4 |
	                |   0   0   0   0 cn-2 an-1 bn-1 |   | :  |    |  : |
	                \   0   0   0   0   0  cn-1  an /    \ xn /    \ dn /
	
	- The matrix A must be strictly diagonally dominant for a stable solution.
	- This algorithm solves this system on (5n-4) multiplications/divisions and
	   (3n-3) subtractions.	
	"""			
	m = a * 0.
	l = c * 0.
	y = d * 0.
	n = (np.shape(a))[0]

	# LU decomposition
	# forward substitution

	m[0,:] = a[0,:]*1.
	y[0,:] = d[0,:]*1.
	for i in np.arange(1,n):
		i_1 = i-1
		l[i_1,:] = c[i_1,:] / m[i_1,:]
		m[i,:] = a[i,:] - l[i_1,:] * b[i_1,:]
		y[i,:] = d[i,:] - l[i_1,:] * y[i_1,:]

	# backward substitution
	x = y * 0.
	x[-1,:] = y[-1,:] / m[-1,:]
	for i in reversed(np.arange(n-1)): x[i,:] = (y[i,:] - b[i,:]*x[i+1,:])/m[i,:]

	return x
		
		
def gsderiv(u,**kwargs):
	dims = np.shape(u)
	modulus = kwargs.pop('modulus',False)
	mod_angle = kwargs.pop('mod_angle',False)
	order = kwargs.pop('order',0)
	sigma = kwargs.pop('sigma',0)
	m = modulus*1 + mod_angle*1

	# Smoothing kernel
	if sigma != 0:
		kernel_size = np.ceil(3*sigma)
		kernel_index = np.arange(2*kernel_size+1) - kernel_size
		kernel = np.exp(-0.5*(kernel_index/sigma)**2)
		kernel = kernel / np.sum(kernel)

		# Convolve image with kernel to smooth
		us = convolv(u,kernel)
	else:
		kernel = 1.
		kernel_size = 1
		kernel_index = 0.
		us = u * 1.

	# Calculate image derivatives
	if order == 0:
		return us
	elif order == 1:
		y1 = deriv(us,direction=1)
		y2 = deriv(us,direction=2)
		if m == 1:
			return np.sqrt(y1*y1 + y2*y2)
		elif m ==2:
			return np.sqrt(y1*y1 + y2*y2),180./np.pi * np.arctan2(y2,y1)
		else:
			return y1,y2
	elif order == 2:
		y0 = deriv(us,direction=1)
		y2 = deriv(us,direction=2)
		hx = deriv(y0,direction=1)
		hy = deriv(y0,direction=2)
		return hx,hy,deriv(y2,direction=2)

def orideriv(u,**kwargs):

	kernel = np.zeros((3,3))
	kernel[:,0] = np.array([3,0,-3])
	kernel[:,1] = np.array([10,0,-10])
	kernel[:,2] = np.array([3,0,-3])
	kernel = kernel / 32.
	#kernel = kernel / np.sum(kernel)

	if 'order' in kwargs.keys():
		order = kwargs['order']
		del kwargs['order']
	else:
		order = 0

	if order == 0:
		return convolv(u,kernel,**kwargs),convolv(u,kernel.T,**kwargs)
	elif order == 1:
		return convolv(u,kernel,**kwargs)
	elif order == 2:
		return convolv(u,kernel.T,**kwargs)
	else:
		return convolv(u,kernel,**kwargs)
		
def oriedgezero(u,width=2):
	w = np.arange(-width,width,1)
	for i in w: 
		u[:,i] = 0.
		u[i,:] = 0.
	return u
			
				
def Cmcalc(m):
	"""
	calculates for the diffusivity g(u) = 1 - exp[-cm * u^(-m)] the value of cm for which the flux u g(u) has an extremum
	"""
	func = lambda u: (np.exp(-u) * (1. + m * u) - 1.) * (u != 1)*1 + (u == 1)*1
	u0 = 2. if m >= 4 else 1.
	Cm = scipy.optimize.fsolve(func, u0)
	return Cm

def convolv(x,b,**kwargs):
	"""
	Simple wrapper for scipy routines
	"""
	dims = np.shape(x)
	if len(dims) == 1:
		return np.convolve(x,b)
	elif len(dims) == 2:
		d = np.shape(b)
		if len(d) == 1: 
			return scipy.signal.sepfir2d(x,b,b,**kwargs)
		else:
			kwargs['mode'] = 'same'
			return scipy.signal.fftconvolve(x,b,**kwargs)
	return y

def deriv(*args,**kwargs):
	"""
	Derivative
	"""
	order = kwargs.pop('order',2)
	second = kwargs.pop('second',False)
	direction = kwargs.pop('direction',1)
	if 'axis' in kwargs.keys(): direction = kwargs['axis'] + 1
	
	if len(args) == 2:
		x = args[0]
		u = args[1]
	else:
		u = args[0]
		dims = np.shape(u)
		if len(dims) == 1:
			x = np.arange(dims[0])
		elif len(dims) == 2:
			x = np.arange(dims[direction-1])
	
	dx = x[1]-x[0]
	
	dims = np.shape(u)
	ldims = len(dims)
	
	if ldims == 1:
		n = dims[0]
		if second == False:
			if order ==  2:
				du = 0.50 * shift(u,-1) - 0.50 * shift(u,1)
				du[0] = -0.50 * u[2] + 2.0 * u[1] - 1.50 * u[0]
				du[n-1] = 0.50 * u[n-3] - 2.0 * u[n-2] + 1.50 * u[n-1]
			else:
				du = (-shift(u,-2) + 8.0*shift(u,-1) - 8.0*shift(u,1) + shift(u,2)) / 12.0
				du[0] = (-3.0*u[4] + 16.0*u[3] - 36.0*u[2] + 48.0*u[1] - 25.0*u[0]) / 12.0
				du[1] = (u[4] - 6.0*u[3] + 18.0*u[2] - 10.0*u[1] - 3.0*u[0]) / 12.0
				du[n-1] = (3.0*u[n-5] - 16.0*u[n-4] + 36.0*u[n-3] - 48.0*u[n-2] + 25.0*u[n-1]) / 12.0
				du[n-2] = (-u[n-5] + 6.0*u[n-4] - 18.0*u[n-3] + 10.0*u[n-2] + 3.0*u[n-1]) / 12.0
			du = du / dx
		else:
			if order == 2:
				du = shift(u,-1) - 2.0 * u + shift(u,1)
				du[0] = -u[3] + 4.0*u[2] - 5.0*u[1] + 2.0*u[0]
				du[n-1] = -u[n-4] + 4.0*u[n-3] - 5.0*u[n-2] + 2.0*u[n-1]
			else:
				du = (-shift(u,-2) + 16.0*shift(u,-1) - 30.0*u + 16.0*shift(u,1) - shift(u,2)) / 12.0
				du[0] = (-10.0*u[5] + 61.0*u[4] - 156.0*u[3] + 214.0*u[2] - 154.0*u[1] + 45.0*u[0]) / 12.0
				du[1] = (u[5] - 6.0*u[4] + 14.0*u[3] - 4.0*u[2] - 15.0*u[1] + 10.0*u[0]) / 12.0
				du[n-1] = (-10.0*u[n-6] + 61.0*u[n-5] - 156.0*u[n-4] + 214.0*u[n-3] - 154.0*u[n-2] + 45.0*u[n-1]) / 12.0
				du[n-2] = (u[n-6] - 6.0*u[n-5] + 14.0*u[n-4] - 4.0*u[n-3] - 15.0*u[n-2] + 10.0*u[n-1]) / 12.0	
			du = du / (dx*dx)
	
	if (ldims == 2) and (direction == 1):
		n = dims[0]
		if second == False:
			if order == 2:
				du = 0.50 * shift(u,[-1,0]) - 0.50 * shift(u,[1,0])
				du[0,:] = -0.50 * u[2,:] + 2.0 * u[1,:] - 1.50 * u[0,:]
				du[n-1,:] = 0.50 * u[n-3,:] - 2.0 * u[n-2,:] + 1.50 * u[n-1,:]
			else:
				du = (-shift(u,[-2,0]) + 8.0*shift(u,[-1,0]) - 8.0*shift(u,[1,0]) + shift(u,[2,0])) / 12.0
				du[0,:] = (-3.0*u[4,:] + 16.0*u[3,:] - 36.0*u[2,:] + 48.0*u[1,:] - 25.0*u[0,:]) / 12.0
				du[1,:] = (u[4,:] - 6.0*u[3,:] + 18.0*u[2,:] - 10.0*u[1,:] - 3.0*u[0,:]) / 12.0
				du[n-1,:] = (3.0*u[n-5,:] - 16.0*u[n-4,:] + 36.0*u[n-3,:] - 48.0*u[n-2,:] + 25.0*u[n-1,:]) / 12.0
				du[n-2,:] = (-u[n-5,:] + 6.0*u[n-4,:] - 18.0*u[n-3,:] + 10.0*u[n-2,:] + 3.0*u[n-1,:]) / 12.0
			du = du / dx
		else:
			if order == 2:
				du = shift(u,[-1,0]) - 2.0 * u + shift(u,[1,0])
				du[0,:] = -u[3,:] + 4.0*u[2,:] - 5.0*u[1,:] + 2.0*u[0,:]
				du[n-1,:] = -u[n-4,:] + 4.0*u[n-3,:] - 5.0*u[n-2,:] + 2.0*u[n-1,:]
			else:
				du = (-shift(u,[-2,0]) + 16.0*shift(u,[-1,0]) - 30.0*u + 16.0*shift(u,[1,0]) - shift(u,[2,0])) / 12.0
				du[0,:] = (-10.0*u[5,:] + 61.0*u[4,:] - 156.0*u[3,:] + 214.0*u[2,:] - 154.0*u[1,:] + 45.0*u[0,:]) / 12.0
				du[1,:] = (u[5,:] - 6.0*u[4,:] + 14.0*u[3,:] - 4.0*u[2,:] - 15.0*u[1,:] + 10.0*u[0,:]) / 12.0
				du[n-1,:] = (-10.0*u[n-6,:] + 61.0*u[n-5,:] - 156.0*u[n-4,:] + 214.0*u[n-3,:] - 154.0*u[n-2,:] + 45.0*u[n-1,:]) / 12.0
				du[n-2,:] = (u[n-6,:] - 6.0*u[n-5,:] + 14.0*u[n-4,:] - 4.0*u[n-3,:] - 15.0*u[n-2,:] + 10.0*u[n-1,:]) / 12.0		
			du = du / (dx*dx)

	if (ldims == 2) and (direction == 2):
		n = dims[1]
		if second == False:
			if order == 2:
				du = 0.50 * shift(u,[0,-1]) - 0.50 * shift(u,[0,1])
				du[:,0] = -0.50 * u[:,2] + 2.0 * u[:,1] - 1.50 * u[:,0]
				du[:,n-1] = 0.50 * u[:,n-3] - 2.0 * u[:,n-2] + 1.50 * u[:,n-1]
			else:
				du = (-shift(u,[0,-2]) + 8.0*shift(u,[0,-1]) - 8.0*shift(u,[0,1]) + shift(u,[0,2])) / 12.0
				du[:,0] = (-3.0*u[:,4] + 16.0*u[:,3] - 36.0*u[:,2] + 48.0*u[:,1] - 25.0*u[:,0]) / 12.0
				du[:,1] = (u[:,4] - 6.0*u[:,3] + 18.0*u[:,2] - 10.0*u[:,1] - 3.0*u[:,0]) / 12.0
				du[:,n-1] = (3.0*u[:,n-5] - 16.0*u[:,n-4] + 36.0*u[:,n-3] - 48.0*u[:,n-2] + 25.0*u[:,n-1]) / 12.0
				du[:,n-2] = (-u[:,n-5] + 6.0*u[:,n-4] - 18.0*u[:,n-3] + 10.0*u[:,n-2] + 3.0*u[:,n-1]) / 12.0
			du = du / dx
		else:
			if order == 2:
				du = shift(u,[0,-1]) - 2.0 * u + shift(u,[0,1])
				du[:,0] = -u[:,3] + 4.0*u[:,2] - 5.0*u[:,1] + 2.0*u[:,0]
				du[:,n-1] = -u[:,n-4] + 4.0*u[:,n-3] - 5.0*u[:,n-2] + 2.0*u[:,n-1]
			else:
				du = (-shift(u,[0,-2]) + 16.0*shift(u,[0,-1]) - 30.0*u + 16.0*shift(u,[0,1]) - shift(u,[0,2])) / 12.0
				du[:,0] = (-10.0*u[:,5] + 61.0*u[:,4] - 156.0*u[:,3] + 214.0*u[:,2] - 154.0*u[:,1] + 45.0*u[:,0]) / 12.0
				du[:,1] = (u[:,5] - 6.0*u[:,4] + 14.0*u[:,3] - 4.0*u[:,2] - 15.0*u[:,1] + 10.0*u[:,0]) / 12.0
				du[:,n-1] = (-10.0*u[:,n-6] + 61.0*u[:,n-5] - 156.0*u[:,n-4] + 214.0*u[:,n-3] - 154.0*u[:,n-2] + 45.0*u[:,n-1]) / 12.0
				du[:,n-2] = (u[:,n-6] - 6.0*u[:,n-5] + 14.0*u[:,n-4] - 4.0*u[:,n-3] - 15.0*u[:,n-2] + 10.0*u[:,n-1]) / 12.0	
			du = du / (dx*dx)
	
	return du
		
def grad(*args,**kwargs):
	"""
	Gradient
	"""	
	u = args[0]
	dims = np.shape(u)
	ndims = len(dims)
	
	abs = kwargs.pop('abs',False)
	
	if ndims == 2:
		if len(args) == 3:
			x = args[1]
			d = np.shape(x)
			if len(d) == 2: x = x[:,0]
			y = args[2]
			d = np.shape(y)
			if len(d) == 2: y = y[0,:]
		else:
			x = np.arange(dims[0])
			y = np.arange(dims[1])
		dudx = deriv(x,u,direction=1)
		dudy = deriv(y,u,direction=2)
		if abs:
			return np.sqrt(dudx**2 + dudy**2)
		else:
			return dudx,dudy
	
	if ndims == 3:
		if len(args) == 4:
			x = args[1]
			d = np.shape(x)
			if len(d) == 3: x = x[:,0,0]
			y = args[2]
			d = np.shape(y)
			if len(d) == 3: y = y[0,:.0]
			z = args[2]
			d = np.shape(z)
			if len(d) == 3: z = z[0,0,:]
		else:
			x = np.arange(dims[0])
			y = np.arange(dims[1])
			z = np.arange(dims[2])
		dudx = deriv(x,u,direction=1)
		dudy = deriv(y,u,direction=2)
		dudz = deriv(z,u,direction=3)
		if abs:
			return np.sqrt(dudx**2 + dudy**2 + dudz**2)
		else:
			return dudx,dudy,dudz


def shift(arr,offset):
	"""
	Equivalent of IDL syntax of shift
	"""
	#out = deque(arr)
	#out.rotate(int(offset))
	dims = np.shape(arr)
	if len(dims) == 1: out = np.roll(arr,int(offset))
	elif len(dims) == 2:
		out = np.array(arr)
		if offset[0] != 0:
			for i in np.arange(dims[1]): out[:,i] = shift(out[:,i],offset[0])			
		if offset[1] != 0:
			for j in np.arange(dims[0]): out[j,:] = shift(out[j,:],offset[1])
	elif len(dims) == 3:
		out = np.array(arr)
		if offset[0] != 0:
			for i in np.arange(dims[1]):
				for j in np.arange(dims[2]): out[:,i,j] = shift(out[:,i,j],offset[0]) 
		if offset[1] != 0:
			for i in np.arange(dims[0]):
				for j in np.arange(dims[2]): out[i,:,j] = shift(out[i,:,j],offset[1]) 
		if offset[1] != 0:
			for i in np.arange(dims[0]):
				for j in np.arange(dims[1]): out[i,j,:] = shift(out[i,j,:],offset[2]) 
							
	return out

def roll(x,shft,rot=False):
	"""
	shift matrix elements (same as shift.pro but copying elements instead of rotating)
	"""	
	d = np.shape(shft)
	if len(d) == 0: shft = [shft]
	shft = np.array(shft)
	dims = np.shape(x)
	l = len(dims)
	if l != len(shft):
		print 'Error: incorrect dimensions of shft'
		return -1
	
	if rot:
		y = shift(x,shft)
	else:
		y = x * 1.
		for i in np.arange(l):
			shfti = shft * 0
			shfti[i] = shft[i]
			ii = shft[i] if shft[i] > 0 else dims[i]-1+shft[i]
			y = shift(y,shfti)
			txt = 'y['
			for j in np.arange(i): txt = txt + ':,'
			txt = txt + 'j,'
			for j in np.arange(i+1,l-1): txt = txt + ':,'
			txt = txt[:-1] + '] = y['
			for j in np.arange(i): txt = txt + ':,'
			txt = txt + 'ii,'
			for j in np.arange(i+1,l-1): txt = txt + ':,'
			txt = txt[:-1] + ']'
			if shft[i] > 0:
				for j in np.arange(ii): exec txt
			else:
				for j in np.arange(ii+1,dims[i]): exec txt			
	return y
	

# ====================================================================================================================================

def performance():
	"""
	Performance test.
	"""
	nn = 8
	print "nldif performance test across " + str(nn) + ' scales. Please be patient'
	n = 2.**(5. + np.arange(nn))
	n = n.astype('int')
	t = np.zeros((nn,3))
	nstep = 10

	lam = 0.2
	for i in np.arange(nn):
		print i,n[i]
		img = np.random.rand(n[i],n[i])
		t0 = time.time()
		img_l = ldif(img,stepsize=0.1,nosteps=10)
		t[i,0] = time.time() - t0
		t0 = time.time()
		img_nl = nldif(img,lam=lam,sigma=1.,rho=0.1,m=12,stepsize=0.1,nosteps=nstep,aos=True)
		t[i,1] = time.time() - t0
		t0 = time.time()
		img_nla,alfa,vx,vy,ang_st = cedif(img,lam=lam,sigma=1.0,rho=0.0,m=10,stepsize=0.1,nosteps=nstep,info=True,ori=True)  
		t[i,2] = time.time() - t0
	
	fig = plt.figure(num=0,figsize=[8,6],dpi=100,facecolor='White',edgecolor='White')
	plt.xscale('log')
	plt.xlim(10,1e4)
	plt.xlabel(r'$N$',color='Black',fontsize=13)
	plt.yscale('log')
	plt.ylim(1e-4,10)
	plt.ylabel(r'$t/n_\mathrm{step}$',color='Black',fontsize=13)
	plt.plot(n,t[:,2]/np.float(nstep),marker='o',markerfacecolor='Red',markeredgecolor='Red',color='Red')
	plt.plot(n,t[:,1]/np.float(nstep),marker='o',markerfacecolor='Blue',markeredgecolor='Blue',color='Blue')
	plt.plot(n,t[:,0]/np.float(nstep),marker='o',markerfacecolor='Black',markeredgecolor='Black',color='Black')
	plt.plot(n,2e-3*(n/100.)**2,linestyle='--',linewidth=1,color='Black')
	
	ax = plt.gca()
	xy = (n[0],t[0,2]/(0.75*np.float(nstep))) 
	ax.annotate('anisotropic \nnonlinear ', xy=xy, xytext=xy,
				xycoords='data', horizontalalignment='right', verticalalignment='center', fontsize=13, color='Red')
	xy = (n[5],t[5,1]/(2.*np.float(nstep))) 
	ax.annotate('nonlinear ', xy=xy, xytext=xy,
				xycoords='data', horizontalalignment='left', verticalalignment='center', fontsize=13, color='Blue')
	xy = (n[4],t[4,0]/(2.*np.float(nstep))) 
	ax.annotate('linear ', xy=xy, xytext=xy,
				xycoords='data', horizontalalignment='left', verticalalignment='center', fontsize=13, color='Black')
	xy = (30,3e-4)
	ax.annotate(r'$N^2$ ', xy=xy, xytext=xy,
				xycoords='data', horizontalalignment='right', verticalalignment='center', fontsize=13, color='Black')
	
	return n,t

def demo(nr,dir_path=os.path.join(os.path.expanduser('~'),'images')):

	if nr == 1:	
		print '_____________________________Non Linear diffusion demonstration  1________________________'
		print ' '
		print '  Non linear diffusion, also known as nonlinear diffusion, is a very powerfull image processing'
		print 'technique which can be used to image denoising and image simplification.'
		print '  The non linear diffusion is based on an analogy of physical diffusion processes, like the'
		print 'temperature diffusion on a metal bar, or the diffusion between two fluids put together.'
		print 'These physical diffusion processes are modeled by the following differtial equation:'
		print ' '
		print '          dH/dt = div( d * grad(H) )  ,  where : H is the concentration (temperature)'
		print '                                                 d is the diffusivity (termal conductance)'
		print ' '
		print 'That means that assuming the diffusivity is constant, which is true for the hot metal bar example and'
		print 'for many others, the concentration variation will be faster where the concentration gradient is higher.'
		print 'This phenomena is the linear diffusion - see ldifdemo for more details.'
		print '  This kind of behavior is not very interesting for some image processing tasks, as sometimes there is'
		print 'the need to preserve image borders (which are, by definition, high gradient areas). Using an linear'
		print 'diffusion would quickly destroy the borders.'
		print '  To solve this problem, the nonlinear diffusion makes the diffusivity parameter (d) no longer a'
		print 'constant value, but insted the diffusivity becomes a function of the concentration gradient which'
		print 'decreases for high gradients'
		print ' '
		print '          dH/dt = div( d(grad(H)) * grad(H) )'
		print ' '
		print '  This new formulation allows to perform a image denoising while preserving the borders.'
		print ' '
		print '  To better understand take a look at the following linear and the non linear diffusion examples.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		nx = 30
		ny = 100
		i0 = 15
		
		b = np.zeros((nx,ny))
		b[:,0:np.floor(3*ny/4.).astype('int')+1] = 255.
		
		fig = plt.figure(num=1,figsize=[10,4],dpi=100,facecolor='White',edgecolor='White')
		multi = [0,6,1]
		plt.subplot(multi[2],multi[1],1)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$I(x,y)$')
		plt.axis('off')
		plt.imshow(b.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		plt.plot(i0-5 + b[i0,:]/255.*10,np.arange(ny),color='Black',linewidth=2)	
				
		print '  This image represents a metal bar. The light side at the left is colder then the dark part at the right.'
		print 'As the time passes, the temperature on the bar tends to equalize: this is an example of linear diffusion.'
		print 'Watch the linear diffusion process applied to this image.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		y = ldif(b,stepsize=0.1,nosteps=150)
		
		plt.subplot(multi[2],multi[1],2)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_L(I)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		plt.plot(i0-5 + y[i0,:]/255.*10,np.arange(ny),color='Black',linewidth=2)	
				
		print '  In oposition to what happened in the linear diffusion, if we apply a nonlinear diffusion to this'
		print 'same image the result is that nothing will happen! Why? Because as the diffusivity is very low (even zero) on'
		print 'high contrast areas, the diffusion (heat propagation) between these areas is inhibited. Watch the nonlinear'
		print 'diffusion in progress.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		y = nldif(b,lam=4.,sigma=0.1,rho=0.1,m=16,stepsize=0.2,nosteps=50,dfstep=500)

		plt.subplot(multi[2],multi[1],3)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_{NL}(I)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		plt.plot(i0-5 + y[i0,:]/255.*10,np.arange(ny),color='Black',linewidth=2)	

		print '  So, what is the good thing about this non linear diffusion if it does not change the image at all? The'
		print 'point is, the non linear diffusion does not affect image borders - high gradient (contrast) areas -, but '
		print 'it does affect other parts of the image where the gradient is not so high. Take this example : imagine a noisy'
		print 'version of the metal bar image as seen in the picture.'
		print '  If we want to know which part of the bar is hot and which is cold it would be much better if we could remove'
		print 'the noise and recover the previous image where the two distinct areas appear. As previoully seen, the linear'
		print 'diffusion can remove this noise but it will also blur the entire image (or let the heat go from one side to the'
		print 'other). What would happen if we use the non linear diffusion? We expect that the low gradient areas (noise) -'
		print 'note that the gradient is taken on a blured version of the original image so that the random noise does not affect'
		print 'very much this gradient measure - have a strong diffusion (because their diffusivity is high) while the high gradient'
		print 'areas (the temperature interface) have a much weaker diffusion. Take a look at the results'
		print ' '
		a = raw_input('Press any key to continue...')
		
		bn = b + np.random.randn(nx,ny) * 40. 
		
		plt.subplot(multi[2],multi[1],4)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$I + N$')
		plt.axis('off')
		plt.imshow(bn.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		plt.plot(i0-5 + bn[i0,:]/255.*10,np.arange(ny),color='Black',linewidth=2)	
		
		y = nldif(bn,percentile=70.,sigma=2.5,rho=0.1,m=16,stepsize=0.2,nosteps=20,aos=True)
		
		plt.subplot(multi[2],multi[1],5)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_{NL}(I+N)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		plt.plot(i0-5 + y[i0,:]/255.*10,np.arange(ny),color='Black',linewidth=2)	
				
		print '  Now its clear the power of the nonlinear diffusion filters - they can be used to denoise an image while'
		print 'preserving its borders !!!'
		print '  If you take a close look at the image right at the temperature interface you will see that the noise on it'
		print 'has not been removed. This happens because, as previouly said, the interface is a high gradient area, and the'
		print 'diffusivity is near zero on high gradient areas. So there is almost no diffusion (or no diffusion at all) on '
		print 'the interface, therefore the noise cannot be removed from it.'
		print '  To solve this problem you can use one of both approaches: 1) Make the sigma parameter decrease and the lambda increase'
		print 'as the diffusion approaches its end or 2) use the edge enhance diffusion.'
		print '  Decreasing the sigma parameter makes the gradient to be calculated at a less blured version of the image. As a'
		print 'consequence, the temperature interface effect gets more and more concentrated to the exact points of the interface. The'
		print 'increase on the lambda parameter is needded as while using a less blurred image the noise effect becomes more sensible,'
		print 'however, increasing lambda may make the border to be lost (lambda controls which gradient intensities will be diffused'
		print 'and which will not), so this strategy is a little complicated and a fine tune on the parameter modification must be done.)'
		print 'To help to see this effect look at the image gradient and diffusivity as the diffusion goes on. After the 20th step the'
		print 'sigma parameter begins to decrease and lambda begins to increase. Note the effect on the calculated image gradient and '
		print 'on the diffusivity. '
		print ' '
		a = raw_input('Press any key to continue...')
		
		lam = np.repeat(10,20)
		lam = np.append(lam,20 + np.arange(20)/19.*30.)
		sigma = np.repeat(2.5,20)
		sigma = np.append(sigma,2. - np.arange(20)/19.*(2.-0.01))
		y = nldif(bn,lam=lam,sigma=sigma,rho=0.1,m=12,stepsize=2.,nosteps=40,aos=True)

		plt.subplot(multi[2],multi[1],6)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_{NLA}(I+N)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		plt.plot(i0-5 + y[i0,:]/255.*10,np.arange(ny),color='Black',linewidth=2)	
		
	elif nr == 2:
		print ' '
		print 'Demo 1 showed the basics about the nonlinear diffusion. Now it`s time to'
		print '  go a step further and see the real power of this technique.'
		print 'Observe the following synthetic image'
		print 'Note, make sure the image dif_im1.jpg is in the correct path, edit line for path above!'
		print ' '
		a = raw_input('Press any key to continue...')
				
		im1 = Image.open(os.path.join(dir_path,'dif_im1.jpg'))
		im1 = np.array(im1.rotate(-90))
		dims = np.shape(im1)
		nx = dims[0]
		ny = dims[1]
		
		fig = plt.figure(num=2,figsize=[10,4],dpi=100,facecolor='White',edgecolor='White')
		multi = [0,3,3]
		
		plt.subplot(multi[2],multi[1],1)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$I(x,y)$')
		plt.axis('off')
		plt.imshow(im1.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
				
		print '  Lets put some noise on it. Starting whith an additive gaussian noise stand. dev. = 25% of image amplitude.'
		print 'Watch the result.'
		print '  Note that the image scale has changed as now there are points outside the [0,255] interval; that`s'
		print 'why the white area has become gray.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		im1n = im1 + np.random.randn(nx,ny) * 50
		plt.subplot(multi[2],multi[1],2)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$I + N_{25\%}$')
		plt.axis('off')
		plt.imshow(im1n.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])

		print '  Let`s run the nonlinear diffusion filter to see what we can do to improve this image'
		print ' '
		a = raw_input('Press any key to continue...')
		
		lam = 3. + np.arange(40)/39.*(15.-3.)
		lam = np.append(lam,np.repeat(15,10))
		sigma = 4. - np.arange(40)/39.*3.
		sigma = np.append(sigma,np.repeat(1,10))
		y = nldif(im1n,lam=lam,sigma=sigma,rho=0.1,m=12,stepsize=10.,nosteps=50,aos=True)
		
		plt.subplot(multi[2],multi[1],3)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$L_{NLA}(I + N_{25\%})$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		print '  That was nice, wasn`t it? Well, the only problem is that the circle in the lower part of the cross'
		print 'has disapeared. That happend because the contrast between that circle and the cross itself was too low.'
		print 'This could be avoided by putting lambda (contrast parameter) dow. But then we would have problems to'
		print 'eliminate the noise. So, this is one limitation of this technique. However sometimes this is a good'
		print 'feature, when you want to perform not a denoising but a image simplification, we`ll see it later.'
		print ' '
		print '  OK. Now let do it harder. How about gaussian noise std. dev = 50% of amplitude?'
		print ' '
		a = raw_input('Press any key to continue...')
		
		im1n = im1 + np.random.randn(nx,ny) * 100
		plt.subplot(multi[2],multi[1],4)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$I + N_{50\%}$')
		plt.axis('off')
		plt.imshow(im1n.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		y = nldif(im1n,lam=lam,sigma=sigma,rho=0.1,m=12,stepsize=10.,nosteps=50,aos=True)
		plt.subplot(multi[2],multi[1],5)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$L_{NLA}(I + N_{50\%})$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		print '  The final test. Gaussian noise std. dev = 100% image amplitude.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		im1n = im1 + np.random.randn(nx,ny) * 200
		plt.subplot(multi[2],multi[1],6)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$I + N_{100\%}$')
		plt.axis('off')
		plt.imshow(im1n.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		y = nldif(im1n,lam=lam,sigma=sigma,rho=0.1,m=12,stepsize=10.,nosteps=50,aos=True)
		plt.subplot(multi[2],multi[1],7)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$L_{NLA}(I + N_{100\%})$')
		plt.axis('off')
		plt.imshow(im1n.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
	elif nr == 3:
		print '  Now let`s exam the image simplification capacity of the nonlinear diffusion. As a first example'
		print 'let`s take a sinthetic shark image. Say we want to simplify this image so that we get rid of any small detail.'
		print 'That`s easy.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		im2 = Image.open(os.path.join(dir_path ,'dif_im2.jpg'))
		im2 = np.array(im2.rotate(-90))
		im2 = im2.astype(np.float64)
		dims = np.shape(im2)
		nx = dims[0]
		ny = dims[1]
		
		fig = plt.figure(num=3,figsize=[5,8],dpi=100,facecolor='White',edgecolor='White')
		multi = [0,1,3]
		
		plt.subplot(multi[2],multi[1],1)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$I(x,y)$')
		plt.axis('off')
		plt.imshow(im2.T,cmap='Blues',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		y = nldif(im2,lam=4.0,sigma=1.0,rho=0.10,m=12,stepsize=10. + np.arange(14)/13.*90.,nosteps=14,aos=True)
		plt.subplot(multi[2],multi[1],2)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_{NLA}(I)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Blues',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		print '  Despite the eye and teeth are small (size) they have a too big contrast to the rest of the image, so'
		print 'they are not eliminated. However, any small contrast structure is removed while the main shape in kept intact.'
		print 'We can achieve different simplifications (segmentations) by properly adjusting the parameters.'
		print 'Watch it.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		lam = 2. + np.arange(10)/9.
		lam = np.append(lam,3.5 + np.arange(40)/39.*0.25)
		sigma = np.repeat(1,50)
		y = nldif(im2,lam=lam,sigma=sigma,rho=0.1,m=20,stepsize=100 + np.arange(50)/49.*900,nosteps=50,aos=True)
		plt.subplot(multi[2],multi[1],3)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_{NLA}(I)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Blues',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])

	elif nr == 4:
		
		print '_____________________________Coherence enhancing diffusion demonstration________________________'
		print ' '
		print '  Coherence enhancing diffusion is an advanced type of non linear diffusion where the problem'
		print 'of not removing noise at image borders is solved. This solution consists in making the diffusion'
		print 'at the borders perpendicular to the image gradient (and not parallel to it as usual) or following'
		print 'the borders. This kind of diffusion is able to eliminate noise at borders without loosing them.'
		print '  In order to make this kind of diffusion possible, it necessary to modify a little the diffusion'
		print 'equation, because we are going to work with two different diffusivities: one parallel to the gradient'
		print 'and a new one perpendicular to it.'
		print ' '
		print '          dH/dt = div( D * grad(H) )  ,  where : H is the concentration, '
		print '                                                 D is the diffusion tensor '
		print ' '
		print '         D = transpose(R) x /d1   0\ x R    ,  where : R is the rotation matrix with'
		print '                            \ 0  d2/                      the gradient orientation,'
		print '                                                       d1 is the diffusivity parallel to the grad,'
		print '                                                       d2 is the diffusivity perp. to the grad.'
		print '         R = ______ 1 ______ * /Gx  -Gy\ '
		print '             sqrt(Gx^2+Gy^2)   \Gy   Gx/ '
		print ' '
		print '  The existence of a diffusivity perpendicular to the gradient may sound weird at first as the multiplying'
		print 'of perpendicular vectors results zero. However we must remember that, as the orientation of the image is '
		print 'calculated based on a blurred (rho) version of the gradient of a blurred (sigma) version of the image, this'
		print 'orientation and the real image gradient ( grad(y) ) will almost never have the same direction.'
		print '  The blurring (sigma) of the image has already been explained in the nhdidemo files. It is necessary to '
		print 'supress small image details that would cause large gradient magnitudes in noisy regions. The ideia of the'
		print 'blurring (rho) of the gradient to calculate its orientation follows the same principle. We do not want to'
		print 'have the orientation affected be small variations in the gradient. Instead, we want to extract the main flow'
		print 'directions of the image.'
		print ' '
		print '  Before using the Coherence Enhancing diffusion, let`s take a look at the orientations and the effect of the rho'
		print 'parameter. Look at the following image'
		print ' '
		a = raw_input('Press any key to continue...')
		
		nx = 100
		ny = 100
		b = np.zeros((nx,ny))
		b[:,0:74] = 255
		b[29:69,49:99] = 0
		
		fig = plt.figure(num=4,figsize=[12,8],dpi=100,facecolor='White',edgecolor='White')
		multi = [0,6,3]
		
		plt.subplot(multi[2],multi[1],1)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$I(x,y)$')
		plt.axis('off')
		plt.imshow(b.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		print ' '
		print '  Let`s calculate the structure orientation for some different rho values starting with rho = 0. We will use'
		print 'sigma = 1 for the following examples. We will add a very small amount of noise to the image to better see the'
		print 'effects of rho, otherwise the orientation would be the same for almost the entire image.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		bm = b + np.random.randn(nx,ny) * 20.
		y,alfa,vx,vy,ang_st = cedif(bm,lam=4.0,sigma=1.0,rho=0.0,m=10,stepsize=0.1,nosteps=1,info=True)
		
		plt.subplot(multi[2],multi[1],2)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_{NLC}(I)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],3)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$\mathrm{coherence}(I)$')
		plt.axis('off')
		plt.imshow(alfa.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],4)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$v_x(I)$')
		plt.axis('off')
		plt.imshow(vx.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],5)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$v_y(I)$')
		plt.axis('off')
		plt.imshow(vy.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],6)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$\mathrm{angle}(I)$')
		plt.axis('off')
		plt.imshow(ang_st.T,cmap='coolwarm',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		print ' '
		print '  Using very small values (zero) for rho cause the image structure orientation to be too sensitive too noise.'
		print 'We added gaussian noise with std.dev. = 0.1% of the image amplitude (invisible) and the structure get extremelly'
		print 'noisy. Let`s gradually increase the value of rho to see its effect on the orientation.'
		print ' '
		print '  Using rho = 1.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		y,alfa,vx,vy,ang_st = cedif(bm,lam=4.0,sigma=1.0,rho=1.0,m=10,stepsize=0.1,nosteps=1,info=True)
		
		plt.subplot(multi[2],multi[1],8)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_{NLC}(I)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],9)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$\mathrm{coherence}(I)$')
		plt.axis('off')
		plt.imshow(alfa.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],10)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$v_x(I)$')
		plt.axis('off')
		plt.imshow(vx.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],11)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$v_y(I)$')
		plt.axis('off')
		plt.imshow(vy.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],12)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$\mathrm{angle}(I)$')
		plt.axis('off')
		plt.imshow(ang_st.T,cmap='coolwarm',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		print ' '
		print '  We can see that the orientation is not changing so fast now. Let`s increase rho even more.'
		print ' '
		print '  Using rho = 3.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		y,alfa,vx,vy,ang_st = cedif(bm,lam=4.0,sigma=1.0,rho=3.0,m=10,stepsize=0.1,nosteps=1,info=True)
		
		plt.subplot(multi[2],multi[1],14)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_{NLC}(I)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],15)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$\mathrm{coherence}(I)$')
		plt.axis('off')
		plt.imshow(alfa.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],16)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$v_x(I)$')
		plt.axis('off')
		plt.imshow(vx.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],17)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$v_y(I)$')
		plt.axis('off')
		plt.imshow(vy.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],18)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$\mathrm{angle}(I)$')
		plt.axis('off')
		plt.imshow(ang_st.T,cmap='coolwarm',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		print ' '
		print '  We can observe that, differently from what happend with the nonlinear diffusion, the noise on the'
		print 'borders is quickly eliminated. However, as diffusion is not inhibited on borders, a rounding effect occours.'
		print ' '
			
	elif nr == 5:
	
		print '  Let`s use another image to see the results. This image has already been used on the nldifdemo2. We added'
		print 'gaussian noise std.dev. 50% amplitude to it.'
		print ' '
		a = raw_input('Press any key to continue...')
		
		im1 = Image.open(os.path.join(dir_path ,'dif_im1.jpg'))
		im1 = np.array(im1.rotate(-90))
		im1 = im1.astype(np.float64)
		dims = np.shape(im1)
		nx = dims[0]
		ny = dims[1]
		
		im1n = im1 + np.random.randn(nx,ny) * 50
		
		fig = plt.figure(num=5,figsize=[10,5],dpi=100,facecolor='White',edgecolor='White')
		multi = [0,6,3]
				
		plt.subplot(multi[2],multi[1],1)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$I+N$')
		plt.axis('off')
		plt.imshow(im1n.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		y,alfa,vx,vy,ang_st = cedif(im1n,lam=1.5,sigma=4.0,rho=1.0,m=10,stepsize=0.24,nosteps=50,info=True,ori=True)
		
		plt.subplot(multi[2],multi[1],2)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$D_{NLC}(I)$')
		plt.axis('off')
		plt.imshow(y.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],3)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$\mathrm{coherence}(I)$')
		plt.axis('off')
		plt.imshow(alfa.T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],4)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$v_x(I)$')
		plt.axis('off')
		plt.imshow(oriedgezero(vx,width=4).T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],5)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$v_y(I)$')
		plt.axis('off')
		plt.imshow(oriedgezero(vy,width=4).T,cmap='Reds',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		plt.subplot(multi[2],multi[1],6)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		plt.title(r'$\mathrm{angle}(I)$')
		plt.axis('off')
		plt.imshow(ang_st.T,cmap='coolwarm',interpolation='bilinear',origin='lower',extent=[0,nx,0,ny])
		
		print ' '
		print '  Once we can see the quick noise removing on borders and the rounding effect. This rounding causes small'
		print 'structures - like the small inner cross - to be too blurred.'
		print '  The observation of the properties of the coherence Enhancing diffusion makes us think about using it together'
		print 'with the nonlinear diffusion to achieve better results. The first would be used just a little to remove'
		print 'noise on borders and the latter would take the job from this point to the end.'
		
	else:
		for i in np.arange(1,5): demo(i)

if __name__ == '__main__':
	#n,t = performance()
	"""
	Set path to images before running!
	"""
	dir_path = ''
	if os.path.exists(dir_path):
		demo(0,dir_path=dir_path)
	else:
		print 'Set string dir_path to be directory path to images before running demo!'
		print '  you can find dir_path at bottom of nldif.py'
		
