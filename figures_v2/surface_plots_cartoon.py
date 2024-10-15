import matplotlib.pyplot as plt
import numpy
from matplotlib import cm
import numpy as np

v = numpy.linspace(-1,1,100)
theta = numpy.linspace(0,2*numpy.pi,100)
v, theta = np.mgrid[-1:1:100j, 0.0:2.0*numpy.pi:100j]
z = numpy.sinh(v)
x = numpy.cosh(v)*numpy.cos(theta)
y = numpy.cosh(v)*numpy.sin(theta)

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x,y,z, vmin=z.min() * 2, cmap=cm.Blues,alpha=.5)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
i=10
j=80
iis= range(1,55,10)
jjs = [85]
# ax.scatter(x[i,j],y[i,j],z[i,j],s=40,color='k')
# ax.scatter(x[50:-5:2,50:-1:2],y[50:-5:2,50:-5:2],z[50:-5:2,50:-5:2],s=20,color='k')
ax.plot(numpy.hstack([[x[i,j] for i in iis] for j in jjs]),numpy.hstack([[y[i,j] for i in iis] for j in jjs]),zs=numpy.hstack([[z[i,j] for i in iis] for j in jjs]),linestyle='', marker='.',c='k',markersize=10)

plt.grid(b=None)
plt.axis('off')
ax.set_aspect("equal")
plt.savefig('/home/andrewstier/Downloads/membatross/hyperboloid_cartoon.png',dpi=300)


s, t = np.mgrid[-1:1:100j, -1:1:100j]
z = 8*s
x = 6-6*t-6*s
y = -10*t
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x,y,z, vmin=z.min() * 2, cmap=cm.Blues,alpha=.5)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
i=10
j=80
iis= range(20,90,10)
jjs = [70]
ax.plot(numpy.hstack([[x[i,j] for i in iis] for j in jjs]),numpy.hstack([[y[i,j] for i in iis] for j in jjs]),zs=numpy.hstack([[z[i,j] for i in iis] for j in jjs]),linestyle='', marker='.',c='k',markersize=10)
R=0.0589735
ax.plot(x[numpy.isclose(s**2+t**2,R)],y[numpy.isclose(s**2+t**2,R)],z[numpy.isclose(s**2+t**2,R)],linestyle='', marker='.',c='k',markersize=10,alpha=.25)
plt.grid(b=None)
plt.axis('off')
ax.set_aspect("equal")
plt.savefig('/home/andrewstier/Downloads/membatross/plane_cartoon.png',dpi=300)

print()