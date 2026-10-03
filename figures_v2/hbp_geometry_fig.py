import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))

# 1. Poincaré Disk boundary
disk = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
ax.add_patch(disk)

# 2. Label Origin 'O'
ax.text(-0.05, 0.05, r'O', fontsize=28, ha='center', va='center')

def draw_geodesic(p1, p2, ax):
    x1, y1 = p1
    x2, y2 = p2
    
    # Check if p1 and p2 are collinear with origin
    det = x1*y2 - x2*y1
    if np.isclose(det, 0):
        # Line through origin
        # Intersect with unit circle
        # Line equation: ax + by = 0
        # direction vector
        v = np.array([x1, y1])
        if np.allclose(v, 0):
            v = np.array([x2, y2])
        v = v / np.linalg.norm(v)
        ax.plot([-v[0], v[0]], [-v[1], v[1]], color='black', linewidth=1.8)
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=3.5)
        return

    # Circle center orthogonal to unit circle and passing through p1, p2
    A = np.array([
        [2*x1, 2*y1],
        [2*x2, 2*y2]
    ])
    b = np.array([x1**2 + y1**2 + 1, x2**2 + y2**2 + 1])
    xc, yc = np.linalg.solve(A, b)
    R = np.sqrt(xc**2 + yc**2 - 1)

    alpha = np.arctan2(yc, xc)
    C = np.hypot(xc, yc)
    beta = np.arccos((C**2 + R**2 - 1) / (2 * C * R))

    theta1 = alpha + np.pi - beta
    theta2 = alpha + np.pi + beta

    theta = np.linspace(theta1, theta2, 400)
    x_arc = xc + R * np.cos(theta)
    y_arc = yc + R * np.sin(theta)

    ax.plot(x_arc, y_arc, color='black', linewidth=1.8)

    ang1 = np.arctan2(y1 - yc, x1 - xc)
    ang2 = np.arctan2(y2 - yc, x2 - xc)
    ang_diff = (ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi
    
    theta_seg = np.linspace(ang1, ang1 + ang_diff, 200)
    x_seg = xc + R * np.cos(theta_seg)
    y_seg = yc + R * np.sin(theta_seg)
    ax.plot(x_seg, y_seg, color='black', linewidth=3.5)

# 5 points with 0.2 magnitude increments (starting at r = 0.1)
magnitudes = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

disk = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
ax.add_patch(disk)
for r in magnitudes:
    px = np.array([r * np.cos(0), r * np.sin(0)])
    py = np.array([r * np.cos(np.radians(30)), r * np.sin(np.radians(30))])
    draw_geodesic(px, py, ax)
    ax.plot(px[0], px[1], 'ko', markersize=12)
    ax.plot(py[0], py[1], 'ko', markersize=12)
    # Annotate x and y on outermost pair
    if r == magnitudes[-1]:
        ax.text(px[0] + 0.05, px[1] + 0.12, r'x', fontsize=28, ha='center', va='center')
        ax.text(py[0] + 0.10, py[1] - 0.02, r'y', fontsize=28, ha='center', va='center')

# Aspect ratio and axis styling
ax.set_aspect('equal')
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.axis('off')

plt.tight_layout()

plt.savefig('./hbp_plane.png',dpi=300)
print()



fig, ax = plt.subplots(figsize=(8, 8))

# 1. Outer boundary circle (Unit Disk)
disk = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
ax.add_patch(disk)

# 2. Label Origin 'O'
ax.text(-0.05, 0.05, r'O', fontsize=28, ha='center', va='center')

def draw_euclidean_line(p1, p2, ax):
    """
    Draws a straight Euclidean line extended across the unit disk,
    with a bold segment connecting p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2

    # Check if p1 and p2 are the exact same point (e.g., at the origin r=0)
    if np.allclose(p1, p2):
        return

    # Direction vector of the line passing through p1 and p2
    dx = x2 - x1
    dy = y2 - y1
    
    # Parametric line: P(t) = p1 + t * (dx, dy)
    # Intersect with unit circle x^2 + y^2 = 1
    # (x1 + t*dx)^2 + (y1 + t*dy)^2 = 1  =>  A*t^2 + B*t + C = 0
    A = dx**2 + dy**2
    B = 2 * (x1 * dx + y1 * dy)
    C = x1**2 + y1**2 - 1

    discriminant = B**2 - 4 * A * C
    
    if discriminant >= 0:
        t1 = (-B - np.sqrt(discriminant)) / (2 * A)
        t2 = (-B + np.sqrt(discriminant)) / (2 * A)
        
        # Extended straight line spanning the unit disk
        p_start = p1 + t1 * np.array([dx, dy])
        p_end = p1 + t2 * np.array([dx, dy])
        ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color='black', linewidth=1.2)

    # Bold straight segment directly connecting p1 and p2
    ax.plot([x1, x2], [y1, y2], color='black', linewidth=3.5)

# 5 points with 0.2 magnitude increments
magnitudes = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

for r in magnitudes:
    px = np.array([r * np.cos(0), r * np.sin(0)])
    py = np.array([r * np.cos(np.radians(30)), r * np.sin(np.radians(30))])
    
    # Draw straight Euclidean line and bold inner segment
    draw_euclidean_line(px, py, ax)
    
    # Plot black dots for x_i and y_i
    ax.plot(px[0], px[1], 'ko', markersize=12)
    ax.plot(py[0], py[1], 'ko', markersize=12)
    
    # Annotate x and y on the outermost pair
    if r == magnitudes[-1]:
        ax.text(px[0] + 0.05, px[1] + 0.12, r'x', fontsize=28, ha='center', va='center')
        ax.text(py[0] + 0.10, py[1] - 0.02, r'y', fontsize=28, ha='center', va='center')

# Aspect ratio and axis styling
ax.set_aspect('equal')
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.axis('off')

plt.tight_layout()
plt.savefig('./eu_plane.png', dpi=300)
print()