import numpy as np
import moderngl
import pyglet
from numba import njit, prange

#ray-sphere intersection
@njit(fastmath=True, inline='always')
def ray_sphere_intersection(ox, oy, oz, dx, dy, dz, cx, cy, cz, radius):
    # Move ray origin into sphere space
    rx = ox - cx
    ry = oy - cy
    rz = oz - cz

    b = rx * dx + ry * dy + rz * dz
    c = rx * rx + ry * ry + rz * rz - radius * radius

    return b * b - c >= 0.0
    
#ray-tri intersection
@njit(fastmath=True, inline='always')
def rayTriangleIntersection(ox, oy, oz, dx, dy, dz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z):
    # Edges
    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z

    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z

    # p = D × e2
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x

    det = e1x * px + e1y * py + e1z * pz

    # Ray parallel to triangle
    if det > -1e-8 and det < 1e-8:
        return -1.0

    inv_det = 1.0 / det

    # tvec = O - V0
    tx = ox - v0x
    ty = oy - v0y
    tz = oz - v0z

    u = (tx * px + ty * py + tz * pz) * inv_det
    if u < 0.0 or u > 1.0:
        return -1.0

    # q = tvec × e1
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x

    v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return -1.0

    # t = e2 · q
    t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    if t <= 0.0:
        return -1.0

    return t

def load_obj(filename):
    verts = []
    tris = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                # OBJ faces are 1-based, may contain v/vt/vn
                idx = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                tris.append(idx)

    verts = np.array(verts, dtype=np.float32)
    tris  = np.array(tris, dtype=np.int32)
    return verts, tris

WIDTH  = 200
HEIGHT = 200
RES = 2

#cam
CAMERA_ORIGIN = np.array([0.0, 0.0, -125.0], dtype=np.float32)

#light
LIGHT_POS = CAMERA_ORIGIN + np.array([-100.0, -100.0, 0.0], dtype=np.float32)

@njit(parallel=True, fastmath=True)
def render(image, cam_origin, verts, tris):
    print("draw")
    h, w, _ = image.shape
    aspect = w / h

    ox, oy, oz = cam_origin
    lx, ly, lz = LIGHT_POS

    for y in prange(h):
        for x in range(w):
            # Normalized screen coordinates [-1, 1]
            u = (2.0 * (x + 0.5) / w - 1.0) * aspect
            v = 1.0 - 2.0 * (y + 0.5) / h

            # Ray direction
            dx = u
            dy = v
            dz = 1.0
            inv_len = 1.0 / np.sqrt(dx*dx + dy*dy + dz*dz)
            dx *= inv_len
            dy *= inv_len
            dz *= inv_len

            # Initialize closest hit
            closest_t = 1e20
            hit = False
            nx = ny = nz = 0.0
            px = py = pz = 0.0

            # Loop over all triangles
            for i in range(tris.shape[0]):
                i0, i1, i2 = tris[i]

                t = rayTriangleIntersection(
                    ox, oy, oz,
                    dx, dy, dz,
                    verts[i0,0], verts[i0,1], verts[i0,2],
                    verts[i1,0], verts[i1,1], verts[i1,2],
                    verts[i2,0], verts[i2,1], verts[i2,2]
                )

                if t > 0.0 and t < closest_t:
                    closest_t = t
                    hit = True

                    # Intersection point
                    px = ox + dx * t
                    py = oy + dy * t
                    pz = oz + dz * t

                    # Triangle edges
                    e1x = verts[i1,0] - verts[i0,0]
                    e1y = verts[i1,1] - verts[i0,1]
                    e1z = verts[i1,2] - verts[i0,2]

                    e2x = verts[i2,0] - verts[i0,0]
                    e2y = verts[i2,1] - verts[i0,1]
                    e2z = verts[i2,2] - verts[i0,2]

                    # Triangle normal
                    nx = e1y*e2z - e1z*e2y
                    ny = e1z*e2x - e1x*e2z
                    nz = e1x*e2y - e1y*e2x

                    # Normalize
                    inv_len_n = 1.0 / np.sqrt(nx*nx + ny*ny + nz*nz)
                    nx *= inv_len_n
                    ny *= inv_len_n
                    nz *= inv_len_n

            # Write color
            if hit:
                # Compute light vector
                lxv = lx - px
                lyv = ly - py
                lzv = lz - pz
                inv_len_l = 1.0 / np.sqrt(lxv*lxv + lyv*lyv + lzv*lzv)
                lxv *= inv_len_l
                lyv *= inv_len_l
                lzv *= inv_len_l

                # Lambertian shading
                brightness = max(nx*lxv + ny*lyv + nz*lzv, 0.0)
                val = int(255 * brightness)
                image[y, x, 0] = val
                image[y, x, 1] = val
                image[y, x, 2] = val
            else:
                image[y, x, :] = 0

class RayWindow(pyglet.window.Window):
	def __init__(self):
		super().__init__(WIDTH * RES, HEIGHT * RES, "Raytracer", resizable=False)
		
		self.verts, self.tris = load_obj("C:/Users/lucac/Documents/uni year 3/objs/donut.obj")
		Rx = np.array([[1, 0, 0],[0, 0, -1],[0, 1, 0]], dtype=np.float32)
		self.verts = self.verts @ Rx.T
		self.verts += np.array([0, 70, 0], dtype=np.float32)
		
		self.ctx = moderngl.create_context()
		
		self.image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
		
		self.texture = self.ctx.texture((WIDTH, HEIGHT), 3, self.image)
		self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
		
		self.program = self.ctx.program(
			vertex_shader="""
			#version 330
			in vec2 in_pos;
			out vec2 uv;
			void main() {
				uv = in_pos * 0.5 + 0.5;
				gl_Position = vec4(in_pos, 0.0, 1.0);
			}
			""",
			fragment_shader="""
			#version 330
			uniform sampler2D tex;
			in vec2 uv;
			out vec4 fragColor;
			void main() {
				fragColor = texture(tex, uv);
			}
			"""
		)
		
		vertices = np.array([
			-1.0, -1.0,
			1.0, -1.0,
			-1.0,  1.0,
			1.0,  1.0
		], dtype=np.float32)
		
		self.vbo = self.ctx.buffer(vertices)
		self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, "in_pos")
		
		render(self.image, CAMERA_ORIGIN, self.verts, self.tris)
		
	def on_draw(self):
		render(self.image, CAMERA_ORIGIN, self.verts, self.tris)
		self.texture.write(self.image)
		self.ctx.clear(0.0, 0.0, 0.0)
		self.texture.use()
		self.vao.render(moderngl.TRIANGLE_STRIP)

if __name__ == "__main__":
    RayWindow()
    pyglet.app.run()
