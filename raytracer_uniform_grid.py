import numpy as np
import moderngl
import pyglet
from numba import njit, prange
import time
start = time.perf_counter()

#ray-tri intersection
@njit(fastmath=True, inline='always')
def rayTriangleIntersection(ox, oy, oz, dx, dy, dz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z):
    # initialize t to invalid
    t = -1.0

    # ---- edges ----
    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z

    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z

    # ---- pvec ----
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x

    det = e1x * px + e1y * py + e1z * pz
    if det > -1e-6 and det < 1e-6:
        return -1.0

    inv_det = 1.0 / det

    # ---- tvec ----
    tx = ox - v0x
    ty = oy - v0y
    tz = oz - v0z

    u = (tx * px + ty * py + tz * pz) * inv_det
    if u < 0.0 or u > 1.0:
        return -1.0

    # ---- qvec ----
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x

    v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return -1.0

    # ---- t ----
    t = (e2x * qx + e2y * qy + e2z * qz) * inv_det

    if t <= 0.0:
        return -1.0

    return t
    
def build_uniform_grid(verts, tris, grid_res):
    tri_count = tris.shape[0]

    scene_min = verts.min(axis=0)
    scene_max = verts.max(axis=0)

    scene_size = scene_max - scene_min
    scene_extent = max(scene_size[0], scene_size[1], scene_size[2])

    cell_size = scene_extent / grid_res

    # 🔑 Expand scene_max so the grid is truly cubic
    scene_max = scene_min + scene_extent

    grid_cells = grid_res ** 3
    cell_counts = np.zeros(grid_cells, dtype=np.int32)

    tri_aabbs_min = np.zeros((tri_count, 3), dtype=np.float32)
    tri_aabbs_max = np.zeros((tri_count, 3), dtype=np.float32)

    for i in range(tri_count):
        v0, v1, v2 = verts[tris[i]]
        tri_min = np.minimum(v0, np.minimum(v1, v2))
        tri_max = np.maximum(v0, np.maximum(v1, v2))

        tri_aabbs_min[i] = tri_min
        tri_aabbs_max[i] = tri_max

        gmin = np.floor((tri_min - scene_min) / cell_size).astype(np.int32)
        gmax = np.floor((tri_max - scene_min) / cell_size).astype(np.int32)

        gmin = np.clip(gmin, 0, grid_res - 1)
        gmax = np.clip(gmax, 0, grid_res - 1)

        for x in range(gmin[0], gmax[0] + 1):
            for y in range(gmin[1], gmax[1] + 1):
                for z in range(gmin[2], gmax[2] + 1):
                    idx = x + y * grid_res + z * grid_res * grid_res
                    cell_counts[idx] += 1

    cell_offsets = np.zeros(grid_cells + 1, dtype=np.int32)
    cell_offsets[1:] = np.cumsum(cell_counts)

    cell_tris = np.zeros(cell_offsets[-1], dtype=np.int32)
    write_ptr = cell_offsets[:-1].copy()

    for i in range(tri_count):
        gmin = np.floor((tri_aabbs_min[i] - scene_min) / cell_size).astype(np.int32)
        gmax = np.floor((tri_aabbs_max[i] - scene_min) / cell_size).astype(np.int32)

        gmin = np.clip(gmin, 0, grid_res - 1)
        gmax = np.clip(gmax, 0, grid_res - 1)

        for x in range(gmin[0], gmax[0] + 1):
            for y in range(gmin[1], gmax[1] + 1):
                for z in range(gmin[2], gmax[2] + 1):
                    idx = x + y * grid_res + z * grid_res * grid_res
                    cell_tris[write_ptr[idx]] = i
                    write_ptr[idx] += 1

    return scene_min, cell_size, grid_res, cell_offsets, cell_tris

@njit(fastmath=True)
def intersect_grid(ox, oy, oz, dx, dy, dz, verts, tris, scene_min_x, scene_min_y, scene_min_z, cell_size, grid_res, cell_starts, cell_tris):

    INF = 1e30
    t = 0.0  # <-- initialize here

    # Grid bounds
    grid_max_x = scene_min_x + grid_res * cell_size
    grid_max_y = scene_min_y + grid_res * cell_size
    grid_max_z = scene_min_z + grid_res * cell_size

    # ---- Ray vs AABB ----
    tmin = 0.0
    tmax = INF

    # X
    if abs(dx) < 1e-8:
        if ox < scene_min_x or ox > grid_max_x:
            return -1.0, -1
    else:
        invd = 1.0 / dx
        t0 = (scene_min_x - ox) * invd
        t1 = (grid_max_x - ox) * invd
        if t0 > t1: t0, t1 = t1, t0
        tmin = max(tmin, t0)
        tmax = min(tmax, t1)
        if tmin > tmax: return -1.0, -1

    # Y
    if abs(dy) < 1e-8:
        if oy < scene_min_y or oy > grid_max_y:
            return -1.0, -1
    else:
        invd = 1.0 / dy
        t0 = (scene_min_y - oy) * invd
        t1 = (grid_max_y - oy) * invd
        if t0 > t1: t0, t1 = t1, t0
        tmin = max(tmin, t0)
        tmax = min(tmax, t1)
        if tmin > tmax: return -1.0, -1

    # Z
    if abs(dz) < 1e-8:
        if oz < scene_min_z or oz > grid_max_z:
            return -1.0, -1
    else:
        invd = 1.0 / dz
        t0 = (scene_min_z - oz) * invd
        t1 = (grid_max_z - oz) * invd
        if t0 > t1: t0, t1 = t1, t0
        tmin = max(tmin, t0)
        tmax = min(tmax, t1)
        if tmin > tmax: return -1.0, -1

    # Entry point
    t = max(tmin, 0.0)
    px = ox + dx * t
    py = oy + dy * t
    pz = oz + dz * t

    # Starting cell
    gx = int((px - scene_min_x) / cell_size)
    gy = int((py - scene_min_y) / cell_size)
    gz = int((pz - scene_min_z) / cell_size)

    gx = min(max(gx, 0), grid_res - 1)
    gy = min(max(gy, 0), grid_res - 1)
    gz = min(max(gz, 0), grid_res - 1)

    # Step directions
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1

    # Next boundary
    next_x = scene_min_x + (gx + (step_x > 0)) * cell_size
    next_y = scene_min_y + (gy + (step_y > 0)) * cell_size
    next_z = scene_min_z + (gz + (step_z > 0)) * cell_size

    # tMax
    tMaxX = t + (next_x - px) / dx if dx != 0 else INF
    tMaxY = t + (next_y - py) / dy if dy != 0 else INF
    tMaxZ = t + (next_z - pz) / dz if dz != 0 else INF

    # tDelta
    tDeltaX = abs(cell_size / dx) if dx != 0 else INF
    tDeltaY = abs(cell_size / dy) if dy != 0 else INF
    tDeltaZ = abs(cell_size / dz) if dz != 0 else INF

    closest_t = INF
    hit_tri = -1

    # ---- DDA loop ----
    while 0 <= gx < grid_res and 0 <= gy < grid_res and 0 <= gz < grid_res:
        cell_idx = gx + gy * grid_res + gz * grid_res * grid_res
        start = cell_starts[cell_idx]
        end   = cell_starts[cell_idx + 1]

        for k in range(start, end):
            tri = cell_tris[k]
            i0, i1, i2 = tris[tri]

            th = rayTriangleIntersection(
                ox, oy, oz, dx, dy, dz,
                verts[i0,0], verts[i0,1], verts[i0,2],
                verts[i1,0], verts[i1,1], verts[i1,2],
                verts[i2,0], verts[i2,1], verts[i2,2]
            )

            if th > 0.0 and th < closest_t:
                closest_t = th
                hit_tri = tri

        # Early exit
        if hit_tri != -1 and closest_t < min(tMaxX, tMaxY, tMaxZ):
            return closest_t, hit_tri

        # Step
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                gx += step_x
                tMaxX += tDeltaX
            else:
                gz += step_z
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                gy += step_y
                tMaxY += tDeltaY
            else:
                gz += step_z
                tMaxZ += tDeltaZ

    return -1.0, -1

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
    
@njit(fastmath=True)
def compute_camera_basis(ox, oy, oz, cx, cy, cz):
    # forward
    fx = cx - ox
    fy = cy - oy
    fz = cz - oz
    inv_len = 1.0 / np.sqrt(fx*fx + fy*fy + fz*fz)
    fx *= inv_len
    fy *= inv_len
    fz *= inv_len

    # world up (DO NOT change unless you want roll)
    upx, upy, upz = 0.0, 1.0, 0.0

    # right = fwd x up
    rx = fy*upz - fz*upy
    ry = fz*upx - fx*upz
    rz = fx*upy - fy*upx
    inv_len = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
    rx *= inv_len
    ry *= inv_len
    rz *= inv_len

    # recompute up = right x fwd
    ux = ry*fz - rz*fy
    uy = rz*fx - rx*fz
    uz = rx*fy - ry*fx

    return fx, fy, fz, rx, ry, rz, ux, uy, uz

@njit(fastmath=True)
def generate_ray(x, y, w, h, aspect, fx, fy, fz, rx, ry, rz, ux, uy, uz):

    u = (2.0*(x + 0.5)/w - 1.0) * aspect
    v = 1.0 - 2.0*(y + 0.5)/h

    dx = fx + u*rx + v*ux
    dy = fy + u*ry + v*uy
    dz = fz + u*rz + v*uz

    inv_len = 1.0 / np.sqrt(dx*dx + dy*dy + dz*dz)
    return dx*inv_len, dy*inv_len, dz*inv_len

@njit(fastmath=True)
def compute_triangle_normal(verts, i0, i1, i2):
    e1x = verts[i1,0] - verts[i0,0]
    e1y = verts[i1,1] - verts[i0,1]
    e1z = verts[i1,2] - verts[i0,2]

    e2x = verts[i2,0] - verts[i0,0]
    e2y = verts[i2,1] - verts[i0,1]
    e2z = verts[i2,2] - verts[i0,2]

    nx = e1y*e2z - e1z*e2y
    ny = e1z*e2x - e1x*e2z
    nz = e1x*e2y - e1y*e2x

    inv_len = 1.0 / np.sqrt(nx*nx + ny*ny + nz*nz)
    return nx*inv_len, ny*inv_len, nz*inv_len

@njit(fastmath=True)
def shade_lambert(px, py, pz, nx, ny, nz, lx, ly, lz):
    lx -= px
    ly -= py
    lz -= pz
    inv_len = 1.0 / np.sqrt(lx*lx + ly*ly + lz*lz)
    lx *= inv_len
    ly *= inv_len
    lz *= inv_len

    d = nx*lx + ny*ly + nz*lz
    return d if d > 0.0 else 0.0
    
@njit(parallel=True, fastmath=True)
def render(image, ox, oy, oz, cx, cy, cz, verts, tris, scene_min_x, scene_min_y, scene_min_z, cell_size, grid_res, cell_offsets, cell_tris, lx, ly, lz):

    h, w, _ = image.shape
    aspect = w / h

    fx, fy, fz, rx, ry, rz, ux, uy, uz = compute_camera_basis(
        ox, oy, oz, cx, cy, cz
    )

    for y in prange(h):
        for x in range(w):

            dx, dy, dz = generate_ray(
                x, y, w, h, aspect,
                fx, fy, fz,
                rx, ry, rz,
                ux, uy, uz
            )

            t_hit, tri_id = intersect_grid(ox, oy, oz, dx, dy, dz, verts, tris, scene_min_x, scene_min_y, scene_min_z, cell_size, grid_res, cell_offsets, cell_tris)

            if tri_id == -1:
                image[y, x, :] = 0
                continue

            px = ox + dx * t_hit
            py = oy + dy * t_hit 
            pz = oz + dz * t_hit

            i0, i1, i2 = tris[tri_id]
            nx, ny, nz = compute_triangle_normal(verts, i0, i1, i2)

            # face camera
            if nx*dx + ny*dy + nz*dz > 0.0:
                nx, ny, nz = -nx, -ny, -nz

            brightness = shade_lambert(px, py, pz, nx, ny, nz, lx, ly, lz)
            c = int(255.0 * brightness)

            image[y, x, 0] = c
            image[y, x, 1] = c
            image[y, x, 2] = c

WIDTH  = 1600
HEIGHT = 800
RES = 1

#cam
CAMERA_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
CAMERA_TARGET = np.array([1.0, 0.0, 0.0], dtype=np.float32)
#light
LIGHT_POS = CAMERA_ORIGIN + np.array([0, 0, 0], dtype=np.float32)

class RayWindow(pyglet.window.Window):
	def __init__(self):
		super().__init__(WIDTH * RES, HEIGHT * RES, "Raytracer", resizable=False)
		
		self.verts, self.tris = load_obj("C:/Users/lucac/Documents/uni year 3/objs/donut.obj")
		Rx = np.array([[1, 0, 0],[0, 0, -1],[0, 1, 0]], dtype=np.float32)
		self.verts = self.verts @ Rx.T
		self.verts += np.array([125, 50, 0], dtype=np.float32)
		
		self.ctx = moderngl.create_context()
		
		self.image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
		
		self.texture = self.ctx.texture((WIDTH, HEIGHT), 3, self.image)
		self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
		
		self.scene_min, self.cell_size, self.grid_res, self.cell_offsets, self.cell_tris = build_uniform_grid(self.verts, self.tris, 32)

		print("built uniform grid")
		
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
		
		self.scene_min_x = np.min(self.verts[:,0]).item()
		self.scene_min_y = np.min(self.verts[:,1]).item()
		self.scene_min_z = np.min(self.verts[:,2]).item()
		
		ox, oy, oz = CAMERA_ORIGIN.astype(np.float32)
		cx, cy, cz = CAMERA_TARGET.astype(np.float32)
		
		cell_size = float(self.cell_size)
		
		render(self.image, float(ox), float(oy), float(oz), float(cx), float(cy), float(cz), self.verts, self.tris, self.scene_min_x, self.scene_min_y, self.scene_min_z, cell_size, self.grid_res, self.cell_offsets, self.cell_tris, float(LIGHT_POS[0]), float(LIGHT_POS[1]), float(LIGHT_POS[2]))
				
	def on_draw(self):
		self.texture.write(self.image)
		self.ctx.clear(0.0, 0.0, 0.0)
		self.texture.use()
		self.vao.render(moderngl.TRIANGLE_STRIP)
		
		end = time.perf_counter()
			

if __name__ == "__main__":
    RayWindow()
    pyglet.app.run()

