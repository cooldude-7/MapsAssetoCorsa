"""THROWAWAY SPIKE: minimal KN5 v5 writer written from the AcTools Kn5Writer field order,
round-tripped through an independent parser (nazariyKul kn5.py) passed as argv[1]."""
import struct, sys, importlib.util, zlib

def s(b: bytearray, txt: str):
    d = txt.encode('utf-8'); b += struct.pack('<i', len(d)) + d

def png_1x1(rgba=(80, 80, 80, 255)):
    raw = b'\x00' + bytes(rgba)
    def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')

def material(b, name, shader, props, maps):
    s(b, name); s(b, shader); b += struct.pack('<B?i', 0, False, 0)
    b += struct.pack('<i', len(props))
    for k, a in props.items():
        s(b, k); b += struct.pack('<f2f3f4f', a, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    b += struct.pack('<i', len(maps))
    for slot, (k, tex) in enumerate(maps):
        s(b, k); b += struct.pack('<i', slot); s(b, tex)

def dummy(b, name, nchildren, m):
    b += struct.pack('<i', 1); s(b, name); b += struct.pack('<i?', nchildren, True)
    b += struct.pack('<16f', *m)

def mesh(b, name, verts, idx, mat, renderable=True, visible=True):
    b += struct.pack('<i', 2); s(b, name); b += struct.pack('<i?', 0, True)
    b += struct.pack('<???', True, visible, False)
    b += struct.pack('<I', len(verts))
    for p, n, uv, t in verts: b += struct.pack('<3f3f2f3f', *p, *n, *uv, *t)
    b += struct.pack('<I', len(idx)) + struct.pack('<%dH' % len(idx), *idx)
    xs, ys, zs = zip(*[v[0] for v in verts]); c = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2)
    r = max(((x-c[0])**2+(y-c[1])**2+(z-c[2])**2)**.5 for x, y, z in zip(xs, ys, zs))
    b += struct.pack('<IIff3ff?', mat, 0, 0.0, 0.0, *c, r, renderable)

def quad(x0, x1, z0, z1, y=0.0, tile=0.1):
    up, tan = (0, 1, 0), (1, 0, 0)
    v = [((x0, y, z0), up, (x0*tile, z0*tile), tan), ((x1, y, z0), up, (x1*tile, z0*tile), tan),
         ((x1, y, z1), up, (x1*tile, z1*tile), tan), ((x0, y, z1), up, (x0*tile, z1*tile), tan)]
    return v, [0, 2, 1, 0, 3, 2]  # winding is a SPIKE question (see report)

def ident(tx=0, ty=0, tz=0):
    return [1,0,0,0, 0,1,0,0, 0,0,1,0, tx,ty,tz,1]  # row-major, translation at 12..14

b = bytearray(b'sc6969') + struct.pack('<i', 5)
b += struct.pack('<i', 1) + struct.pack('<i', 1); s(b, 'asphalt.png'); t = png_1x1(); b += struct.pack('<I', len(t)) + t
b += struct.pack('<i', 1)
material(b, 'road', 'ksPerPixel', {'ksAmbient': .5, 'ksDiffuse': .5, 'ksSpecular': .1, 'ksSpecularEXP': 10}, [('txDiffuse', 'asphalt.png')])
names = ['1ROAD', '1WALL_L', 'AC_START_0', 'AC_PIT_0', 'AC_HOTLAP_START_0', 'AC_AB_START_L', 'AC_AB_START_R', 'AC_AB_FINISH_L', 'AC_AB_FINISH_R']
dummy(b, 'track_root', len(names), ident())
v, i = quad(-4, 4, -200, 0); mesh(b, '1ROAD', v, i, 0)
v, i = quad(-4.5, -4.5, -200, 0); mesh(b, '1WALL_L', v, i, 0, renderable=False)
for n, (x, z) in zip(names[2:], [(0, -10), (0, -5), (0, -15), (-4, -20), (4, -20), (-4, -190), (4, -190)]):
    dummy(b, n, 0, ident(x, 1.0, z))
open(sys.argv[2], 'wb').write(b)
spec = importlib.util.spec_from_file_location('kn5ref', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
model = m.parse(sys.argv[2])
print('bytes', len(b), 'version', model.version, 'textures', list(model.textures), 'materials', [(x.name, x.shader, x.properties, x.textures) for x in model.materials])
print('root', model.root.name, [(c.name, c.kind, c.mesh and len(c.mesh.positions), c.mesh and c.mesh.renderable) for c in model.root.children], 'warnings', model.warnings)
