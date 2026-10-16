// kn5view: renders a GLB converted from a KN5 for render.py (driven over the DevTools protocol).
//
// window.kn5view.load(cfg)    -> loads the GLB, builds lights/sky, returns scene stats
// window.kn5view.render(view) -> renders one view, returns {png: dataURL, info}
//
// All coordinates are AC world space (X east, Y up, Z south), which is also three.js' frame.

import * as THREE from 'three';
import { GLTFLoader } from '../vendor/three/addons/loaders/GLTFLoader.js';

const SKY_TOP = new THREE.Color(0x5d8fcf);
const SKY_HORIZON = new THREE.Color(0xc9dcec);

const state = {
  renderer: null,
  scene: null,
  root: null,
  solid: [], // meshes rays may hit when looking for the ground (opaque, visible)
  sun: null,
  cfg: {},
};

function skyTexture() {
  const c = document.createElement('canvas');
  c.width = 4;
  c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, '#' + SKY_TOP.getHexString());
  grad.addColorStop(0.55, '#a8c8e8');
  grad.addColorStop(1, '#' + SKY_HORIZON.getHexString());
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function convertMaterial(m, cache, cfg, maxAniso) {
  if (cache.has(m)) return cache.get(m);
  const ex = m.userData || {};
  // AC lights a pixel roughly as tex * (ksAmbient * ambient + ksDiffuse * sun * N.L); keep the
  // relative brightness between materials (0.45 + 0.6 is the repo's usual pair).
  const ka = ex.ksAmbient ?? 0.45;
  const kd = ex.ksDiffuse ?? 0.6;
  const scale = cfg.flatMaterials ? 1 : Math.min(1.4, Math.max(0.35, (ka + kd) / 1.05));
  if (m.map) {
    m.map.anisotropy = maxAniso;
    m.map.needsUpdate = true;
  }
  const out = new THREE.MeshLambertMaterial({
    name: m.name,
    map: m.map || null,
    color: m.color.clone().multiplyScalar(scale),
    side: cfg.doubleSided ? THREE.DoubleSide : m.side,
    alphaTest: m.alphaTest,
    transparent: m.transparent,
    depthWrite: m.depthWrite,
    emissive: m.emissive ? m.emissive.clone() : new THREE.Color(0),
    emissiveMap: m.emissiveMap || null,
  });
  out.userData = ex;
  cache.set(m, out);
  return out;
}

async function load(cfg) {
  const t0 = performance.now();
  state.cfg = cfg;
  const renderer = new THREE.WebGLRenderer({
    antialias: cfg.antialias !== false,
    preserveDrawingBuffer: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(1);
  renderer.setSize(cfg.width || 1280, cfg.height || 720, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = !!cfg.shadows;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.body.appendChild(renderer.domElement);
  state.renderer = renderer;

  const scene = new THREE.Scene();
  scene.background = skyTexture();
  state.scene = scene;

  const gltf = await new GLTFLoader().loadAsync(cfg.glb);
  const t1 = performance.now();
  const root = gltf.scene;
  scene.add(root);
  state.root = root;

  const maxAniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  const cache = new Map();
  const collisionMat = new THREE.MeshLambertMaterial({
    color: 0xff3030,
    transparent: true,
    opacity: 0.35,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const hideRe = cfg.hide ? new RegExp(cfg.hide, 'i') : null;
  const onlyRe = cfg.only ? new RegExp(cfg.only, 'i') : null;
  let meshes = 0;
  let hidden = 0;
  let triangles = 0;
  root.traverse((o) => {
    o.matrixAutoUpdate = false;
    o.updateMatrix();
    if (!o.isMesh) return;
    meshes += 1;
    const ud = o.userData || {};
    let visible = true;
    if (ud.hidden) {
      visible = !!cfg.showCollision;
      if (visible) o.material = collisionMat;
    } else {
      o.material = convertMaterial(o.material, cache, cfg, maxAniso);
    }
    if (hideRe && hideRe.test(o.name)) visible = false;
    if (onlyRe && !onlyRe.test(o.name)) visible = false;
    o.visible = visible;
    if (!visible) {
      hidden += 1;
      return;
    }
    const idx = o.geometry.index;
    triangles += (idx ? idx.count : o.geometry.attributes.position.count) / 3;
    o.castShadow = !!cfg.shadows && ud.castShadows !== false;
    o.receiveShadow = !!cfg.shadows;
    const mat = o.material;
    if (!ud.hidden && !mat.transparent && !(mat.alphaTest > 0)) state.solid.push(o);
  });
  root.updateMatrixWorld(true);

  // Sun from data/lighting.ini (pitch = elevation, heading clockwise from north).
  const pitch = THREE.MathUtils.degToRad(cfg.sunPitch ?? 45);
  const heading = THREE.MathUtils.degToRad(cfg.sunHeading ?? 135);
  const toSun = new THREE.Vector3(
    Math.sin(heading) * Math.cos(pitch),
    Math.sin(pitch),
    -Math.cos(heading) * Math.cos(pitch),
  ).normalize();
  // three.js r155+ lights are physical: a Lambert surface returns albedo * I / pi per unit.
  const hemi = new THREE.HemisphereLight(0xdde8f5, 0x8a8470, (cfg.ambient ?? 0.55) * Math.PI);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff4e0, (cfg.sun ?? 0.75) * Math.PI);
  sun.userData.toSun = toSun;
  sun.position.copy(toSun).multiplyScalar(1000);
  if (cfg.shadows) {
    sun.castShadow = true;
    sun.shadow.mapSize.set(4096, 4096);
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.05;
  }
  scene.add(sun);
  scene.add(sun.target);
  state.sun = sun;

  const box = new THREE.Box3().setFromObject(root);
  return {
    meshes,
    hiddenMeshes: hidden,
    triangles,
    materials: cache.size,
    bbox: [box.min.toArray(), box.max.toArray()],
    loadMs: Math.round(t1 - t0),
    setupMs: Math.round(performance.now() - t1),
    renderer: rendererInfo(renderer),
  };
}

function rendererInfo(renderer) {
  const gl = renderer.getContext();
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
}

const _ray = new THREE.Raycaster();
const _down = new THREE.Vector3(0, -1, 0);

// Height of the first opaque surface below (x, yFrom, z) within maxDrop metres, or null.
function groundAt(x, z, yFrom, maxDrop) {
  _ray.set(new THREE.Vector3(x, yFrom, z), _down);
  _ray.near = 0;
  _ray.far = maxDrop;
  const hits = _ray.intersectObjects(state.solid, false);
  return hits.length ? hits[0].point.y : null;
}

function placeSunShadow(focus, radius) {
  const sun = state.sun;
  sun.target.position.copy(focus);
  sun.position.copy(focus).addScaledVector(sun.userData.toSun, 2000);
  sun.target.updateMatrixWorld();
  sun.updateMatrixWorld();
  const cam = sun.shadow.camera;
  cam.left = -radius;
  cam.right = radius;
  cam.top = radius;
  cam.bottom = -radius;
  cam.near = 100;
  cam.far = 4000;
  cam.updateProjectionMatrix();
}

async function render(view) {
  const t0 = performance.now();
  const { renderer, scene } = state;
  const width = view.width || 1280;
  const height = view.height || 720;
  renderer.setSize(width, height, false);

  const anchor = new THREE.Vector3(...view.anchor);
  let ground = null;
  if (view.snap) {
    ground = groundAt(anchor.x, anchor.z, anchor.y + view.snap.up, view.snap.up + view.snap.down);
    if (ground !== null) anchor.y = ground;
  }
  const eye = anchor.clone().add(new THREE.Vector3(...(view.eyeOffset || [0, 0, 0])));
  const target = anchor.clone().add(new THREE.Vector3(...(view.lookOffset || [0, 0, -1])));

  let camera;
  if (view.ortho) {
    const w = view.orthoWidth;
    const h = (w * height) / width;
    camera = new THREE.OrthographicCamera(-w / 2, w / 2, h / 2, -h / 2, 1, view.far || 8000);
  } else {
    camera = new THREE.PerspectiveCamera(
      view.fov || 60,
      width / height,
      view.near || 0.1,
      view.far || 8000,
    );
  }
  camera.position.copy(eye);
  camera.up.set(...(view.up || [0, 1, 0]));
  camera.lookAt(target);
  camera.updateMatrixWorld();

  scene.fog = view.fog ? new THREE.Fog(SKY_HORIZON, view.fog[0], view.fog[1]) : null;
  if (state.cfg.shadows) placeSunShadow(view.ortho ? anchor : target, view.shadowRadius || 250);

  renderer.render(scene, camera);
  const t1 = performance.now();
  const png = renderer.domElement.toDataURL('image/png');
  return {
    png,
    info: {
      ground,
      eye: eye.toArray().map((v) => Math.round(v * 100) / 100),
      target: target.toArray().map((v) => Math.round(v * 100) / 100),
      drawCalls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
      renderMs: Math.round(t1 - t0),
      encodeMs: Math.round(performance.now() - t1),
    },
  };
}

window.kn5view = { ready: true, load, render, groundAt };
