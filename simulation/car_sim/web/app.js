"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const TAU = Math.PI * 2;
const normalizeAngle = (angle) => {
  while (angle > Math.PI) angle -= TAU;
  while (angle < -Math.PI) angle += TAU;
  return angle;
};

const worldCanvas = $("#worldCanvas");
const worldCtx = worldCanvas.getContext("2d");
const mapCanvas = $("#mapCanvas");
const mapCtx = mapCanvas.getContext("2d");
const cameraCanvas = $("#cameraCanvas");
const cameraCtx = cameraCanvas.getContext("2d");
const armCanvas = $("#armCanvas");
const armCtx = armCanvas.getContext("2d");

const sim = {
  sessionId: (
    globalThis.crypto?.randomUUID?.()
    || `sim-${Date.now()}-${Math.random().toString(16).slice(2)}`
  ),
  task: null,
  workflow: "task",
  room: { width: 8, height: 5 },
  car: { x: 1, y: 1, yaw: 0, length: 0.55, width: 0.38 },
  obstacles: [
    { id: 1, type: "rect", x: 3.05, y: 2.75, w: 1.55, h: 0.75 },
    { id: 2, type: "circle", x: 5.85, y: 1.35, r: 0.46 },
    { id: 3, type: "rect", x: 1.35, y: 3.65, w: 1.25, h: 0.55 },
  ],
  items: [
    { id: 1001, type: "item", x: 6.55, y: 3.8, size: 0.22, color: "#e7a33f", held: false },
  ],
  nextObstacleId: 4,
  nextItemId: 1002,
  selectedId: null,
  tool: "select",
  running: false,
  commandLinear: 0,
  commandAngular: 0,
  linear: 0,
  angular: 0,
  pendingDrive: null,
  commandDeadline: 0,
  lastCommandId: 0,
  collision: false,
  goal: null,
  path: [],
  pathIndex: 0,
  travelled: 0,
  // Localization consumes a complete revolution. Autonomous mapping applies
  // its front-sector filter only when writing occupancy cells.
  lidar: { fovDeg: 360, range: 6 },
  physics: {
    trackWidth: 0.32,
    leftScale: 0.985,
    rightScale: 1.015,
    linearSlip: 0.03,
    angularBias: 0.015,
    speedNoise: 0.012,
    commandLatencyMs: 80,
    lidarNoiseM: 0.012,
    lidarDropout: 0.01,
  },
  scans: [],
  trajectory: [],
  grid: null,
  gridWidth: 0,
  gridHeight: 0,
  gridResolution: 0.05,
  lastMapAt: 0,
  lastStateAt: 0,
  lidarPostPending: false,
  lastSceneSignature: "",
  arm: {
    positions: [1500, 1700, 2000, 1100, 1500, 1200],
    starts: [1500, 1700, 2000, 1100, 1500, 1200],
    targets: [1500, 1700, 2000, 1100, 1500, 1200],
    moveStartedAt: 0,
    moveDurationMs: 0,
    moving: false,
    graspedItemId: null,
  },
  camera: { width: 640, height: 360, detections: [] },
  pickupViewActivated: false,
};

let pointerAction = null;
let dragStart = null;
let dragCurrent = null;
let dragOffset = null;
let lastFrameAt = performance.now();
let toastTimer = null;

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return ratio;
}

function viewport(canvas) {
  const ratio = window.devicePixelRatio > 2 ? 2 : (window.devicePixelRatio || 1);
  const padding = 42 * ratio;
  const scale = Math.min(
    (canvas.width - padding * 2) / sim.room.width,
    (canvas.height - padding * 2) / sim.room.height,
  );
  const drawWidth = sim.room.width * scale;
  const drawHeight = sim.room.height * scale;
  return {
    scale,
    x: (canvas.width - drawWidth) / 2,
    y: (canvas.height - drawHeight) / 2,
    width: drawWidth,
    height: drawHeight,
  };
}

function worldToCanvas(point, canvas = worldCanvas) {
  const view = viewport(canvas);
  return {
    x: view.x + point.x * view.scale,
    y: view.y + (sim.room.height - point.y) * view.scale,
  };
}

function canvasToWorld(event) {
  const rect = worldCanvas.getBoundingClientRect();
  const px = (event.clientX - rect.left) * (worldCanvas.width / rect.width);
  const py = (event.clientY - rect.top) * (worldCanvas.height / rect.height);
  const view = viewport(worldCanvas);
  return {
    x: clamp((px - view.x) / view.scale, 0, sim.room.width),
    y: clamp(sim.room.height - (py - view.y) / view.scale, 0, sim.room.height),
  };
}

function resetMap() {
  sim.gridWidth = Math.ceil(sim.room.width / sim.gridResolution);
  sim.gridHeight = Math.ceil(sim.room.height / sim.gridResolution);
  sim.grid = new Uint8Array(sim.gridWidth * sim.gridHeight);
  sim.trajectory = [];
  sim.scans = [];
  sim.travelled = 0;
  updateMapStats();
}

function gridIndex(x, y) {
  const col = Math.floor(x / sim.gridResolution);
  const row = Math.floor(y / sim.gridResolution);
  if (col < 0 || row < 0 || col >= sim.gridWidth || row >= sim.gridHeight) return -1;
  return row * sim.gridWidth + col;
}

function rayRectangle(origin, direction, rect) {
  let near = -Infinity;
  let far = Infinity;
  for (const axis of ["x", "y"]) {
    const d = direction[axis];
    const min = rect[axis];
    const max = rect[axis] + rect[axis === "x" ? "w" : "h"];
    if (Math.abs(d) < 1e-9) {
      if (origin[axis] < min || origin[axis] > max) return Infinity;
      continue;
    }
    const t1 = (min - origin[axis]) / d;
    const t2 = (max - origin[axis]) / d;
    near = Math.max(near, Math.min(t1, t2));
    far = Math.min(far, Math.max(t1, t2));
    if (near > far) return Infinity;
  }
  if (far < 0) return Infinity;
  return near >= 0 ? near : far;
}

function rayCircle(origin, direction, circle) {
  const ox = origin.x - circle.x;
  const oy = origin.y - circle.y;
  const b = 2 * (ox * direction.x + oy * direction.y);
  const c = ox * ox + oy * oy - circle.r * circle.r;
  const disc = b * b - 4 * c;
  if (disc < 0) return Infinity;
  const root = Math.sqrt(disc);
  const t1 = (-b - root) / 2;
  const t2 = (-b + root) / 2;
  if (t1 >= 0) return t1;
  return t2 >= 0 ? t2 : Infinity;
}

function castRay(angle) {
  const origin = { x: sim.car.x, y: sim.car.y };
  const direction = { x: Math.cos(angle), y: Math.sin(angle) };
  const roomWalls = [
    { x: 0, y: -0.015, w: sim.room.width, h: 0.015 },
    { x: 0, y: sim.room.height, w: sim.room.width, h: 0.015 },
    { x: -0.015, y: 0, w: 0.015, h: sim.room.height },
    { x: sim.room.width, y: 0, w: 0.015, h: sim.room.height },
  ];
  let distance = sim.lidar.range;
  let hit = false;
  for (const wall of roomWalls) {
    const d = rayRectangle(origin, direction, wall);
    if (d < distance) {
      distance = d;
      hit = true;
    }
  }
  for (const obstacle of sim.obstacles) {
    const d = obstacle.type === "rect"
      ? rayRectangle(origin, direction, obstacle)
      : rayCircle(origin, direction, obstacle);
    if (d < distance) {
      distance = d;
      hit = true;
    }
  }
  for (const item of sim.items) {
    if (item.held) continue;
    const d = rayCircle(origin, direction, { x: item.x, y: item.y, r: item.size });
    if (d < distance) {
      distance = d;
      hit = true;
    }
  }
  return {
    x: origin.x + direction.x * distance,
    y: origin.y + direction.y * distance,
    distance,
    hit,
  };
}

function randomNormal() {
  const a = Math.max(Number.EPSILON, Math.random());
  const b = Math.max(Number.EPSILON, Math.random());
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(TAU * b);
}

async function publishLidar(scan) {
  if (sim.lidarPostPending) return;
  sim.lidarPostPending = true;
  try {
    await fetch("/api/lidar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...scan,
        simulator_session_id: sim.sessionId,
      }),
    });
  } catch {
    // 页面自身仍可继续显示；Python 雷达适配器会等待下一帧。
  } finally {
    sim.lidarPostPending = false;
  }
}

function performScan(now) {
  // A physical lidar keeps scanning while the chassis is stopped. Publishing
  // only after motion created a startup deadlock: SLAM waited for its first
  // scan while the simulator waited for the first drive command.
  if (now - sim.lastMapAt < 100) return;
  sim.lastMapAt = now;
  const rayCount = Math.round(sim.lidar.fovDeg * 0.8);
  const halfFov = (sim.lidar.fovDeg * Math.PI / 180) / 2;
  const start = sim.car.yaw - halfFov;
  const span = sim.lidar.fovDeg * Math.PI / 180;
  const scans = [];
  const publishedAngles = [];
  const publishedDistances = [];
  for (let i = 0; i < rayCount; i += 1) {
    const angle = start + span * (i / Math.max(1, rayCount - 1));
    const truth = castRay(angle);
    const noisyDistance = clamp(
      truth.distance + randomNormal() * sim.physics.lidarNoiseM,
      0.02,
      sim.lidar.range,
    );
    const result = {
      ...truth,
      distance: noisyDistance,
      x: sim.car.x + Math.cos(angle) * noisyDistance,
      y: sim.car.y + Math.sin(angle) * noisyDistance,
    };
    scans.push(result);
    // 量程内没有回波时不伪造“最大距离障碍点”。否则这些固定在车体前方的
    // 假点会让 ICP 误以为车辆几乎没有移动，导致建图轨迹比例严重缩小。
    if (truth.hit && Math.random() >= sim.physics.lidarDropout) {
      publishedAngles.push(-halfFov + span * (i / Math.max(1, rayCount - 1)));
      publishedDistances.push(noisyDistance);
    }
    const steps = Math.max(1, Math.floor(result.distance / (sim.gridResolution * 0.55)));
    for (let step = 0; step < steps; step += 1) {
      const t = step / steps;
      const x = sim.car.x + (result.x - sim.car.x) * t;
      const y = sim.car.y + (result.y - sim.car.y) * t;
      const index = gridIndex(x, y);
      if (index >= 0 && sim.grid[index] !== 2) sim.grid[index] = 1;
    }
    if (result.hit) {
      const inset = sim.gridResolution * 0.25;
      const x = result.x - Math.cos(angle) * inset;
      const y = result.y - Math.sin(angle) * inset;
      const index = gridIndex(x, y);
      if (index >= 0) sim.grid[index] = 2;
    }
  }
  sim.scans = scans;
  const last = sim.trajectory.at(-1);
  if (!last || Math.hypot(sim.car.x - last.x, sim.car.y - last.y) > 0.025) {
    sim.trajectory.push({ x: sim.car.x, y: sim.car.y });
  }
  publishLidar({
    timestamp_s: Date.now() / 1000,
    angles_rad: publishedAngles,
    distances_m: publishedDistances,
    ground_truth_pose: {
      x_m: sim.car.x,
      y_m: sim.car.y,
      yaw_rad: sim.car.yaw,
    },
  });
  updateMapStats();
}

function carRadius() {
  return Math.hypot(sim.car.length, sim.car.width) / 2;
}

function pointBlocked(x, y, extra = 0) {
  const radius = carRadius() + extra;
  if (x < radius || y < radius || x > sim.room.width - radius || y > sim.room.height - radius) {
    return true;
  }
  const obstacleBlocked = sim.obstacles.some((obstacle) => {
    if (obstacle.type === "circle") {
      return Math.hypot(x - obstacle.x, y - obstacle.y) < radius + obstacle.r;
    }
    const closestX = clamp(x, obstacle.x, obstacle.x + obstacle.w);
    const closestY = clamp(y, obstacle.y, obstacle.y + obstacle.h);
    return Math.hypot(x - closestX, y - closestY) < radius;
  });
  if (obstacleBlocked) return true;
  return sim.items.some((item) => (
    !item.held && Math.hypot(x - item.x, y - item.y) < radius + item.size
  ));
}

function applyMotion(dt) {
  if (!sim.running) return;
  const now = performance.now();
  if (sim.pendingDrive && now >= sim.pendingDrive.applyAt) {
    sim.commandLinear = sim.pendingDrive.linear;
    sim.commandAngular = sim.pendingDrive.angular;
    sim.pendingDrive = null;
  }
  if (sim.commandDeadline && now > sim.commandDeadline) {
    sim.commandLinear = 0;
    sim.commandAngular = 0;
    sim.pendingDrive = null;
    sim.commandDeadline = 0;
    addLog("TTL 到期，底盘自动停车");
  }
  updateAutopilot();

  const physics = sim.physics;
  const active = Math.abs(sim.commandLinear) > 0.0001 || Math.abs(sim.commandAngular) > 0.0001;
  const leftCommand = sim.commandLinear - sim.commandAngular * physics.trackWidth / 2;
  const rightCommand = sim.commandLinear + sim.commandAngular * physics.trackWidth / 2;
  const leftWheel = leftCommand
    * physics.leftScale
    * (1 - physics.linearSlip)
    + (active ? randomNormal() * physics.speedNoise : 0);
  const rightWheel = rightCommand
    * physics.rightScale
    * (1 - physics.linearSlip)
    + (active ? randomNormal() * physics.speedNoise : 0);
  const targetLinear = (leftWheel + rightWheel) / 2;
  const targetAngular = (rightWheel - leftWheel) / physics.trackWidth
    + (active ? physics.angularBias : 0);
  const response = clamp(dt * 8, 0, 1);
  sim.linear += (targetLinear - sim.linear) * response;
  sim.angular += (targetAngular - sim.angular) * response;
  if (!active && Math.abs(sim.linear) < 0.001) sim.linear = 0;
  if (!active && Math.abs(sim.angular) < 0.001) sim.angular = 0;

  const nextYaw = normalizeAngle(sim.car.yaw + sim.angular * dt);
  const nextX = sim.car.x + Math.cos(sim.car.yaw + sim.angular * dt * 0.5) * sim.linear * dt;
  const nextY = sim.car.y + Math.sin(sim.car.yaw + sim.angular * dt * 0.5) * sim.linear * dt;
  if (!pointBlocked(nextX, nextY)) {
    sim.travelled += Math.hypot(nextX - sim.car.x, nextY - sim.car.y);
    sim.car.x = nextX;
    sim.car.y = nextY;
    sim.car.yaw = nextYaw;
    sim.collision = false;
  } else if (Math.abs(sim.linear) > 0.001) {
    sim.commandLinear = 0;
    sim.commandAngular = 0;
    sim.linear = 0;
    sim.angular = 0;
    sim.pendingDrive = null;
    sim.path = [];
    sim.goal = null;
    sim.collision = true;
    addLog("碰撞保护：运动已停止", true);
    toast("检测到碰撞，已停车");
  } else {
    sim.car.yaw = nextYaw;
  }
}

function updateAutopilot() {
  if (!sim.path.length || sim.pathIndex >= sim.path.length) return;
  const target = sim.path[sim.pathIndex];
  const dx = target.x - sim.car.x;
  const dy = target.y - sim.car.y;
  const distance = Math.hypot(dx, dy);
  if (distance < 0.12) {
    sim.pathIndex += 1;
    if (sim.pathIndex >= sim.path.length) {
      sim.commandLinear = 0;
      sim.commandAngular = 0;
      sim.path = [];
      sim.goal = null;
      addLog("goto 完成：已到达目标点");
      toast("已到达目标点");
    }
    return;
  }
  const headingError = normalizeAngle(Math.atan2(dy, dx) - sim.car.yaw);
  sim.commandAngular = clamp(headingError * 2.8, -1.7, 1.7);
  sim.commandLinear = Math.abs(headingError) > 0.7 ? 0 : clamp(distance * 0.65, 0.11, 0.38);
  sim.commandDeadline = 0;
}

function planPath(goal) {
  const resolution = 0.16;
  const cols = Math.ceil(sim.room.width / resolution);
  const rows = Math.ceil(sim.room.height / resolution);
  const key = (col, row) => row * cols + col;
  const fromKey = (value) => ({ col: value % cols, row: Math.floor(value / cols) });
  const toCell = (point) => ({
    col: clamp(Math.floor(point.x / resolution), 0, cols - 1),
    row: clamp(Math.floor(point.y / resolution), 0, rows - 1),
  });
  const start = toCell(sim.car);
  const end = toCell(goal);
  if (pointBlocked(goal.x, goal.y, 0.03)) return [];
  const startKey = key(start.col, start.row);
  const endKey = key(end.col, end.row);
  const open = [startKey];
  const openSet = new Set(open);
  const cameFrom = new Map();
  const gScore = new Map([[startKey, 0]]);
  const fScore = new Map([[startKey, Math.hypot(end.col - start.col, end.row - start.row)]]);
  const directions = [
    [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
    [1, 1, Math.SQRT2], [1, -1, Math.SQRT2],
    [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2],
  ];
  while (open.length) {
    let bestIndex = 0;
    for (let i = 1; i < open.length; i += 1) {
      if ((fScore.get(open[i]) ?? Infinity) < (fScore.get(open[bestIndex]) ?? Infinity)) bestIndex = i;
    }
    const current = open.splice(bestIndex, 1)[0];
    openSet.delete(current);
    if (current === endKey) {
      const cells = [current];
      let cursor = current;
      while (cameFrom.has(cursor)) {
        cursor = cameFrom.get(cursor);
        cells.push(cursor);
      }
      cells.reverse();
      const points = cells.map((cellKey) => {
        const cell = fromKey(cellKey);
        return { x: (cell.col + 0.5) * resolution, y: (cell.row + 0.5) * resolution };
      });
      const simplified = points.filter((_, index) => index % 3 === 0);
      simplified.push({ x: goal.x, y: goal.y });
      return simplified;
    }
    const cell = fromKey(current);
    for (const [dc, dr, cost] of directions) {
      const col = cell.col + dc;
      const row = cell.row + dr;
      if (col < 0 || row < 0 || col >= cols || row >= rows) continue;
      const x = (col + 0.5) * resolution;
      const y = (row + 0.5) * resolution;
      if (pointBlocked(x, y, 0.025)) continue;
      const neighbor = key(col, row);
      const tentative = (gScore.get(current) ?? Infinity) + cost;
      if (tentative >= (gScore.get(neighbor) ?? Infinity)) continue;
      cameFrom.set(neighbor, current);
      gScore.set(neighbor, tentative);
      fScore.set(neighbor, tentative + Math.hypot(end.col - col, end.row - row));
      if (!openSet.has(neighbor)) {
        open.push(neighbor);
        openSet.add(neighbor);
      }
    }
  }
  return [];
}

function prepareGoal(x, y) {
  const goal = {
    x: clamp(Number(x), 0, sim.room.width),
    y: clamp(Number(y), 0, sim.room.height),
  };
  const path = planPath(goal);
  if (!path.length) {
    toast("目标不可达，请换一个位置");
    return false;
  }
  sim.goal = goal;
  sim.path = path;
  sim.pathIndex = 0;
  toast("目标已记录，完成布置后开始执行");
  addLog(`预设目标 → (${goal.x.toFixed(2)}, ${goal.y.toFixed(2)})`);
  return true;
}

function setGoal(x, y) {
  if (sim.workflow !== "execute" && !beginExecution(false)) return false;
  const goal = {
    x: clamp(Number(x), 0, sim.room.width),
    y: clamp(Number(y), 0, sim.room.height),
  };
  const path = planPath(goal);
  if (!path.length) {
    toast("目标不可达，请换一个位置");
    addLog(`goto 拒绝：(${goal.x.toFixed(2)}, ${goal.y.toFixed(2)}) 不可达`, true);
    return false;
  }
  sim.goal = goal;
  sim.path = path;
  sim.pathIndex = 0;
  sim.pendingDrive = null;
  setRunning(true);
  addLog(`goto → (${goal.x.toFixed(2)}, ${goal.y.toFixed(2)})，${path.length} 个路径点`);
  return true;
}

function setTwist(linearMmS, angularMradS, ttlMs = 600) {
  // External mapping scripts do not click through the page workflow first.
  // Treat their first chassis command as selecting the mapping task.
  if (!sim.task) setTask("mapping", false);
  if (sim.workflow !== "execute" && !beginExecution(false)) return;
  sim.path = [];
  sim.goal = null;
  const linear = clamp(Number(linearMmS) / 1000, -.55, .55);
  const angular = clamp(Number(angularMradS) / 1000, -3.5, 3.5);
  const pendingMatches = sim.pendingDrive
    && sim.pendingDrive.linear === linear
    && sim.pendingDrive.angular === angular;
  const activeMatches = sim.commandLinear === linear && sim.commandAngular === angular;
  if (!pendingMatches && !activeMatches) {
    sim.pendingDrive = {
      linear,
      angular,
      applyAt: performance.now() + sim.physics.commandLatencyMs,
    };
  }
  sim.commandDeadline = performance.now() + clamp(Number(ttlMs), 50, 2500);
  setRunning(true);
}

function stopMotion(message = "STOP：底盘停车") {
  sim.commandLinear = 0;
  sim.commandAngular = 0;
  sim.pendingDrive = null;
  sim.commandDeadline = 0;
  sim.path = [];
  sim.goal = null;
  addLog(message);
}

function setWorkflow(stage) {
  sim.workflow = stage;
  for (const name of ["task", "scene", "execute"]) {
    document.body.classList.toggle(`workflow-${name}`, stage === name);
  }
  document.body.classList.toggle("scene-locked", stage === "execute");
  $$("[data-view]").forEach((button) => {
    const view = button.dataset.view;
    let enabled = view === "tasks";
    if (stage === "scene") enabled = enabled || view === "scene";
    if (stage === "execute") {
      enabled = enabled
        || view === "testing"
        || view === "console"
        || (view === "pickup" && sim.task === "pickup");
    }
    button.disabled = !enabled;
  });
  $("#confirmSceneBtn").disabled = stage !== "scene";
  $("#modeBadge").textContent = stage === "execute" ? "场景已锁定" : "编辑中";
  $("#modeBadge").style.background = stage === "execute" ? "#fde3d8" : "";
  $("#modeBadge").style.color = stage === "execute" ? "#a54220" : "";
  if (stage === "scene") {
    $("#canvasTip").textContent = "场景布置阶段：选择工具后编辑";
  } else if (stage === "execute") {
    $("#canvasTip").textContent = "任务运行中：场景已锁定，点击可下发导航目标";
  }
}

function switchView(view, force = false) {
  const targetButton = $(`[data-view="${view}"]`);
  if (!force && targetButton?.disabled) {
    toast("请先完成当前步骤");
    return false;
  }
  for (const name of ["tasks", "scene", "testing", "pickup", "console"]) {
    document.body.classList.toggle(`view-${name}`, view === name);
  }
  $$("[data-view]").forEach((item) => {
    item.classList.toggle("active", item.dataset.view === view);
  });
  requestAnimationFrame(() => {
    renderWorld();
    renderMap();
    renderCamera();
    renderArm();
  });
  return true;
}

function setTask(task, announce = true, resetWorkflow = false) {
  const changed = sim.task !== task;
  sim.task = task;
  sim.pickupViewActivated = false;
  $("#taskBadge").textContent = task === "pickup" ? "物品夹取任务" : "建图探索任务";
  $$("[data-task-card]").forEach((card) => {
    card.classList.toggle("selected", card.dataset.taskCard === task);
  });
  if (changed || resetWorkflow || sim.workflow === "task") setWorkflow("scene");
  if (announce) {
    addLog(`任务切换：${task === "pickup" ? "物品识别与夹取" : "自主探索与建图"}`);
    toast(task === "pickup" ? "已选择夹取任务，请布置物品" : "已选择建图任务");
  }
}

function beginExecution(navigate = true) {
  if (!sim.task) {
    toast("请先选择任务");
    return false;
  }
  if (sim.task === "pickup" && !sim.items.some((item) => !item.held)) {
    toast("夹取任务至少需要一个未夹取物品");
    return false;
  }
  setWorkflow("execute");
  setRunning(true);
  if (navigate) switchView("testing", true);
  addLog("场景布置完成：已锁定场景并进入任务运行");
  return true;
}

function endExecution() {
  stopMotion("任务结束：底盘已停车");
  armStop();
  setRunning(false);
  sim.pickupViewActivated = false;
  setWorkflow("scene");
  switchView("scene", true);
  toast("任务已结束，可以重新布置场景");
}

function updateArm(now = performance.now()) {
  if (!sim.arm.moving) return;
  const t = clamp((now - sim.arm.moveStartedAt) / Math.max(1, sim.arm.moveDurationMs), 0, 1);
  const eased = t * t * (3 - 2 * t);
  sim.arm.positions = sim.arm.starts.map((start, index) => (
    Math.round(start + (sim.arm.targets[index] - start) * eased)
  ));
  if (t >= 1) {
    sim.arm.moving = false;
    if (sim.arm.positions[5] >= 1450) attemptGrasp();
    if (sim.arm.positions[5] <= 1250) releaseItem();
  }
}

function setArmJoints(joints, durationMs = 800) {
  if (sim.workflow !== "execute" && !beginExecution(false)) return;
  updateArm();
  sim.arm.starts = [...sim.arm.positions];
  sim.arm.targets = [...sim.arm.positions];
  for (const [jointId, pulseUs] of joints) {
    if (jointId >= 0 && jointId < 6) sim.arm.targets[jointId] = Number(pulseUs);
  }
  sim.arm.moveStartedAt = performance.now();
  sim.arm.moveDurationMs = clamp(Number(durationMs), 20, 10000);
  sim.arm.moving = true;
  addLog(`SET_ARM_JOINTS ${joints.map(([id, pulse]) => `J${id}=${pulse}`).join(" ")} · ${sim.arm.moveDurationMs}ms`);
}

function armStop() {
  updateArm();
  sim.arm.targets = [...sim.arm.positions];
  sim.arm.moving = false;
  addLog("ARM_STOP：机械臂保持当前位置");
}

function cameraDetections() {
  const width = sim.camera.width;
  const height = sim.camera.height;
  const halfFov = 35 * Math.PI / 180;
  const cameraX = sim.car.x + Math.cos(sim.car.yaw) * sim.car.length * .48;
  const cameraY = sim.car.y + Math.sin(sim.car.yaw) * sim.car.length * .48;
  const detections = [];
  for (const item of sim.items) {
    if (item.held) continue;
    const dx = item.x - cameraX;
    const dy = item.y - cameraY;
    const distance = Math.hypot(dx, dy);
    const bearing = normalizeAngle(Math.atan2(dy, dx) - sim.car.yaw);
    if (distance > 3.5 || Math.abs(bearing) > halfFov) continue;
    const centerX = width / 2 + Math.tan(bearing) / Math.tan(halfFov) * width / 2;
    const boxSize = clamp(118 / Math.max(.28, distance), 34, 190);
    const centerY = height * .61;
    detections.push({
      id: item.id,
      label: "pickup_item",
      confidence: Number(clamp(.98 - distance * .055 - Math.abs(bearing) * .08, .62, .98).toFixed(3)),
      bbox: [
        Math.round(centerX - boxSize / 2),
        Math.round(centerY - boxSize / 2),
        Math.round(boxSize),
        Math.round(boxSize),
      ],
      distance_m: Number(distance.toFixed(3)),
      bearing_rad: Number(bearing.toFixed(4)),
      color: item.color,
    });
  }
  detections.sort((a, b) => a.distance_m - b.distance_m);
  sim.camera.detections = detections;
  return detections;
}

function attemptGrasp() {
  const target = cameraDetections()[0];
  const armYaw = (sim.arm.positions[0] - 1500) / 10 * Math.PI / 180;
  const alignmentError = target
    ? normalizeAngle(target.bearing_rad - armYaw)
    : Infinity;
  if (!target || target.distance_m > 1.25 || Math.abs(alignmentError) > .25) {
    $("#graspBadge").textContent = "未夹到物品";
    toast("夹取失败：物品未进入夹爪范围");
    return false;
  }
  const item = sim.items.find((candidate) => candidate.id === target.id);
  if (!item) return false;
  item.held = true;
  sim.arm.graspedItemId = item.id;
  $("#graspBadge").textContent = `已夹取 #${item.id}`;
  addLog(`GRASP：已夹取物品 #${item.id}`);
  toast("夹取成功");
  return true;
}

function releaseItem() {
  if (sim.arm.graspedItemId == null) {
    $("#graspBadge").textContent = "夹爪已松开";
    return;
  }
  const item = sim.items.find((candidate) => candidate.id === sim.arm.graspedItemId);
  if (item) {
    item.held = false;
    item.x = clamp(
      sim.car.x + Math.cos(sim.car.yaw) * (sim.car.length / 2 + .42),
      item.size,
      sim.room.width - item.size,
    );
    item.y = clamp(
      sim.car.y + Math.sin(sim.car.yaw) * (sim.car.length / 2 + .42),
      item.size,
      sim.room.height - item.size,
    );
  }
  addLog(`RELEASE：已放下物品 #${sim.arm.graspedItemId}`);
  sim.arm.graspedItemId = null;
  $("#graspBadge").textContent = "夹爪已松开";
}

function updateHeldItem() {
  if (sim.arm.graspedItemId == null) return;
  const item = sim.items.find((candidate) => candidate.id === sim.arm.graspedItemId);
  if (!item) return;
  item.x = sim.car.x + Math.cos(sim.car.yaw) * (sim.car.length / 2 + .24);
  item.y = sim.car.y + Math.sin(sim.car.yaw) * (sim.car.length / 2 + .24);
}

function updatePickupMode() {
  if (sim.task !== "pickup" || sim.pickupViewActivated) return;
  const target = cameraDetections()[0];
  if (!target || target.distance_m > 1.2 || Math.abs(target.bearing_rad) > .45) return;
  sim.pickupViewActivated = true;
  stopMotion("已到达物品前方：底盘停车，进入机械臂阶段");
  switchView("pickup");
  toast("已切换到摄像头与机械臂视角");
}

function applyCommand(command) {
  const source = command.source ? ` · ${command.source}` : "";
  switch (command.type) {
    case "set_twist":
      setTwist(command.linear_mm_s, command.angular_mrad_s, command.ttl_ms);
      addLog(`SET_TWIST v=${command.linear_mm_s} w=${command.angular_mrad_s} ttl=${command.ttl_ms}${source}`);
      break;
    case "goto":
      setGoal(command.x_m, command.y_m);
      break;
    case "set_pose":
      if (!pointBlocked(command.x_m, command.y_m)) {
        sim.car.x = command.x_m;
        sim.car.y = command.y_m;
        sim.car.yaw = normalizeAngle(command.yaw_rad || 0);
        stopMotion(`SET_POSE → (${command.x_m.toFixed(2)}, ${command.y_m.toFixed(2)})`);
      } else {
        addLog("SET_POSE 拒绝：目标位置与障碍重叠", true);
      }
      break;
    case "reset_map":
      resetMap();
      addLog("RESET_MAP：地图已清空");
      break;
    case "select_task":
      setTask(command.task, sim.task !== command.task);
      break;
    case "set_arm_joints":
      setArmJoints(command.joints, command.duration_ms);
      break;
    case "arm_stop":
      armStop();
      break;
    case "cancel_all":
      stopMotion("CANCEL_ALL：全部任务已取消");
      armStop();
      break;
    case "stop":
      stopMotion();
      break;
    default:
      addLog(`未知指令 ${command.type}`, true);
  }
}

async function pollCommands() {
  try {
    const response = await fetch(`/api/commands?after=${sim.lastCommandId}`, { cache: "no-store" });
    const data = await response.json();
    for (const command of data.commands || []) {
      sim.lastCommandId = Math.max(sim.lastCommandId, command.id);
      applyCommand(command);
    }
    $("#connectionDot").style.background = "#46a875";
    $("#connectionText").textContent = "指令链路在线";
  } catch {
    $("#connectionDot").style.background = "#c34a3d";
    $("#connectionText").textContent = "指令链路离线";
  }
}

async function reportState() {
  try {
    await fetch("/api/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        simulator_session_id: sim.sessionId,
        online: true,
        mode: sim.running ? "running" : "editing",
        pose: { x_m: sim.car.x, y_m: sim.car.y, yaw_rad: sim.car.yaw },
        linear_mm_s: Math.round(sim.linear * 1000),
        angular_mrad_s: Math.round(sim.angular * 1000),
        commanded_linear_mm_s: Math.round(sim.commandLinear * 1000),
        commanded_angular_mrad_s: Math.round(sim.commandAngular * 1000),
        last_command_id: sim.lastCommandId,
        mapping_progress: mappingProgress(),
        collision: sim.collision,
        goal: sim.goal ? { x_m: sim.goal.x, y_m: sim.goal.y } : null,
        physics: sim.physics,
        task: sim.task || "mapping",
        joint_positions: sim.arm.positions,
        arm_moving: sim.arm.moving,
        grasped_item_id: sim.arm.graspedItemId,
        camera: {
          width: sim.camera.width,
          height: sim.camera.height,
          detections: sim.camera.detections,
        },
      }),
    });
    const scene = {
      room: sim.room,
      obstacles: sim.obstacles,
      car: {
        x: sim.car.x,
        y: sim.car.y,
        yaw: sim.car.yaw,
        length: sim.car.length,
        width: sim.car.width,
      },
      lidar: sim.lidar,
      physics: sim.physics,
      task: sim.task,
      items: sim.items,
    };
    const signature = JSON.stringify(scene);
    if (signature !== sim.lastSceneSignature) {
      await fetch("/api/scene", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: signature,
      });
      sim.lastSceneSignature = signature;
    }
  } catch {
    // 页面仍可离线运行；顶部状态会由指令轮询标记。
  }
}

async function sendCommand(command) {
  try {
    const response = await fetch("/api/commands", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...command, source: command.source || "web-console" }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "发送失败");
    toast(`指令 #${data.command.id} 已进入队列`);
  } catch (error) {
    toast(error.message);
    addLog(error.message, true);
  }
}

function drawGrid(ctx, view) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(view.x, view.y, view.width, view.height);
  ctx.clip();
  ctx.fillStyle = "#f8f7f2";
  ctx.fillRect(view.x, view.y, view.width, view.height);
  const minor = view.scale * 0.25;
  const major = view.scale;
  ctx.lineWidth = 1;
  for (let i = 0; i <= Math.ceil(sim.room.width * 4); i += 1) {
    const x = view.x + i * minor;
    ctx.strokeStyle = i % 4 === 0 ? "#d7dbd5" : "#e9eae5";
    ctx.beginPath();
    ctx.moveTo(x, view.y);
    ctx.lineTo(x, view.y + view.height);
    ctx.stroke();
  }
  for (let i = 0; i <= Math.ceil(sim.room.height * 4); i += 1) {
    const y = view.y + i * minor;
    ctx.strokeStyle = i % 4 === 0 ? "#d7dbd5" : "#e9eae5";
    ctx.beginPath();
    ctx.moveTo(view.x, y);
    ctx.lineTo(view.x + view.width, y);
    ctx.stroke();
  }
  ctx.restore();
  ctx.strokeStyle = "#293a35";
  ctx.lineWidth = 6;
  ctx.strokeRect(view.x, view.y, view.width, view.height);
  ctx.strokeStyle = "rgba(255,255,255,.7)";
  ctx.lineWidth = 1;
  ctx.strokeRect(view.x + 4, view.y + 4, view.width - 8, view.height - 8);
}

function drawObstacle(ctx, obstacle, selected = false) {
  const view = viewport(worldCanvas);
  ctx.save();
  ctx.fillStyle = selected ? "#d9a08a" : "#738079";
  ctx.strokeStyle = selected ? "#b24d2e" : "#47554f";
  ctx.lineWidth = selected ? 3 : 1.5;
  if (obstacle.type === "rect") {
    const point = worldToCanvas({ x: obstacle.x, y: obstacle.y + obstacle.h });
    ctx.fillRect(point.x, point.y, obstacle.w * view.scale, obstacle.h * view.scale);
    ctx.strokeRect(point.x, point.y, obstacle.w * view.scale, obstacle.h * view.scale);
    ctx.strokeStyle = "rgba(255,255,255,.18)";
    for (let x = point.x + 8; x < point.x + obstacle.w * view.scale; x += 14) {
      ctx.beginPath();
      ctx.moveTo(x, point.y);
      ctx.lineTo(x - obstacle.h * view.scale, point.y + obstacle.h * view.scale);
      ctx.stroke();
    }
  } else {
    const point = worldToCanvas(obstacle);
    ctx.beginPath();
    ctx.arc(point.x, point.y, obstacle.r * view.scale, 0, TAU);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(point.x - obstacle.r * view.scale * .2, point.y - obstacle.r * view.scale * .2, obstacle.r * view.scale * .62, 0, TAU);
    ctx.strokeStyle = "rgba(255,255,255,.22)";
    ctx.stroke();
  }
  ctx.restore();
}

function drawItem(ctx, item, selected = false) {
  const view = viewport(worldCanvas);
  const point = worldToCanvas(item);
  const size = item.size * view.scale;
  ctx.save();
  ctx.translate(point.x, point.y);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = item.held ? "#66b8aa" : item.color;
  ctx.strokeStyle = selected ? "#b84234" : "#6f4c19";
  ctx.lineWidth = selected ? 4 : 2;
  ctx.fillRect(-size, -size, size * 2, size * 2);
  ctx.strokeRect(-size, -size, size * 2, size * 2);
  ctx.fillStyle = "rgba(255,255,255,.52)";
  ctx.fillRect(-size * .52, -size * .52, size * .7, size * .7);
  ctx.restore();
  ctx.save();
  ctx.fillStyle = "#6f4c19";
  ctx.font = `700 ${Math.max(9, size * .58)}px "Microsoft YaHei UI"`;
  ctx.textAlign = "center";
  ctx.fillText(item.held ? "夹持中" : `物品 ${item.id}`, point.x, point.y + size * 2.15);
  ctx.restore();
}

function drawCar(ctx) {
  const view = viewport(worldCanvas);
  const center = worldToCanvas(sim.car);
  const length = sim.car.length * view.scale;
  const width = sim.car.width * view.scale;
  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(-sim.car.yaw);
  ctx.fillStyle = "rgba(20,32,28,.22)";
  ctx.fillRect(-length / 2 + 4, -width / 2 + 5, length, width);
  ctx.fillStyle = sim.collision ? "#b84234" : "#ed6a3b";
  ctx.strokeStyle = "#73311c";
  ctx.lineWidth = 2;
  roundedRect(ctx, -length / 2, -width / 2, length, width, Math.min(8, width * .22));
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#253842";
  const wheelLength = length * .22;
  const wheelWidth = Math.max(3, width * .16);
  for (const x of [-length * .28, length * .28]) {
    ctx.fillRect(x - wheelLength / 2, -width / 2 - wheelWidth * .65, wheelLength, wheelWidth);
    ctx.fillRect(x - wheelLength / 2, width / 2 - wheelWidth * .35, wheelLength, wheelWidth);
  }
  ctx.fillStyle = "#f7d7c9";
  ctx.beginPath();
  ctx.moveTo(length * .39, 0);
  ctx.lineTo(length * .17, -width * .22);
  ctx.lineTo(length * .17, width * .22);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#163a35";
  ctx.beginPath();
  ctx.arc(0, 0, Math.max(5, width * .17), 0, TAU);
  ctx.fill();
  ctx.strokeStyle = "#87d6c8";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(0, 0, Math.max(3, width * .10), 0, TAU);
  ctx.stroke();
  ctx.restore();
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function drawScans(ctx) {
  if (!sim.running || !sim.scans.length) return;
  const origin = worldToCanvas(sim.car);
  ctx.save();
  ctx.lineWidth = 0.7;
  sim.scans.forEach((scan, index) => {
    if (index % 3 !== 0) return;
    const end = worldToCanvas(scan);
    ctx.strokeStyle = scan.hit ? "rgba(31,138,121,.22)" : "rgba(31,138,121,.10)";
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
  });
  ctx.restore();
}

function drawPath(ctx) {
  const points = sim.path.slice(sim.pathIndex);
  if (!points.length) return;
  ctx.save();
  ctx.strokeStyle = "#eb6c3e";
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 7]);
  ctx.beginPath();
  let point = worldToCanvas(sim.car);
  ctx.moveTo(point.x, point.y);
  for (const waypoint of points) {
    point = worldToCanvas(waypoint);
    ctx.lineTo(point.x, point.y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawGoal(ctx) {
  if (!sim.goal) return;
  const point = worldToCanvas(sim.goal);
  ctx.save();
  ctx.strokeStyle = "#ed6a3b";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 10, 0, TAU);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(point.x - 15, point.y);
  ctx.lineTo(point.x + 15, point.y);
  ctx.moveTo(point.x, point.y - 15);
  ctx.lineTo(point.x, point.y + 15);
  ctx.stroke();
  ctx.restore();
}

function drawDraft(ctx) {
  if (!dragStart || !dragCurrent) return;
  const start = worldToCanvas(dragStart);
  const current = worldToCanvas(dragCurrent);
  ctx.save();
  if (pointerAction === "orient-car") {
    ctx.strokeStyle = "#ee6b3b";
    ctx.fillStyle = "#ee6b3b";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(current.x, current.y);
    ctx.stroke();
    const angle = Math.atan2(current.y - start.y, current.x - start.x);
    ctx.beginPath();
    ctx.moveTo(current.x, current.y);
    ctx.lineTo(current.x - Math.cos(angle - .5) * 14, current.y - Math.sin(angle - .5) * 14);
    ctx.lineTo(current.x - Math.cos(angle + .5) * 14, current.y - Math.sin(angle + .5) * 14);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    return;
  }
  if (!["rect", "circle"].includes(sim.tool)) {
    ctx.restore();
    return;
  }
  ctx.fillStyle = "rgba(238,107,59,.18)";
  ctx.strokeStyle = "#ee6b3b";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 5]);
  if (sim.tool === "rect") {
    ctx.fillRect(Math.min(start.x, current.x), Math.min(start.y, current.y), Math.abs(current.x - start.x), Math.abs(current.y - start.y));
    ctx.strokeRect(Math.min(start.x, current.x), Math.min(start.y, current.y), Math.abs(current.x - start.x), Math.abs(current.y - start.y));
  } else {
    const radius = Math.hypot(current.x - start.x, current.y - start.y);
    ctx.beginPath();
    ctx.arc(start.x, start.y, radius, 0, TAU);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function renderWorld() {
  const bounds = worldCanvas.getBoundingClientRect();
  if (bounds.width < 2 || bounds.height < 2) return;
  resizeCanvas(worldCanvas);
  const view = viewport(worldCanvas);
  worldCtx.clearRect(0, 0, worldCanvas.width, worldCanvas.height);
  worldCtx.fillStyle = "#dfe0da";
  worldCtx.fillRect(0, 0, worldCanvas.width, worldCanvas.height);
  drawGrid(worldCtx, view);
  drawScans(worldCtx);
  drawPath(worldCtx);
  for (const obstacle of sim.obstacles) drawObstacle(worldCtx, obstacle, obstacle.id === sim.selectedId);
  for (const item of sim.items) drawItem(worldCtx, item, item.id === sim.selectedId);
  drawGoal(worldCtx);
  drawDraft(worldCtx);
  drawCar(worldCtx);
}

function renderMap() {
  const bounds = mapCanvas.getBoundingClientRect();
  if (bounds.width < 2 || bounds.height < 2) return;
  resizeCanvas(mapCanvas);
  const view = viewport(mapCanvas);
  mapCtx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
  mapCtx.fillStyle = "#c7c9c4";
  mapCtx.fillRect(0, 0, mapCanvas.width, mapCanvas.height);
  const cellWidth = view.width / sim.gridWidth;
  const cellHeight = view.height / sim.gridHeight;
  for (let row = 0; row < sim.gridHeight; row += 1) {
    for (let col = 0; col < sim.gridWidth; col += 1) {
      const value = sim.grid[row * sim.gridWidth + col];
      if (!value) continue;
      mapCtx.fillStyle = value === 2 ? "#263b3b" : "#f4f3ed";
      mapCtx.fillRect(
        view.x + col * cellWidth,
        view.y + (sim.gridHeight - row - 1) * cellHeight,
        Math.ceil(cellWidth) + .5,
        Math.ceil(cellHeight) + .5,
      );
    }
  }
  mapCtx.strokeStyle = "#53625c";
  mapCtx.lineWidth = 2;
  mapCtx.strokeRect(view.x, view.y, view.width, view.height);
  if (sim.trajectory.length > 1) {
    mapCtx.strokeStyle = "#e76437";
    mapCtx.lineWidth = 2;
    mapCtx.beginPath();
    sim.trajectory.forEach((point, index) => {
      const canvasPoint = worldToCanvas(point, mapCanvas);
      if (index === 0) mapCtx.moveTo(canvasPoint.x, canvasPoint.y);
      else mapCtx.lineTo(canvasPoint.x, canvasPoint.y);
    });
    mapCtx.stroke();
  }
  const carPoint = worldToCanvas(sim.car, mapCanvas);
  mapCtx.fillStyle = "#e76437";
  mapCtx.beginPath();
  mapCtx.arc(carPoint.x, carPoint.y, 4, 0, TAU);
  mapCtx.fill();
}

function renderCamera() {
  const bounds = cameraCanvas.getBoundingClientRect();
  if (bounds.width < 2 || bounds.height < 2) return;
  resizeCanvas(cameraCanvas);
  const width = cameraCanvas.width;
  const height = cameraCanvas.height;
  const sx = width / sim.camera.width;
  const sy = height / sim.camera.height;
  const horizon = height * .42;
  const sky = cameraCtx.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, "#5f7772");
  sky.addColorStop(1, "#a9b7af");
  cameraCtx.fillStyle = sky;
  cameraCtx.fillRect(0, 0, width, horizon);
  const floor = cameraCtx.createLinearGradient(0, horizon, 0, height);
  floor.addColorStop(0, "#9c9d91");
  floor.addColorStop(1, "#393e39");
  cameraCtx.fillStyle = floor;
  cameraCtx.fillRect(0, horizon, width, height - horizon);
  cameraCtx.strokeStyle = "rgba(255,255,255,.13)";
  cameraCtx.lineWidth = Math.max(1, sx);
  for (let i = -7; i <= 7; i += 1) {
    cameraCtx.beginPath();
    cameraCtx.moveTo(width / 2, horizon);
    cameraCtx.lineTo(width / 2 + i * width * .13, height);
    cameraCtx.stroke();
  }
  for (let row = 1; row <= 5; row += 1) {
    const y = horizon + (height - horizon) * (row / 5) ** 1.7;
    cameraCtx.beginPath();
    cameraCtx.moveTo(0, y);
    cameraCtx.lineTo(width, y);
    cameraCtx.stroke();
  }

  const detections = cameraDetections();
  for (const detection of [...detections].reverse()) {
    const [x, y, boxWidth, boxHeight] = detection.bbox;
    const px = x * sx;
    const py = y * sy;
    const pw = boxWidth * sx;
    const ph = boxHeight * sy;
    cameraCtx.save();
    cameraCtx.translate(px + pw / 2, py + ph / 2);
    cameraCtx.rotate(Math.PI / 4);
    cameraCtx.fillStyle = detection.color;
    cameraCtx.strokeStyle = "#fff3c8";
    cameraCtx.lineWidth = Math.max(2, sx * 2);
    cameraCtx.fillRect(-pw * .32, -ph * .32, pw * .64, ph * .64);
    cameraCtx.strokeRect(-pw * .32, -ph * .32, pw * .64, ph * .64);
    cameraCtx.restore();
    cameraCtx.strokeStyle = "#62e3c6";
    cameraCtx.lineWidth = Math.max(2, sx * 2);
    cameraCtx.strokeRect(px, py, pw, ph);
    cameraCtx.fillStyle = "#17342e";
    cameraCtx.fillRect(px, Math.max(0, py - 22 * sy), Math.min(pw, 188 * sx), 22 * sy);
    cameraCtx.fillStyle = "#bff8ea";
    cameraCtx.font = `${Math.max(10, 11 * sy)}px Consolas, monospace`;
    cameraCtx.fillText(
      `ITEM #${detection.id} ${(detection.confidence * 100).toFixed(0)}%`,
      px + 5 * sx,
      Math.max(12 * sy, py - 7 * sy),
    );
  }

  const target = detections[0];
  $("#cameraStatus").textContent = target ? "检测到物品" : "等待物品";
  $("#cameraTarget").textContent = target ? `#${target.id}` : "无";
  $("#cameraDistance").textContent = target ? `${target.distance_m.toFixed(2)} m` : "--";
  $("#cameraBearing").textContent = target ? `${(target.bearing_rad * 180 / Math.PI).toFixed(1)}°` : "--";
  $("#cameraConfidence").textContent = target ? `${Math.round(target.confidence * 100)}%` : "--";
}

function renderArm() {
  const bounds = armCanvas.getBoundingClientRect();
  if (bounds.width < 2 || bounds.height < 2) return;
  resizeCanvas(armCanvas);
  const width = armCanvas.width;
  const height = armCanvas.height;
  const positions = sim.arm.positions;
  armCtx.clearRect(0, 0, width, height);
  armCtx.fillStyle = "#e5e6df";
  armCtx.fillRect(0, 0, width, height);
  armCtx.strokeStyle = "rgba(61,79,72,.10)";
  armCtx.lineWidth = 1;
  for (let x = 0; x < width; x += width / 12) {
    armCtx.beginPath(); armCtx.moveTo(x, 0); armCtx.lineTo(x, height); armCtx.stroke();
  }
  for (let y = 0; y < height; y += height / 8) {
    armCtx.beginPath(); armCtx.moveTo(0, y); armCtx.lineTo(width, y); armCtx.stroke();
  }

  const base = { x: width * .22, y: height * .83 };
  const scale = Math.min(width / 480, height / 260);
  const linkLengths = [76, 66, 53].map((value) => value * scale);
  const joint1 = -1.25 + ((positions[1] - 800) / 900) * .62;
  const joint2 = .90 - ((positions[2] - 1500) / 700) * 1.15;
  const joint3 = .42 - ((positions[3] - 800) / 700) * .82;
  const angles = [joint1, joint1 + joint2, joint1 + joint2 + joint3];
  const points = [base];
  for (let index = 0; index < 3; index += 1) {
    const previous = points.at(-1);
    points.push({
      x: previous.x + Math.cos(angles[index]) * linkLengths[index],
      y: previous.y + Math.sin(angles[index]) * linkLengths[index],
    });
  }

  armCtx.fillStyle = "#263a35";
  armCtx.fillRect(base.x - 36 * scale, base.y + 7 * scale, 72 * scale, 18 * scale);
  armCtx.fillStyle = "#ee6b3b";
  armCtx.beginPath();
  armCtx.ellipse(base.x, base.y + 4 * scale, 29 * scale, 12 * scale, 0, 0, TAU);
  armCtx.fill();
  armCtx.strokeStyle = "#30443e";
  armCtx.lineWidth = Math.max(8, 13 * scale);
  armCtx.lineCap = "round";
  for (let index = 0; index < 3; index += 1) {
    armCtx.beginPath();
    armCtx.moveTo(points[index].x, points[index].y);
    armCtx.lineTo(points[index + 1].x, points[index + 1].y);
    armCtx.stroke();
  }
  points.forEach((point, index) => {
    armCtx.fillStyle = index === 0 ? "#ee6b3b" : "#f5b28e";
    armCtx.beginPath();
    armCtx.arc(point.x, point.y, Math.max(6, 9 * scale), 0, TAU);
    armCtx.fill();
    armCtx.strokeStyle = "#30443e";
    armCtx.lineWidth = Math.max(2, 3 * scale);
    armCtx.stroke();
  });

  const wrist = points.at(-1);
  const wristAngle = angles.at(-1) + ((positions[4] - 1500) / 600) * .7;
  const clawGap = clamp((1550 - positions[5]) / 350, 0, 1) * 19 * scale + 4 * scale;
  armCtx.save();
  armCtx.translate(wrist.x, wrist.y);
  armCtx.rotate(wristAngle);
  armCtx.fillStyle = "#ee6b3b";
  armCtx.fillRect(-2 * scale, -10 * scale, 24 * scale, 20 * scale);
  armCtx.strokeStyle = "#263a35";
  armCtx.lineWidth = Math.max(3, 5 * scale);
  armCtx.beginPath();
  armCtx.moveTo(20 * scale, -3 * scale);
  armCtx.lineTo(39 * scale, -clawGap);
  armCtx.moveTo(20 * scale, 3 * scale);
  armCtx.lineTo(39 * scale, clawGap);
  armCtx.stroke();
  if (sim.arm.graspedItemId != null) {
    armCtx.fillStyle = "#e7a33f";
    armCtx.translate(44 * scale, 0);
    armCtx.rotate(Math.PI / 4);
    armCtx.fillRect(-10 * scale, -10 * scale, 20 * scale, 20 * scale);
  }
  armCtx.restore();

  armCtx.fillStyle = "#4b5b55";
  armCtx.font = `${Math.max(10, 11 * scale)}px Consolas, monospace`;
  armCtx.fillText(`BASE YAW ${(positions[0] - 1500) / 10}°`, width * .55, height * .84);
  armCtx.fillText(`WRIST ${(positions[4] - 1500) / 10}°`, width * .55, height * .91);
  positions.forEach((position, index) => {
    $(`#joint${index}`).textContent = String(position);
  });
  if (sim.arm.moving) $("#graspBadge").textContent = "机械臂运动中";
  else if (sim.arm.graspedItemId != null) $("#graspBadge").textContent = `已夹取 #${sim.arm.graspedItemId}`;
  else if (positions[5] >= 1450) $("#graspBadge").textContent = "夹爪已闭合";
  else $("#graspBadge").textContent = "夹爪空闲";
}

function mappingProgress() {
  if (!sim.grid?.length) return 0;
  let known = 0;
  sim.grid.forEach((value) => { if (value) known += 1; });
  return known / sim.grid.length;
}

function updateMapStats() {
  let occupied = 0;
  sim.grid?.forEach((value) => { if (value === 2) occupied += 1; });
  $("#mapProgress").textContent = `${Math.round(mappingProgress() * 100)}%`;
  $("#occupiedCount").textContent = String(occupied);
  $("#pathDistance").textContent = `${sim.travelled.toFixed(1)} m`;
}

function updateTelemetry() {
  $("#poseX").textContent = `${sim.car.x.toFixed(2)} m`;
  $("#poseY").textContent = `${sim.car.y.toFixed(2)} m`;
  $("#poseYaw").textContent = `${(sim.car.yaw * 180 / Math.PI).toFixed(1)}°`;
  $("#commandLinearSpeed").textContent = `${Math.round(sim.commandLinear * 1000)} mm/s`;
  $("#linearSpeed").textContent = `${Math.round(sim.linear * 1000)} mm/s`;
  $("#commandAngularSpeed").textContent = `${Math.round(sim.commandAngular * 1000)} mrad/s`;
  $("#angularSpeed").textContent = `${Math.round(sim.angular * 1000)} mrad/s`;
}

function animationLoop(now) {
  let remainingDt = Math.min((now - lastFrameAt) / 1000, 1.0);
  lastFrameAt = now;
  // 后台标签页的 requestAnimationFrame 可能降到约 1 Hz。按 50 ms 子步补算
  // 真实经过时间，避免脚本速度在网页失焦后变成原来的几十分之一。
  while (remainingDt > 0) {
    const step = Math.min(remainingDt, 0.05);
    applyMotion(step);
    remainingDt -= step;
  }
  updateArm(now);
  updateHeldItem();
  updatePickupMode();
  performScan(now);
  renderWorld();
  renderMap();
  renderCamera();
  renderArm();
  updateTelemetry();
  if (now - sim.lastStateAt > 500) {
    sim.lastStateAt = now;
    reportState();
  }
  requestAnimationFrame(animationLoop);
}

function obstacleAt(point) {
  for (let i = sim.items.length - 1; i >= 0; i -= 1) {
    const item = sim.items[i];
    if (!item.held && Math.hypot(point.x - item.x, point.y - item.y) <= item.size * 1.35) return item;
  }
  for (let i = sim.obstacles.length - 1; i >= 0; i -= 1) {
    const obstacle = sim.obstacles[i];
    if (obstacle.type === "circle" && Math.hypot(point.x - obstacle.x, point.y - obstacle.y) <= obstacle.r) return obstacle;
    if (
      obstacle.type === "rect"
      && point.x >= obstacle.x && point.x <= obstacle.x + obstacle.w
      && point.y >= obstacle.y && point.y <= obstacle.y + obstacle.h
    ) return obstacle;
  }
  return null;
}

function canEditScene() {
  return document.body.classList.contains("view-scene")
    && sim.workflow === "scene"
    && !sim.running;
}

function selectedItem() {
  return sim.items.find((item) => item.id === sim.selectedId && !item.held) || null;
}

function updateItemInspector() {
  const item = selectedItem();
  const picker = $("#itemSelect");
  picker.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = sim.items.some((candidate) => !candidate.held)
    ? "请选择"
    : "暂无物品";
  picker.appendChild(placeholder);
  for (const candidate of sim.items.filter((entry) => !entry.held)) {
    const option = document.createElement("option");
    option.value = String(candidate.id);
    option.textContent = `物品 #${candidate.id}`;
    picker.appendChild(option);
  }
  picker.value = item ? String(item.id) : "";
  $("#itemInspector").classList.toggle("unavailable", !item);
  $("#selectedItemLabel").textContent = item ? `已选中物品 #${item.id}` : "未选中物品";
  $("#itemInspector > div:first-child span").textContent = item
    ? "修改后立即反映到场景和摄像头"
    : "先用“选择”点击一个物品";
  $("#itemSize").disabled = !item;
  $("#itemSmallerBtn").disabled = !item;
  $("#itemLargerBtn").disabled = !item;
  if (item) $("#itemSize").value = item.size.toFixed(2);
}

function itemSizeFits(item, size) {
  if (
    item.x - size < 0
    || item.y - size < 0
    || item.x + size > sim.room.width
    || item.y + size > sim.room.height
  ) return false;
  if (Math.hypot(item.x - sim.car.x, item.y - sim.car.y) < size + carRadius()) return false;
  for (const other of sim.items) {
    if (other.id !== item.id && !other.held && Math.hypot(item.x - other.x, item.y - other.y) < size + other.size) {
      return false;
    }
  }
  return !sim.obstacles.some((obstacle) => {
    if (obstacle.type === "circle") {
      return Math.hypot(item.x - obstacle.x, item.y - obstacle.y) < size + obstacle.r;
    }
    const closestX = clamp(item.x, obstacle.x, obstacle.x + obstacle.w);
    const closestY = clamp(item.y, obstacle.y, obstacle.y + obstacle.h);
    return Math.hypot(item.x - closestX, item.y - closestY) < size;
  });
}

function setSelectedItemSize(value) {
  if (!canEditScene()) return toast("只能在场景布置阶段修改物品");
  const item = selectedItem();
  if (!item) return toast("请先选中一个物品");
  const size = clamp(Number(value), .08, .6);
  if (!itemSizeFits(item, size)) {
    $("#itemSize").value = item.size.toFixed(2);
    return toast("该尺寸会与墙体、小车或障碍物重叠");
  }
  item.size = size;
  $("#itemSize").value = item.size.toFixed(2);
  toast(`物品 #${item.id} 尺寸已调整为 ${item.size.toFixed(2)} m`);
}

worldCanvas.addEventListener("pointerdown", (event) => {
  const point = canvasToWorld(event);
  if (!canEditScene()) {
    if (sim.workflow === "execute" && sim.running) {
      setGoal(point.x, point.y);
    } else {
      toast("当前页面不能布置场景，请按步骤进入“场景布置”");
    }
    return;
  }
  worldCanvas.setPointerCapture(event.pointerId);
  if (sim.running) {
    setGoal(point.x, point.y);
    return;
  }
  if (sim.tool === "select") {
    const obstacle = obstacleAt(point);
    sim.selectedId = obstacle?.id ?? null;
    updateItemInspector();
    if (obstacle) {
      pointerAction = "move";
      dragOffset = { x: point.x - obstacle.x, y: point.y - obstacle.y };
    }
  } else if (sim.tool === "car") {
    if (!pointBlocked(point.x, point.y)) {
      sim.car.x = point.x;
      sim.car.y = point.y;
      pointerAction = "orient-car";
      dragStart = point;
      dragCurrent = point;
    } else toast("该位置放不下小车");
  } else if (sim.tool === "item") {
    const item = {
      id: sim.nextItemId++,
      type: "item",
      x: point.x,
      y: point.y,
      size: .22,
      color: "#e7a33f",
      held: false,
    };
    if (!pointBlocked(point.x, point.y, -carRadius() + item.size)) {
      sim.items.push(item);
      sim.selectedId = item.id;
      updateItemInspector();
      toast(`已放置物品 #${item.id}`);
    } else {
      toast("该位置与墙体或障碍物重叠");
    }
  } else if (sim.tool === "goal") {
    prepareGoal(point.x, point.y);
  } else {
    pointerAction = "draw";
    dragStart = point;
    dragCurrent = point;
  }
});

worldCanvas.addEventListener("pointermove", (event) => {
  if (!canEditScene() && pointerAction) return;
  const point = canvasToWorld(event);
  if (pointerAction === "draw") {
    dragCurrent = point;
  } else if (pointerAction === "orient-car") {
    dragCurrent = point;
    const dx = point.x - dragStart.x;
    const dy = point.y - dragStart.y;
    if (Math.hypot(dx, dy) > .04) {
      sim.car.yaw = normalizeAngle(Math.atan2(dy, dx));
      $("#carYaw").value = (sim.car.yaw * 180 / Math.PI).toFixed(0);
    }
  } else if (pointerAction === "move" && sim.selectedId) {
    const selected = sim.obstacles.find((item) => item.id === sim.selectedId)
      || sim.items.find((item) => item.id === sim.selectedId);
    if (!selected) return;
    if (selected.type === "rect") {
      selected.x = clamp(point.x - dragOffset.x, 0, sim.room.width - selected.w);
      selected.y = clamp(point.y - dragOffset.y, 0, sim.room.height - selected.h);
    } else if (selected.type === "item") {
      selected.x = clamp(point.x - dragOffset.x, selected.size, sim.room.width - selected.size);
      selected.y = clamp(point.y - dragOffset.y, selected.size, sim.room.height - selected.size);
    } else {
      selected.x = clamp(point.x - dragOffset.x, selected.r, sim.room.width - selected.r);
      selected.y = clamp(point.y - dragOffset.y, selected.r, sim.room.height - selected.r);
    }
  }
});

function finishPointer(event) {
  if (event?.pointerId != null && worldCanvas.hasPointerCapture(event.pointerId)) {
    worldCanvas.releasePointerCapture(event.pointerId);
  }
  if (pointerAction === "draw" && dragStart && dragCurrent) {
    if (sim.tool === "rect") {
      const obstacle = {
        id: sim.nextObstacleId++,
        type: "rect",
        x: Math.min(dragStart.x, dragCurrent.x),
        y: Math.min(dragStart.y, dragCurrent.y),
        w: Math.abs(dragCurrent.x - dragStart.x),
        h: Math.abs(dragCurrent.y - dragStart.y),
      };
      if (obstacle.w >= 0.12 && obstacle.h >= 0.12) {
        sim.obstacles.push(obstacle);
        sim.selectedId = obstacle.id;
      }
    } else if (sim.tool === "circle") {
      const radius = Math.hypot(dragCurrent.x - dragStart.x, dragCurrent.y - dragStart.y);
      if (radius >= 0.1) {
        const obstacle = {
          id: sim.nextObstacleId++,
          type: "circle",
          x: dragStart.x,
          y: dragStart.y,
          r: Math.min(radius, dragStart.x, dragStart.y, sim.room.width - dragStart.x, sim.room.height - dragStart.y),
        };
        sim.obstacles.push(obstacle);
        sim.selectedId = obstacle.id;
      }
    }
  } else if (pointerAction === "orient-car") {
    resetMap();
    toast(`小车起点和朝向已更新：${(sim.car.yaw * 180 / Math.PI).toFixed(0)}°`);
  }
  pointerAction = null;
  dragStart = null;
  dragCurrent = null;
  dragOffset = null;
}

worldCanvas.addEventListener("pointerup", finishPointer);
worldCanvas.addEventListener("pointercancel", finishPointer);

function selectTool(tool, force = false) {
  if (!force && !canEditScene()) {
    toast("只能在场景布置阶段选择编辑工具");
    return;
  }
  sim.tool = tool;
  $$(".tool[data-tool]").forEach((button) => button.classList.toggle("active", button.dataset.tool === tool));
  const tips = {
    select: "点击障碍物选择并拖动",
    rect: "拖拽绘制矩形障碍",
    circle: "从圆心向外拖拽",
    car: "按下确定小车位置，拖向车头方向",
    item: "点击放置一个可夹取物品",
    goal: "点击设置导航目标",
  };
  $("#canvasTip").textContent = sim.running ? "运行中：点击场景可下发目标点" : tips[tool];
}

$$(".tool[data-tool]").forEach((button) => button.addEventListener("click", () => selectTool(button.dataset.tool)));

$$("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.view === "tasks" && sim.workflow !== "task") {
      stopMotion("返回任务选择：当前运动已停止");
      armStop();
      setRunning(false);
      setWorkflow("task");
      switchView("tasks", true);
      return;
    }
    switchView(button.dataset.view);
  });
});

$$("[data-select-task]").forEach((button) => {
  button.addEventListener("click", () => {
    const task = button.dataset.selectTask;
    setTask(task, true, true);
    sendCommand({ type: "select_task", task });
    if (task === "pickup") selectTool("item", true);
    switchView("scene");
  });
});

const armActions = {
  home: { joints: [[0, 1500], [1, 1700], [2, 2000], [3, 1100], [4, 1500], [5, 1200]], duration_ms: 1200 },
  left: { joints: [[0, 2200]], duration_ms: 800 },
  right: { joints: [[0, 800]], duration_ms: 800 },
  down: { joints: [[1, 1200], [2, 2100], [3, 1000]], duration_ms: 1000 },
  release: { joints: [[5, 1200]], duration_ms: 500 },
  grip: { joints: [[5, 1500]], duration_ms: 500 },
};

$$("[data-arm-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = armActions[button.dataset.armAction];
    if (action) sendCommand({ type: "set_arm_joints", ...action });
  });
});

$$("[data-editor-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    const section = button.dataset.editorTab;
    $$("[data-editor-tab]").forEach((item) => item.classList.toggle("active", item === button));
    $$("[data-editor-section]").forEach((item) => {
      item.classList.toggle("active", item.dataset.editorSection === section);
    });
  });
});

$("#editorToggleBtn").addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("editor-collapsed");
  $("#editorToggleBtn").textContent = collapsed ? "打开面板" : "隐藏面板";
  $("#editorToggleBtn").setAttribute("aria-expanded", String(!collapsed));
  requestAnimationFrame(renderWorld);
});

$("#deleteBtn").addEventListener("click", () => {
  if (!canEditScene()) return toast("只能在场景布置阶段删除");
  if (sim.selectedId == null) return toast("请先选择障碍物或物品");
  sim.obstacles = sim.obstacles.filter((item) => item.id !== sim.selectedId);
  sim.items = sim.items.filter((item) => item.id !== sim.selectedId || item.held);
  sim.selectedId = null;
  updateItemInspector();
});

function setRunning(running) {
  sim.running = running;
  $("#runBtn").textContent = "结束任务";
  $("#modeBadge").textContent = sim.workflow === "execute" ? "场景已锁定" : "编辑中";
  $("#modeBadge").style.background = sim.workflow === "execute" ? "#fde3d8" : "";
  $("#modeBadge").style.color = sim.workflow === "execute" ? "#a54220" : "";
  $("#canvasTip").textContent = sim.workflow === "execute"
    ? "任务运行中：场景已锁定，点击可下发导航目标"
    : "选择左侧工具编辑环境";
  if (!running) {
    sim.commandLinear = 0;
    sim.commandAngular = 0;
    sim.linear = 0;
    sim.angular = 0;
    sim.pendingDrive = null;
    sim.commandDeadline = 0;
  }
}

$("#confirmSceneBtn").addEventListener("click", () => beginExecution(true));
$("#runBtn").addEventListener("click", endExecution);

$("#applyRoomBtn").addEventListener("click", () => {
  if (!canEditScene()) return toast("只能在场景布置阶段调整房间");
  const width = clamp(Number($("#roomWidth").value), 3, 20);
  const height = clamp(Number($("#roomHeight").value), 3, 15);
  sim.room = { width, height };
  sim.obstacles = sim.obstacles.filter((obstacle) => {
    if (obstacle.type === "circle") return obstacle.x + obstacle.r <= width && obstacle.y + obstacle.r <= height;
    return obstacle.x + obstacle.w <= width && obstacle.y + obstacle.h <= height;
  });
  sim.items = sim.items.filter((item) => item.held || (
    item.x + item.size <= width && item.y + item.size <= height
  ));
  if (pointBlocked(sim.car.x, sim.car.y)) {
    sim.car.x = carRadius() + .2;
    sim.car.y = carRadius() + .2;
  }
  stopMotion("房间尺寸已更新");
  resetMap();
  if (!selectedItem()) sim.selectedId = null;
  updateItemInspector();
  toast(`房间已调整为 ${width} × ${height} m`);
});

function syncCarConfig() {
  sim.car.length = clamp(Number($("#carLength").value), .2, 1.5);
  sim.car.width = clamp(Number($("#carWidth").value), .2, 1.2);
  sim.lidar.fovDeg = clamp(Number($("#lidarFov").value), 60, 360);
  sim.lidar.range = clamp(Number($("#lidarRange").value), 1, 12);
}

["#carLength", "#carWidth", "#lidarFov", "#lidarRange"].forEach((selector) => {
  $(selector).addEventListener("change", () => {
    if (!canEditScene()) return toast("只能在场景布置阶段修改小车参数");
    syncCarConfig();
  });
});

function setCarYawDegrees(value) {
  if (!canEditScene()) return toast("只能在场景布置阶段调整朝向");
  const degrees = clamp(Number(value), -180, 180);
  sim.car.yaw = normalizeAngle(degrees * Math.PI / 180);
  $("#carYaw").value = (sim.car.yaw * 180 / Math.PI).toFixed(0);
}

$("#carYaw").addEventListener("change", () => setCarYawDegrees($("#carYaw").value));
$$("[data-yaw-step]").forEach((button) => {
  button.addEventListener("click", () => {
    const current = sim.car.yaw * 180 / Math.PI;
    setCarYawDegrees(current + Number(button.dataset.yawStep));
  });
});
$("#yawResetBtn").addEventListener("click", () => setCarYawDegrees(0));

$("#itemSize").addEventListener("change", () => setSelectedItemSize($("#itemSize").value));
$("#itemSelect").addEventListener("change", () => {
  if (!canEditScene()) return toast("只能在场景布置阶段选择物品");
  const value = $("#itemSelect").value;
  sim.selectedId = value ? Number(value) : null;
  selectTool("select", true);
  updateItemInspector();
});
$("#itemSmallerBtn").addEventListener("click", () => {
  const item = selectedItem();
  if (item) setSelectedItemSize(item.size - .04);
});
$("#itemLargerBtn").addEventListener("click", () => {
  const item = selectedItem();
  if (item) setSelectedItemSize(item.size + .04);
});

const physicsPresets = {
  ideal: {
    leftScale: 1,
    rightScale: 1,
    linearSlip: 0,
    angularBias: 0,
    speedNoise: 0,
    commandLatencyMs: 0,
    lidarNoiseM: 0,
    lidarDropout: 0,
  },
  mild: {
    leftScale: .985,
    rightScale: 1.015,
    linearSlip: .03,
    angularBias: .015,
    speedNoise: .012,
    commandLatencyMs: 80,
    lidarNoiseM: .012,
    lidarDropout: .01,
  },
  strong: {
    leftScale: .92,
    rightScale: 1.06,
    linearSlip: .14,
    angularBias: .06,
    speedNoise: .035,
    commandLatencyMs: 260,
    lidarNoiseM: .045,
    lidarDropout: .09,
  },
};

function writePhysicsInputs() {
  $("#leftScale").value = sim.physics.leftScale;
  $("#rightScale").value = sim.physics.rightScale;
  $("#commandLatency").value = sim.physics.commandLatencyMs;
  $("#linearSlip").value = sim.physics.linearSlip;
  $("#lidarNoise").value = sim.physics.lidarNoiseM;
  $("#lidarDropout").value = sim.physics.lidarDropout;
}

function syncPhysicsConfig() {
  sim.physics.leftScale = clamp(Number($("#leftScale").value), .7, 1.3);
  sim.physics.rightScale = clamp(Number($("#rightScale").value), .7, 1.3);
  sim.physics.commandLatencyMs = clamp(Number($("#commandLatency").value), 0, 1000);
  sim.physics.linearSlip = clamp(Number($("#linearSlip").value), 0, .5);
  sim.physics.lidarNoiseM = clamp(Number($("#lidarNoise").value), 0, .2);
  sim.physics.lidarDropout = clamp(Number($("#lidarDropout").value), 0, .5);
  $("#errorPreset").value = "custom";
}

$("#errorPreset").addEventListener("change", (event) => {
  if (!canEditScene()) return toast("只能在场景布置阶段修改偏差参数");
  const preset = physicsPresets[event.target.value];
  if (!preset) return;
  Object.assign(sim.physics, preset);
  writePhysicsInputs();
  toast(`已应用${event.target.options[event.target.selectedIndex].text}`);
});

["#leftScale", "#rightScale", "#commandLatency", "#linearSlip", "#lidarNoise", "#lidarDropout"].forEach((selector) => {
  $(selector).addEventListener("change", () => {
    if (!canEditScene()) return toast("只能在场景布置阶段修改偏差参数");
    syncPhysicsConfig();
  });
});

function driveCommand(direction) {
  const commands = {
    forward: [350, 0],
    backward: [-300, 0],
    left: [0, 2500],
    right: [0, -2500],
  };
  if (direction === "stop") return stopMotion("手动停车");
  const [linear, angular] = commands[direction];
  setTwist(linear, angular, 2500);
}

$$("[data-drive]").forEach((button) => {
  button.addEventListener("pointerdown", () => driveCommand(button.dataset.drive));
  button.addEventListener("pointerup", () => stopMotion("手动控制松开，停车"));
  button.addEventListener("pointerleave", () => {
    if (sim.linear || sim.angular) stopMotion("手动控制松开，停车");
  });
});

const keys = { w: "forward", a: "left", s: "backward", d: "right" };
window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) return;
  const direction = keys[event.key.toLowerCase()];
  if (direction && !event.repeat) {
    event.preventDefault();
    driveCommand(direction);
  }
});
window.addEventListener("keyup", (event) => {
  if (keys[event.key.toLowerCase()]) stopMotion("键盘松开，停车");
});

$("#resetMapBtn").addEventListener("click", () => {
  resetMap();
  toast("实时地图已清空");
});

$("#saveSceneBtn").addEventListener("click", () => {
  if (!canEditScene()) return toast("请在场景布置阶段保存");
  localStorage.setItem("car-sim-scene", JSON.stringify({
    room: sim.room,
    car: { ...sim.car },
    obstacles: sim.obstacles,
    items: sim.items,
    lidar: sim.lidar,
    physics: sim.physics,
    task: sim.task,
  }));
  toast("场景已保存在本机浏览器");
});

$("#loadSceneBtn").addEventListener("click", () => {
  if (!canEditScene()) return toast("只能在场景布置阶段载入");
  const saved = localStorage.getItem("car-sim-scene");
  if (!saved) return toast("还没有保存过场景");
  try {
    const scene = JSON.parse(saved);
    sim.room = scene.room;
    Object.assign(sim.car, scene.car);
    sim.obstacles = scene.obstacles;
    sim.items = scene.items || sim.items;
    sim.lidar = scene.lidar || sim.lidar;
    sim.physics = { ...sim.physics, ...(scene.physics || {}) };
    if (scene.task) setTask(scene.task, false);
    sim.nextObstacleId = Math.max(0, ...sim.obstacles.map((item) => item.id)) + 1;
    sim.nextItemId = Math.max(1000, ...sim.items.map((item) => item.id)) + 1;
    $("#roomWidth").value = sim.room.width;
    $("#roomHeight").value = sim.room.height;
    $("#carLength").value = sim.car.length;
    $("#carWidth").value = sim.car.width;
    $("#lidarFov").value = sim.lidar.fovDeg;
    $("#lidarRange").value = sim.lidar.range;
    $("#carYaw").value = (sim.car.yaw * 180 / Math.PI).toFixed(0);
    $("#errorPreset").value = "custom";
    writePhysicsInputs();
    stopMotion("本机场景已载入");
    resetMap();
    sim.selectedId = null;
    updateItemInspector();
    toast("场景载入成功");
  } catch {
    toast("保存的场景数据无效");
  }
});

$("#sendCommandBtn").addEventListener("click", () => {
  try {
    sendCommand(JSON.parse($("#commandInput").value));
  } catch {
    toast("JSON 格式有误");
  }
});

const quickCommands = {
  forward: { type: "set_twist", linear_mm_s: 350, angular_mrad_s: 0, ttl_ms: 600 },
  spin: { type: "set_twist", linear_mm_s: 0, angular_mrad_s: 2500, ttl_ms: 500 },
  demo: { type: "goto", x_m: 6.4, y_m: 3.8 },
  stop: { type: "stop" },
};

$$("[data-command]").forEach((button) => {
  button.addEventListener("click", () => sendCommand(quickCommands[button.dataset.command]));
});

$("#clearLogBtn").addEventListener("click", () => { $("#commandLog").innerHTML = ""; });

function addLog(message, error = false) {
  const item = document.createElement("li");
  if (error) item.className = "error";
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  item.textContent = message;
  item.appendChild(time);
  $("#commandLog").prepend(item);
  while ($("#commandLog").children.length > 20) $("#commandLog").lastElementChild.remove();
}

function toast(message) {
  $("#toast").textContent = message;
  $("#toast").classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("#toast").classList.remove("show"), 2200);
}

async function startSimulatorPage() {
  await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ simulator_session_id: sim.sessionId }),
  });
  resetMap();
  syncCarConfig();
  writePhysicsInputs();
  setWorkflow("task");
  updateItemInspector();
  addLog("仿真器就绪，等待指令");
  setInterval(pollCommands, 160);
  pollCommands();
  requestAnimationFrame(animationLoop);
}

startSimulatorPage().catch((error) => {
  addLog(`Simulator session startup failed: ${error}`, true);
});
