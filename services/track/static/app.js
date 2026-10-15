const $ = id => document.getElementById(id);
const DEEPSEEK_FLASH_KEY = 'final_work.deepseek_flash_api_key';
const DRAFT_PREFIX = 'final_work.chat_draft.';

const state = {
  conversationId: null,
  task: null,
  robot: null,
  pose: null,
  mapMeta: { resolution_m: 0.05, origin: [-10, -10, 0] },
  mapLayers: [],
  activeMapLayer: null,
  settings: null,
  followLatest: true,
  actionBusy: false,
  mapView: { scale: 1.35, panX: 0, panY: 0, dragging: false, pointerId: null, x: 0, y: 0 },
};

const translations = {
  'zh-CN': {
    newTask: '＋ 新建任务', history: '聊天历史', settings: '设置', logs: '运行日志',
    emergencyStop: '紧急停止', map: '地图', stereo: '双目视觉', mapTitle: '雷达地图与定位',
    recenter: '重新居中', refresh: '刷新', liveState: '实时状态', heading: '航向',
    quality: '定位质量', semanticLayer: '语义图层', startMapping: '开始建图', stop: '停止',
    stereoTitle: '双目实时视觉', startPreview: '启动预览', leftCamera: '左摄像头',
    rightCamera: '右摄像头', tracking: '追踪', grasp: '抓取', controlMode: '控制模式',
    settingsTitle: 'Agent 设置', language: '界面语言', plannerProvider: '规划模型',
    modelId: '模型 ID', apiKey: 'API 密钥', saveSettings: '保存设置',
  },
  'en-US': {
    newTask: '+ New task', history: 'Chat history', settings: 'Settings', logs: 'Logs',
    emergencyStop: 'Emergency stop', map: 'Map', stereo: 'Stereo vision', mapTitle: 'Lidar map & localization',
    recenter: 'Recenter', refresh: 'Refresh', liveState: 'Live status', heading: 'Heading',
    quality: 'Localization', semanticLayer: 'Semantic layer', startMapping: 'Start mapping', stop: 'Stop',
    stereoTitle: 'Live stereo vision', startPreview: 'Start preview', leftCamera: 'Left camera',
    rightCamera: 'Right camera', tracking: 'Tracking', grasp: 'Grasp', controlMode: 'Control mode',
    settingsTitle: 'Agent settings', language: 'Interface language', plannerProvider: 'Planning model',
    modelId: 'Model ID', apiKey: 'API key', saveSettings: 'Save settings',
  },
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  let value = {};
  try { value = await response.json(); }
  catch { value = { error: await response.text() }; }
  if (!response.ok) throw new Error(value.detail || value.error || `HTTP ${response.status}`);
  return value;
}

function toast(message, error = false) {
  const node = $('toast');
  node.textContent = message;
  node.className = `toast show${error ? ' error' : ''}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.className = 'toast'; }, 2800);
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

function timeText(ms) {
  if (!ms) return '';
  return new Date(ms).toLocaleString(state.settings?.language || 'zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  });
}

async function loadConversations() {
  const result = await api('/api/conversations');
  const box = $('conversations');
  box.innerHTML = result.conversations.map(item => `
    <div class="conversation ${item.id === state.conversationId ? 'active' : ''}" data-id="${item.id}">
      ${esc(item.title)}<small>${item.message_count || 0} 条消息 · ${timeText(item.updated_at)}</small>
    </div>`).join('');
  box.querySelectorAll('.conversation').forEach(node => {
    node.onclick = () => openConversation(node.dataset.id);
  });
  if (!state.conversationId && result.conversations.length) {
    await openConversation(result.conversations[0].id);
  }
}

async function newConversation() {
  saveCurrentDraft();
  $('chatInput').value = '';
  const result = await api('/api/conversations', { method: 'POST', body: JSON.stringify({ title: '新任务' }) });
  state.conversationId = result.conversation.id;
  state.task = null;
  await loadConversations();
  await openConversation(state.conversationId);
  $('chatInput').focus();
}

async function openConversation(id) {
  saveCurrentDraft();
  state.conversationId = id;
  const result = await api(`/api/conversations/${id}`);
  $('conversationTitle').textContent = result.conversation.title;
  $('messages').innerHTML = result.messages.length
    ? result.messages.map(messageHtml).join('')
    : '<div class="empty-state"><div class="empty-icon">⌁</div><h2>告诉我希望小车完成什么</h2><p>聊天记录会持久保存，模型计划需要确认后执行。</p></div>';
  state.task = result.tasks[0] || null;
  renderTask();
  await loadConversations();
  scrollMessages(true);
  restoreDraft();
  closeResponsivePanels();
}

function messageHtml(message) {
  return `<div class="message ${message.role}"><div class="bubble">${esc(message.content)}</div><div class="message-meta">${message.role === 'user' ? '你' : 'Robot Agent'} · ${timeText(message.created_at)}</div></div>`;
}

function scrollMessages(force = false) {
  const box = $('messages');
  if (force || state.followLatest) {
    box.scrollTo({ top: box.scrollHeight, behavior: force ? 'auto' : 'smooth' });
    state.followLatest = true;
    $('scrollLatest').hidden = true;
  }
}

function handleMessageScroll() {
  const box = $('messages');
  const distance = box.scrollHeight - box.scrollTop - box.clientHeight;
  state.followLatest = distance < 80;
  $('scrollLatest').hidden = state.followLatest;
}

async function sendMessage(text) {
  if (!state.conversationId) await newConversation();
  $('sendButton').disabled = true;
  $('sendButton').textContent = '…';
  $('chatInput').disabled = true;
  $('phaseText').textContent = '模型正在规划…';
  const messages = $('messages');
  messages.querySelector('.empty-state')?.remove();
  messages.insertAdjacentHTML('beforeend', `${messageHtml({ role: 'user', content: text, created_at: Date.now() })}
    <div class="message assistant planning-indicator"><div class="bubble"><span class="planning-dots"><i></i><i></i><i></i></span>正在理解指令并生成可执行计划</div></div>`);
  scrollMessages(true);
  localStorage.removeItem(`${DRAFT_PREFIX}${state.conversationId}`);
  try {
    let result;
    if (state.settings?.provider === 'deepseek_flash_api') {
      const key = localStorage.getItem(DEEPSEEK_FLASH_KEY) || '';
      if (!key) throw new Error('请先在设置中填写 DeepSeek Flash API 密钥');
      result = await planWithDeviceDeepSeek(text, key);
    } else {
      result = await api('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ conversation_id: state.conversationId, message: text }),
      });
    }
    state.task = result.task;
    await openConversation(result.conversation_id);
    $('phaseText').textContent = '计划等待确认';
    toast('计划已生成');
  } catch (error) {
    $('messages').querySelector('.planning-indicator')?.remove();
    toast(error.message, true);
    $('phaseText').textContent = '规划失败';
  } finally {
    $('sendButton').disabled = false;
    $('sendButton').textContent = '↑';
    $('chatInput').disabled = false;
    $('chatInput').focus();
  }
}

async function planWithDeviceDeepSeek(text, apiKey) {
  const context = await api('/api/planning/client-context', {
    method: 'POST', body: JSON.stringify({ message: text }),
  });
  const requestBody = JSON.stringify({
    model: context.model || 'deepseek-flash',
    messages: [{ role: 'user', content: context.prompt }],
    temperature: 0,
    max_tokens: 360,
  });
  const requestOptions = {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: requestBody,
  };
  let response;
  try {
    response = await fetch('https://api.deepseek.com/chat/completions', requestOptions);
  } catch (directError) {
    try {
      response = await fetch('http://127.0.0.1:8765/v1/chat/completions', requestOptions);
    } catch (bridgeError) {
      throw new Error('当前浏览器无法直连 DeepSeek，且本机密钥桥未启动（127.0.0.1:8765）');
    }
  }
  let value;
  try { value = await response.json(); }
  catch { throw new Error(`DeepSeek 返回了无法解析的响应（HTTP ${response.status}）`); }
  if (!response.ok) throw new Error(value?.error?.message || value?.error || `DeepSeek HTTP ${response.status}`);
  const output = value?.choices?.[0]?.message?.content;
  if (typeof output !== 'string' || !output.trim()) throw new Error('DeepSeek 没有返回计划内容');
  return api('/api/chat/client-plan', {
    method: 'POST',
    body: JSON.stringify({ conversation_id: state.conversationId, message: text, model_output: output }),
  });
}

function stepState(id) { return state.task?.execution?.steps?.[id]?.status || 'pending'; }
const statusNames = {
  pending: '等待中', completed: '已完成', running: '执行中', awaiting_verification: '待确认',
  failed: '失败', blocked: '已阻塞', paused: '已暂停', confirmed: '已确认', draft: '待确认',
};
const actionNames = {
  inspect_semantic_map: '读取语义地图', patrol_search: 'AI确认观察点并巡航', locate_object: '视觉寻找目标',
  navigate_to: '导航到目标附近', align_for_grasp: '视觉追踪并对准', grasp: '机械臂抓取',
  verify_grasp: '确认抓取', locate_destination: '定位放置点', release: '放下物体',
  verify_completion: '确认完成', return_home: '返回起点', report_result: '汇报结果',
  drive_relative: '雷达保护移动',
};

function renderTask() {
  const task = state.task;
  $('refreshTask').disabled = !task;
  if (!task) {
    $('taskState').textContent = '未规划';
    $('taskState').className = 'pill';
    $('planSummary').textContent = '发送一条指令后，这里会显示可核验的结构化步骤。';
    $('planSteps').innerHTML = '';
    $('planBadge').textContent = '—';
    setTaskPrimary({ label: '等待任务', disabled: true });
    return;
  }
  $('taskState').textContent = statusNames[task.status] || task.status;
  $('taskState').className = 'pill ready';
  $('planSummary').innerHTML = `<strong>${esc(task.plan.summary || '任务计划')}</strong><br><small>${esc(task.plan.source || '')} · ${esc(task.plan.intent || '')}</small>`;
  const steps = task.plan.steps || [];
  const currentIndex = steps.findIndex(step => stepState(step.id) !== 'completed');
  const completedCount = steps.filter(step => stepState(step.id) === 'completed').length;
  $('planBadge').textContent = `${completedCount}/${steps.length}`;
  $('planSteps').innerHTML = steps.map((step, index) => {
    const status = stepState(step.id);
    const className = status === 'completed' ? 'completed' : (index === currentIndex ? 'current' : 'locked');
    return `<div class="plan-step ${className}" ${index === currentIndex ? 'aria-current="step"' : ''}><div class="plan-step-head"><span class="step-index">${status === 'completed' ? '✓' : index + 1}</span><b>${esc(actionNames[step.action] || step.action)}</b><span class="step-status ${status}">${esc(statusNames[status] || status)}</span></div><p>${esc(step.description || '')}</p></div>`;
  }).join('');
  renderTaskPrimary(steps, currentIndex);
}

function setTaskPrimary({ label, disabled = false, mode = '', stepId = '' }) {
  const button = $('taskPrimaryAction');
  button.textContent = label;
  button.disabled = disabled;
  button.dataset.mode = mode;
  button.dataset.stepId = stepId;
}

function renderTaskPrimary(steps, currentIndex) {
  const task = state.task;
  if (!task) return setTaskPrimary({ label: '等待任务', disabled: true });
  if (state.actionBusy) return setTaskPrimary({ label: '正在处理，请稍候…', disabled: true, mode: 'busy' });
  if (task.status === 'completed' || currentIndex < 0) return setTaskPrimary({ label: '任务已完成', disabled: true });
  if (['draft', 'paused'].includes(task.status)) return setTaskPrimary({ label: '确认计划并检查地图', mode: 'confirm' });
  const step = steps[currentIndex];
  const status = stepState(step.id);
  const action = actionNames[step.action] || step.action;
  const robotBusy = Boolean(state.robot?.managed?.running);
  if (['running', 'awaiting_verification'].includes(status)) {
    if (robotBusy) return setTaskPrimary({ label: `正在执行：${action}`, disabled: true, mode: 'busy', stepId: step.id });
    const returnCode = state.robot?.managed?.returncode;
    if (returnCode != null && returnCode !== 0) {
      return setTaskPrimary({ label: `重新执行：${action}`, mode: 'execute', stepId: step.id });
    }
    return setTaskPrimary({ label: `确认完成：${action}`, mode: 'complete', stepId: step.id });
  }
  return setTaskPrimary({ label: `${['failed', 'blocked'].includes(status) ? '重试' : '执行'}：${action}`, mode: 'execute', stepId: step.id });
}

async function confirmPlan() {
  if (!state.task) return;
  setTaskPrimary({ label: '正在确认计划…', disabled: true, mode: 'busy' });
  try {
    const result = await api(`/api/tasks/${state.task.id}/confirm`, { method: 'POST', body: '{}' });
    state.task = result.task; renderTask();
    const first = state.task.plan?.steps?.[0];
    if (first?.action === 'inspect_semantic_map' && stepState(first.id) !== 'completed') {
      toast('计划已确认，正在检查地图');
      await executeStep(first.id);
    } else toast('计划已确认，可以逐步执行');
  } catch (error) { renderTask(); toast(error.message, true); }
}

async function refreshTask() {
  if (!state.task) return;
  $('refreshTask').disabled = true;
  try {
    const result = await api(`/api/tasks/${state.task.id}`);
    state.task = result.task; renderTask(); toast('任务状态已刷新');
  } catch (error) { toast(error.message, true); }
  finally { $('refreshTask').disabled = false; }
}

async function executeStep(id) {
  setTaskPrimary({ label: '正在发送指令…', disabled: true, mode: 'busy', stepId: id });
  try {
    const result = await api(`/api/tasks/${state.task.id}/steps/${id}/execute`, { method: 'POST', body: '{}' });
    state.task = result.task; renderTask();
    closeResponsivePanels();
    const blockedMessage = result.result.detail?.message || result.result.detail?.decision?.clarification_question || '当前地图信息不足，任务已暂停';
    toast(result.result.status === 'blocked' ? blockedMessage : `步骤状态：${result.result.status}`, result.result.status === 'blocked');
  } catch (error) { await refreshTask(); toast(error.message, true); }
}

async function completeStep(id) {
  setTaskPrimary({ label: '正在确认…', disabled: true, mode: 'busy', stepId: id });
  try {
    const result = await api(`/api/tasks/${state.task.id}/steps/${id}/complete`, { method: 'POST', body: '{}' });
    state.task = result.task; renderTask(); toast('步骤已确认完成');
  } catch (error) { toast(error.message, true); }
}

async function runTaskPrimary() {
  const button = $('taskPrimaryAction');
  if (button.disabled || state.actionBusy) return;
  const mode = button.dataset.mode;
  const stepId = button.dataset.stepId;
  state.actionBusy = true;
  renderTask();
  try {
    if (mode === 'confirm') await confirmPlan();
    else if (mode === 'execute') await executeStep(stepId);
    else if (mode === 'complete') await completeStep(stepId);
  } finally {
    state.actionBusy = false;
    renderTask();
  }
}

function draftKey() { return state.conversationId ? `${DRAFT_PREFIX}${state.conversationId}` : ''; }
function resizeChatInput() {
  const input = $('chatInput');
  input.style.height = 'auto';
  input.style.height = `${Math.min(130, Math.max(36, input.scrollHeight))}px`;
}
function saveCurrentDraft() {
  const key = draftKey();
  if (!key) return;
  const value = $('chatInput').value;
  if (value) localStorage.setItem(key, value); else localStorage.removeItem(key);
}
function restoreDraft() {
  $('chatInput').value = localStorage.getItem(draftKey()) || '';
  resizeChatInput();
}

function closeResponsivePanels() {
  document.body.classList.remove('sidebar-open', 'mission-open');
  $('toggleSidebar').setAttribute('aria-expanded', 'false');
  $('toggleMission').setAttribute('aria-expanded', 'false');
  $('uiBackdrop').hidden = true;
}
function toggleResponsivePanel(name) {
  const className = `${name}-open`;
  const open = !document.body.classList.contains(className);
  closeResponsivePanels();
  if (open) {
    document.body.classList.add(className);
    $(`toggle${name[0].toUpperCase()}${name.slice(1)}`).setAttribute('aria-expanded', 'true');
    $('uiBackdrop').hidden = false;
  }
}

async function pollRobot() {
  try {
    const robot = await api('/api/robot/state');
    state.robot = robot;
    if (robot.static_map) state.mapMeta = robot.static_map;
    $('piDot').className = 'ok';
    $('piConnection').textContent = '树莓派桥接在线';
    $('hardwareMode').textContent = robot.hardware_enabled ? '实车模式' : '安全模式';
    const managed = robot.managed || {};
    $('trackingState').textContent = managed.kind === 'visual_pursuit' ? '运行中' : (managed.kind || '待机');
    const grasp = robot.grasp || {};
    $('graspState').textContent = grasp.connected ? (grasp.data?.grasp?.phase || '待机') : '离线';
    const map = robot.map?.data || {};
    renderPose(map);
    const semantic = robot.semantic || {};
    $('semanticState').textContent = semantic.status === 'stable_release'
      ? `稳定发布 · ${semantic.entity_count ?? 0} 个语义实体`
      : semantic.status === 'compatibility_fallback'
        ? `兼容标注 · ${semantic.entity_count ?? 0} 个实体（稳定发布包缺失）`
        : (semantic.warning || '语义标注不可用');
    const active = state.mapLayers.find(item => item.id === state.activeMapLayer);
    $('mapSource').textContent = (active?.source_path || robot.static_map?.source_path || '').split('/').pop() || 'ble_scene_17.pgm';
    $('mapToolState').textContent = state.pose ? '小车定位在线' : '底图就绪';
    $('visionToolState').textContent = grasp.connected ? '视频服务在线' : '点击查看';
    $('observationAdapter').textContent = robot.interfaces?.observation_viewpoint_selection === 'connected_car1_search_route'
      ? 'car1.0 A* 多观察点已接入' : '观测点适配器离线';
    $('navigationAdapter').textContent = robot.interfaces?.text_map_navigation === 'connected_car1_object_search'
      ? '自然语言 → 语义区域 → 雷达导航' : '导航适配器离线';
    renderRobotLiveStatus(robot);
    renderTask();
  } catch {
    $('piDot').className = '';
    $('piConnection').textContent = '树莓派桥接离线';
    renderRobotOffline();
  }
}

const processNames = {
  idle: '待机', camera_preview: '双目预览', visual_pursuit: '视觉追踪',
  map_navigation_search: '地图导航与目标搜索', mapping: '雷达建图', grasp: '机械臂抓取',
};

const phaseNames = {
  localizing: '地图定位', planning: '规划路线', navigating: '导航移动',
  observing: '视觉观察', searching: '搜索目标', tracking: '持续追踪',
  aligning: '抓取对准', grasping: '机械臂抓取', verifying: '结果确认',
  completed: '任务完成', stopped: '已停止', idle: '待机',
};

function currentPlanProgress(robotTask, taskPlan) {
  const steps = Array.isArray(taskPlan?.steps) ? taskPlan.steps : [];
  const total = Number(taskPlan?.total_steps ?? robotTask?.queue_total ?? steps.length ?? 0);
  let completed = Number(taskPlan?.completed_steps ?? 0);
  if (!Number.isFinite(completed) || completed < 0) completed = steps.filter(step => step.status === 'completed').length;
  const active = steps.find(step => ['active', 'running', 'awaiting_verification'].includes(step.status));
  return { total: Math.max(0, total), completed: Math.max(0, completed), active };
}

function webTaskProgress() {
  const steps = state.task?.plan?.steps || [];
  const states = state.task?.execution?.steps || {};
  const values = Object.entries(states);
  return {
    total: steps.length,
    completed: values.filter(([, value]) => value?.status === 'completed').length,
    failed: values.slice().reverse().find(([, value]) => ['failed', 'blocked'].includes(value?.status)),
    active: values.find(([, value]) => ['running', 'awaiting_verification'].includes(value?.status)),
  };
}

function renderRobotLiveStatus(robot) {
  const managed = robot.managed || {};
  const companion = robot.companion?.data || {};
  const robotTask = companion.robot_task || {};
  const taskPlan = companion.task_plan || {};
  const mapState = robot.map?.data || {};
  const mapCommand = mapState.command || mapState.navigation?.command || {};
  const command = companion.robot_command || mapCommand || {};
  const mapAgeMs = Number(robot.time_ms || Date.now()) - Number(mapState._updated_at || 0);
  const freshMapTask = mapAgeMs >= 0 && mapAgeMs < 5000;
  const running = Boolean(managed.running || (freshMapTask && (robotTask.active || mapState.running)));
  const completed = Boolean(robotTask.completed || taskPlan.status === 'completed');
  const webProgress = webTaskProgress();
  const processLabel = processNames[managed.kind] || managed.kind || '待机';
  const phaseKey = String(robotTask.phase || command.state || '').toLowerCase();
  const phaseLabel = phaseNames[phaseKey] || robotTask.phase_label || command.state || processLabel;
  const plannedInstruction = mapState.task?.instruction
    || mapState.navigation?.task?.instruction
    || taskPlan.instruction
    || state.task?.instruction;
  const instruction = running
    ? (plannedInstruction || '正在执行受管任务')
    : ((webProgress.failed || webProgress.active)
      ? (state.task?.instruction || '任务步骤等待确认') : '等待新的执行指令');
  const detail = command.reason
    || robotTask.phase_label
    || currentPlanProgress(robotTask, taskPlan).active?.detail
    || (running ? `${processLabel}正在运行` : '小车已停止，等待下一条指令');
  const robotProgress = currentPlanProgress(robotTask, taskPlan);
  const progress = running && robotProgress.total ? robotProgress : webProgress;
  const ratio = progress.total ? Math.min(100, progress.completed / progress.total * 100) : (completed ? 100 : 0);
  const failedDetail = webProgress.failed?.[1]?.detail || {};
  const failureMessage = failedDetail.error || failedDetail.message
    || failedDetail.decision?.clarification_question || '步骤未能启动，请查看运行日志';
  const failed = !running && Boolean(webProgress.failed);
  const awaitingConfirmation = !running && Boolean(webProgress.active);
  const finishedDetail = managed.returncode === 0
    ? '底层任务已正常结束，等待确认当前步骤'
    : `底层任务已结束（退出码 ${managed.returncode ?? '未知'}），请检查日志`;

  $('robotLiveDot').className = failed ? 'error' : (running ? 'running' : 'online');
  $('robotLiveState').textContent = failed ? '步骤未启动'
    : (awaitingConfirmation ? '等待步骤确认' : (completed ? '任务完成' : (running ? phaseLabel : '小车待机')));
  $('robotLiveMode').textContent = `${robot.hardware_enabled ? '实车模式' : '安全模式'} · ${processLabel}`;
  $('robotLiveCommand').textContent = instruction;
  $('robotLiveCommand').title = instruction;
  const visibleDetail = failed ? failureMessage : (awaitingConfirmation ? finishedDetail : detail);
  $('robotLiveDetail').textContent = visibleDetail;
  $('robotLiveDetail').title = visibleDetail;
  $('robotLiveProgress').textContent = progress.total ? `${progress.completed} / ${progress.total}` : (running ? '执行中' : '—');
  $('robotLiveProgressBar').style.width = `${ratio}%`;
  $('robotLiveUpdated').textContent = `更新 ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
  $('phaseText').textContent = failed
    ? `未启动 · ${failureMessage}`
    : (awaitingConfirmation ? finishedDetail
      : (running ? `${phaseLabel} · ${detail}` : (state.task ? `任务 ${state.task.status}` : '等待指令')));
}

function renderRobotOffline() {
  $('robotLiveDot').className = 'error';
  $('robotLiveState').textContent = '小车离线';
  $('robotLiveMode').textContent = '无法连接树莓派桥接';
  $('robotLiveCommand').textContent = '状态暂不可用';
  $('robotLiveDetail').textContent = '页面会继续自动重连';
  $('robotLiveProgress').textContent = '—';
  $('robotLiveProgressBar').style.width = '0%';
  $('robotLiveUpdated').textContent = `重试 ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
  $('phaseText').textContent = '树莓派桥接离线，正在重连';
}

function renderPose(status) {
  const pose = status.pose || status.robot_pose || {};
  const yaw = pose.yaw_deg ?? (pose.yaw_rad == null ? null : Number(pose.yaw_rad) * 180 / Math.PI);
  const values = [pose.x_m, pose.y_m, yaw, status.rmse_m ?? status.localization_quality];
  const fields = $('poseFacts').querySelectorAll('dd');
  fields[0].textContent = values[0] == null ? '—' : `${Number(values[0]).toFixed(2)} m`;
  fields[1].textContent = values[1] == null ? '—' : `${Number(values[1]).toFixed(2)} m`;
  fields[2].textContent = values[2] == null ? '—' : `${Number(values[2]).toFixed(1)}°`;
  fields[3].textContent = values[3] == null ? '—' : String(values[3]);
  state.pose = values[0] == null || values[1] == null ? null : { x: Number(values[0]), y: Number(values[1]), yaw: Number(yaw || 0) };
  updateRobotMarker();
}

function layoutMapFrame() {
  const viewport = $('mapViewport');
  const image = $('mapImage');
  if (!image.naturalWidth || !viewport.clientWidth) return;
  const availableWidth = viewport.clientWidth * .94;
  const availableHeight = viewport.clientHeight * .94;
  const ratio = image.naturalWidth / image.naturalHeight;
  let width = availableWidth;
  let height = width / ratio;
  if (height > availableHeight) { height = availableHeight; width = height * ratio; }
  $('mapFrame').style.width = `${width}px`;
  $('mapFrame').style.height = `${height}px`;
  updateRobotMarker();
}

function updateMapTransform() {
  const view = state.mapView;
  $('mapPanLayer').style.transform = `translate3d(${view.panX}px, ${view.panY}px, 0) scale(${view.scale})`;
  $('zoomValue').textContent = `${Math.round(view.scale * 100)}%`;
}

function updateRobotMarker() {
  const marker = $('robotMarker');
  const image = $('mapImage');
  if (!state.pose || !image.naturalWidth) { marker.hidden = true; return; }
  const resolution = Number(state.mapMeta.resolution_m || .05);
  const origin = state.mapMeta.origin || [-10, -10, 0];
  const pixelX = (state.pose.x - Number(origin[0])) / resolution;
  const pixelY = image.naturalHeight - (state.pose.y - Number(origin[1])) / resolution;
  if (pixelX < 0 || pixelY < 0 || pixelX > image.naturalWidth || pixelY > image.naturalHeight) {
    marker.hidden = true; return;
  }
  marker.hidden = false;
  marker.style.left = `${pixelX / image.naturalWidth * 100}%`;
  marker.style.top = `${pixelY / image.naturalHeight * 100}%`;
  marker.style.transform = `rotate(${-state.pose.yaw + 90}deg)`;
}

function resetMapView() {
  const view = state.mapView;
  view.scale = 1.35; view.panX = 0; view.panY = 0;
  const marker = $('robotMarker');
  const frame = $('mapFrame');
  if (!marker.hidden && frame.clientWidth) {
    const markerX = parseFloat(marker.style.left) / 100 * frame.clientWidth;
    const markerY = parseFloat(marker.style.top) / 100 * frame.clientHeight;
    view.panX = -view.scale * (markerX - frame.clientWidth / 2);
    view.panY = -view.scale * (markerY - frame.clientHeight / 2);
  }
  updateMapTransform();
}

function setMapScale(next) {
  state.mapView.scale = Math.max(.65, Math.min(4.5, next));
  updateMapTransform();
}

function reloadMap() {
  const image = $('mapImage');
  $('mapEmpty').style.display = 'grid';
  $('mapEmpty').textContent = '地图载入中';
  image.onload = () => {
    $('mapEmpty').style.display = 'none';
    layoutMapFrame();
    resetMapView();
  };
  image.onerror = () => {
    $('mapEmpty').style.display = 'grid';
    $('mapEmpty').textContent = '地图暂不可用';
  };
  const layer = state.activeMapLayer;
  image.src = layer
    ? `/api/robot/map/layer/${encodeURIComponent(layer)}.png?t=${Date.now()}`
    : `/api/robot/map.png?t=${Date.now()}`;
}

function semanticItems(value) {
  const entities = value?.entities;
  if (entities && typeof entities === 'object' && !Array.isArray(entities)) {
    return Object.entries(entities).map(([id, item]) => {
      const annotation = item?.qwen_annotation || {};
      return {
        id,
        name: annotation.name_zh || item.name || item.semantic_label || id,
        review: Boolean(annotation.needs_human_review),
      };
    });
  }
  const items = value?.landmarks || value?.objects || [];
  return Array.isArray(items) ? items.map((item, index) => ({
    id: item.id || item.target_id || String(index),
    name: item.name || item.label || item.category || item.id || `实体 ${index + 1}`,
    review: Boolean(item.needs_human_review),
  })) : [];
}

async function loadMapLayers() {
  try {
    const result = await api('/api/robot/map/layers');
    state.mapLayers = Array.isArray(result.layers) ? result.layers : [];
    const available = new Set(state.mapLayers.map(item => item.id));
    if (!state.activeMapLayer || !available.has(state.activeMapLayer)) {
      state.activeMapLayer = state.mapLayers.find(item => item.is_default)?.id
        || state.mapLayers.find(item => item.id === 'base')?.id
        || state.mapLayers[0]?.id
        || null;
    }
    $('mapLayerSelect').innerHTML = state.mapLayers.map(item =>
      `<option value="${esc(item.id)}">${esc(item.label)}</option>`
    ).join('') || '<option value="">地图不可用</option>';
    $('mapLayerSelect').value = state.activeMapLayer || '';
    const semantic = result.semantic || {};
    if (semantic.warning) $('semanticState').title = semantic.warning;
  } catch (error) {
    state.mapLayers = [];
    $('mapLayerSelect').innerHTML = '<option value="">地图图层不可用</option>';
    $('semanticState').textContent = error.message;
  }
}

async function loadSemanticMap() {
  try {
    const semantic = await api('/api/robot/semantic-map');
    const items = semanticItems(semantic);
    $('semanticEntities').innerHTML = items.slice(0, 40).map(item =>
      `<span class="semantic-chip${item.review ? ' review' : ''}" title="${esc(item.id)}">${esc(item.name)}</span>`
    ).join('');
    if (!items.length) $('semanticEntities').innerHTML = '<span class="semantic-chip">暂无实体</span>';
  } catch (error) {
    $('semanticEntities').innerHTML = `<span class="semantic-chip review">${esc(error.message)}</span>`;
  }
}

function connectVideoStreams() {
  const stamp = Date.now();
  ['left', 'right'].forEach(side => {
    const image = $(`${side}Video`);
    const label = $(`${side}CameraState`);
    label.textContent = '连接中';
    image.onload = () => { label.textContent = '实时'; };
    image.onerror = () => { label.textContent = '未连接'; };
    image.src = `/api/robot/video/${side}?t=${stamp}`;
  });
}

function disconnectVideoStreams() {
  $('leftVideo').src = '';
  $('rightVideo').src = '';
}

async function startPreview() {
  try {
    await api('/api/robot/preview/start', { method: 'POST', body: '{}' });
    toast('双目预览正在启动，随后自动校准左右');
    setTimeout(connectVideoStreams, 1300);
    setTimeout(calibrateCameras, 1700);
  } catch (error) { toast(error.message, true); }
}

async function stopPreview() {
  try {
    await api('/api/robot/preview/stop', { method: 'POST', body: '{}' });
    disconnectVideoStreams();
    $('leftCameraState').textContent = '已停止';
    $('rightCameraState').textContent = '已停止';
    toast('双目预览已停止');
  } catch (error) { toast(error.message, true); }
}

function renderCameraOrder(value) {
  const node = $('cameraOrderState');
  if (!value || value.source === 'default') {
    node.textContent = '尚未校准';
    return;
  }
  const order = value.swapped ? '已交换' : '左右正常';
  const confidence = value.confidence ? ` · ${Math.round(value.confidence * 100)}%` : '';
  node.textContent = `${order}${confidence}`;
  node.title = value.reason || value.source || '';
}

async function loadCameraOrder() {
  try {
    const result = await api('/api/robot/camera-order');
    renderCameraOrder(result.camera_order);
  } catch (error) {
    $('cameraOrderState').textContent = '状态不可用';
  }
}

async function calibrateCameras() {
  $('calibrateCameras').disabled = true;
  $('cameraOrderState').textContent = 'Qwen 正在比较双目视差…';
  try {
    const result = await api('/api/robot/camera-order/calibrate', { method: 'POST', body: '{}' });
    renderCameraOrder(result.camera_order);
    disconnectVideoStreams();
    connectVideoStreams();
    const inference = result.inference || {};
    toast(result.applied
      ? `左右校准完成：${inference.order === 'swapped' ? '已自动交换' : '顺序正常'}`
      : 'AI 证据不足，未自动更改；可使用手动交换', !result.applied);
  } catch (error) {
    toast(error.message, true);
    await loadCameraOrder();
  } finally {
    $('calibrateCameras').disabled = false;
  }
}

async function swapCameras() {
  try {
    const result = await api('/api/robot/camera-order/swap', { method: 'POST', body: '{}' });
    renderCameraOrder(result.camera_order);
    disconnectVideoStreams();
    connectVideoStreams();
    toast(result.camera_order.swapped ? '左右画面已交换' : '已恢复原始顺序');
  } catch (error) { toast(error.message, true); }
}

async function robotControl(name) {
  try { await api(`/api/robot/${name}`, { method: 'POST', body: '{}' }); toast('指令已发送'); }
  catch (error) { toast(error.message, true); }
}

async function logs() {
  try {
    const result = await api('/api/robot/logs');
    $('logOutput').textContent = result.text || '当前进程尚无日志。';
  } catch (error) { $('logOutput').textContent = error.message; }
}

function openModal(name) {
  const modal = $(`${name}Modal`);
  modal.hidden = false;
  document.body.classList.add('modal-open');
  modal.querySelector('.modal-close')?.focus();
  if (name === 'map') {
    Promise.all([loadMapLayers(), loadSemanticMap()]).finally(() => requestAnimationFrame(() => reloadMap()));
  }
  if (name === 'vision') { connectVideoStreams(); loadCameraOrder(); }
  if (name === 'settings') loadSettings().catch(error => toast(error.message, true));
}

function closeModal(modal) {
  if (!modal || modal.hidden) return;
  if (modal.id === 'visionModal') disconnectVideoStreams();
  modal.hidden = true;
  if (![...document.querySelectorAll('.modal-overlay')].some(item => !item.hidden)) {
    document.body.classList.remove('modal-open');
  }
}

function applyLanguage(language) {
  const dictionary = translations[language] || translations['zh-CN'];
  document.documentElement.lang = language;
  document.querySelectorAll('[data-i18n]').forEach(node => {
    if (dictionary[node.dataset.i18n]) node.textContent = dictionary[node.dataset.i18n];
  });
}

function providerDetails(id) {
  return state.settings?.providers?.find(item => item.id === id) || null;
}

function updateSettingsFields() {
  const provider = $('providerSelect').value;
  const external = provider !== 'local_qwen';
  $('modelField').style.display = external ? 'block' : 'none';
  $('keyField').style.display = external ? 'block' : 'none';
  $('clearApiKey').style.display = external ? 'block' : 'none';
  const details = providerDetails(provider);
  $('modelInput').value = details?.model || '';
  $('apiKeyInput').value = '';
  if (details?.client_managed_key) {
    const configured = Boolean(localStorage.getItem(DEEPSEEK_FLASH_KEY));
    $('keyStatus').textContent = configured
      ? '已保存在当前浏览器，留空保持不变'
      : '尚未在当前浏览器配置';
  } else {
    $('keyStatus').textContent = details?.configured ? '已配置，留空保持不变' : '尚未配置';
  }
}

function renderActiveModel() {
  const provider = providerDetails(state.settings?.provider);
  $('activeModel').textContent = provider?.label || '本地 Qwen';
}

async function loadSettings() {
  const result = await api('/api/settings');
  state.settings = result.settings;
  $('languageSelect').value = state.settings.language;
  $('providerSelect').value = state.settings.provider;
  applyLanguage(state.settings.language);
  updateSettingsFields();
  renderActiveModel();
}

async function saveSettings(event) {
  event.preventDefault();
  const provider = $('providerSelect').value;
  const payload = { provider, language: $('languageSelect').value };
  if (provider !== 'local_qwen') {
    payload.model = $('modelInput').value.trim();
    if (provider === 'deepseek_flash_api') {
      if ($('apiKeyInput').value.trim()) localStorage.setItem(DEEPSEEK_FLASH_KEY, $('apiKeyInput').value.trim());
    } else if ($('apiKeyInput').value.trim()) {
      payload.api_key = $('apiKeyInput').value.trim();
    }
  }
  try {
    const result = await api('/api/settings', { method: 'PATCH', body: JSON.stringify(payload) });
    state.settings = result.settings;
    applyLanguage(state.settings.language);
    updateSettingsFields();
    renderActiveModel();
    closeModal($('settingsModal'));
    toast('设置已保存');
  } catch (error) { toast(error.message, true); }
}

async function clearApiKey() {
  const provider = $('providerSelect').value;
  if (provider === 'local_qwen') return;
  if (provider === 'deepseek_flash_api') localStorage.removeItem(DEEPSEEK_FLASH_KEY);
  try {
    const result = await api('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify({ provider, language: $('languageSelect').value, clear_api_key: true }),
    });
    state.settings = result.settings;
    updateSettingsFields();
    toast('当前 API 密钥已清除');
  } catch (error) { toast(error.message, true); }
}

document.querySelectorAll('.tab').forEach(button => {
  button.onclick = () => {
    document.querySelectorAll('.tab').forEach(item => item.classList.toggle('active', item === button));
    document.querySelectorAll('.tab-panel').forEach(item => item.classList.remove('active'));
    $(`tab-${button.dataset.tab}`).classList.add('active');
    if (button.dataset.tab === 'logs') logs();
  };
});

document.querySelectorAll('[data-close-modal]').forEach(button => {
  button.onclick = () => closeModal(button.closest('.modal-overlay'));
});
document.querySelectorAll('.modal-overlay').forEach(modal => {
  modal.addEventListener('mousedown', event => { if (event.target === modal) closeModal(modal); });
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    const modal = [...document.querySelectorAll('.modal-overlay')].find(item => !item.hidden);
    if (modal) closeModal(modal); else closeResponsivePanels();
  }
});

const viewport = $('mapViewport');
viewport.addEventListener('wheel', event => {
  event.preventDefault();
  setMapScale(state.mapView.scale * (event.deltaY > 0 ? .9 : 1.1));
}, { passive: false });
viewport.addEventListener('pointerdown', event => {
  state.mapView.dragging = true; state.mapView.pointerId = event.pointerId;
  state.mapView.x = event.clientX; state.mapView.y = event.clientY;
  viewport.setPointerCapture(event.pointerId); viewport.classList.add('dragging');
});
viewport.addEventListener('pointermove', event => {
  if (!state.mapView.dragging || event.pointerId !== state.mapView.pointerId) return;
  state.mapView.panX += event.clientX - state.mapView.x;
  state.mapView.panY += event.clientY - state.mapView.y;
  state.mapView.x = event.clientX; state.mapView.y = event.clientY;
  updateMapTransform();
});
function endMapDrag(event) {
  if (event.pointerId !== state.mapView.pointerId) return;
  state.mapView.dragging = false; state.mapView.pointerId = null; viewport.classList.remove('dragging');
}
viewport.addEventListener('pointerup', endMapDrag);
viewport.addEventListener('pointercancel', endMapDrag);
viewport.addEventListener('dblclick', resetMapView);
window.addEventListener('resize', () => { if (!$('mapModal').hidden) { layoutMapFrame(); resetMapView(); } });

$('newChat').onclick = newConversation;
$('chatForm').onsubmit = event => {
  event.preventDefault();
  const text = $('chatInput').value.trim();
  if (text) { $('chatInput').value = ''; resizeChatInput(); sendMessage(text); }
};
$('chatInput').addEventListener('input', () => { resizeChatInput(); saveCurrentDraft(); });
$('chatInput').onkeydown = event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('chatForm').requestSubmit(); }
};
$('messages').addEventListener('scroll', handleMessageScroll, { passive: true });
$('scrollLatest').onclick = () => scrollMessages(true);
document.querySelectorAll('.quick-prompts button').forEach(button => {
  button.onclick = () => { $('chatInput').value = button.textContent; $('chatInput').focus(); };
});
$('taskPrimaryAction').onclick = runTaskPrimary;
$('refreshTask').onclick = refreshTask;
$('toggleSidebar').onclick = () => toggleResponsivePanel('sidebar');
$('toggleMission').onclick = () => toggleResponsivePanel('mission');
$('closeMission').onclick = closeResponsivePanels;
$('uiBackdrop').onclick = closeResponsivePanels;
$('reloadMap').onclick = reloadMap;
$('mapLayerSelect').onchange = () => {
  state.activeMapLayer = $('mapLayerSelect').value || null;
  const active = state.mapLayers.find(item => item.id === state.activeMapLayer);
  $('mapSource').textContent = (active?.source_path || '').split('/').pop() || '地图';
  reloadMap();
};
$('zoomIn').onclick = () => setMapScale(state.mapView.scale * 1.18);
$('zoomOut').onclick = () => setMapScale(state.mapView.scale / 1.18);
$('resetMap').onclick = resetMapView;
$('startPreview').onclick = startPreview;
$('stopPreview').onclick = stopPreview;
$('calibrateCameras').onclick = calibrateCameras;
$('swapCameras').onclick = swapCameras;
$('refreshLogs').onclick = logs;
$('openMap').onclick = () => openModal('map');
$('openVision').onclick = () => openModal('vision');
$('openSettings').onclick = () => openModal('settings');
$('providerSelect').onchange = updateSettingsFields;
$('languageSelect').onchange = () => applyLanguage($('languageSelect').value);
$('settingsForm').onsubmit = saveSettings;
$('clearApiKey').onclick = clearApiKey;
$('emergencyStop').onclick = () => {
  if (confirm('立即停止底盘、追踪任务和机械臂？')) robotControl('emergency-stop');
};
$('startMapping').onclick = () => robotControl('mapping/start');
$('stopMapping').onclick = () => robotControl('mapping/stop');

loadSettings().catch(error => toast(error.message, true));
loadConversations().then(() => { if (!state.conversationId) newConversation(); }).catch(error => toast(error.message, true));
pollRobot();
setInterval(pollRobot, 1500);
