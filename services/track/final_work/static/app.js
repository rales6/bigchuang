const $ = id => document.getElementById(id);
const DEEPSEEK_FLASH_KEY = 'final_work.deepseek_flash_api_key';
const DEEPSEEK_CHAT_URL = 'https://api.deepseek.com/chat/completions';
const DEEPSEEK_RESPONSES_URL = 'https://api.deepseek.com/responses';
const DEEPSEEK_MODELS_URL = 'https://api.deepseek.com/models';
const DEEPSEEK_CHAT_TIMEOUT_MS = 60000;
const DEEPSEEK_TEST_TIMEOUT_MS = 15000;
const DRAFT_PREFIX = 'final_work.chat_draft.';

const state = {
  conversationId: null,
  task: null,
  robot: null,
  pose: null,
  poseFresh: false,
  visionMode: null,
  activeTab: 'observe',
  observeView: 'split',
  pollBusy: false,
  mapLoading: false,
  mapLoadedLayer: null,
  mapMeta: { resolution_m: 0.05, origin: [-10, -10, 0] },
  mapLayers: [],
  activeMapLayer: 'relationship_release',
  settings: null,
  followLatest: true,
  actionBusy: false,
  autoRunTaskId: null,
  autoAdvancing: false,
  taskSyncBusy: false,
  lastTaskSyncAt: 0,
  mapView: { scale: 1.1, panX: 0, panY: 0, dragging: false, pointerId: null, x: 0, y: 0 },
};

const translations = {
  'zh-CN': {
    newTask: '＋ 新建任务', history: '聊天历史', settings: '设置', logs: '运行日志',
    emergencyStop: '紧急停止', map: '地图与定位', stereo: '双目视觉', mapTitle: '雷达地图与定位',
    recenter: '重新居中', refresh: '刷新', liveState: '实时状态', heading: '航向',
    quality: '定位质量', semanticLayer: '语义图层', startMapping: '开始建图', stop: '停止',
    stereoTitle: '双目实时视觉', startPreview: '启动预览', leftCamera: '左摄像头',
    rightCamera: '右摄像头', tracking: '追踪', grasp: '抓取', controlMode: '控制模式',
    settingsTitle: 'Agent 设置', language: '界面语言', plannerProvider: '规划模型',
    modelId: '模型 ID', apiKey: 'API 密钥', saveSettings: '保存设置', testConnection: '测试连接',
  },
  'en-US': {
    newTask: '+ New task', history: 'Chat history', settings: 'Settings', logs: 'Logs',
    emergencyStop: 'Emergency stop', map: 'Map & position', stereo: 'Stereo vision', mapTitle: 'Lidar map & localization',
    recenter: 'Recenter', refresh: 'Refresh', liveState: 'Live status', heading: 'Heading',
    quality: 'Localization', semanticLayer: 'Semantic layer', startMapping: 'Start mapping', stop: 'Stop',
    stereoTitle: 'Live stereo vision', startPreview: 'Start preview', leftCamera: 'Left camera',
    rightCamera: 'Right camera', tracking: 'Tracking', grasp: 'Grasp', controlMode: 'Control mode',
    settingsTitle: 'Agent settings', language: 'Interface language', plannerProvider: 'Planning model',
    modelId: 'Model ID', apiKey: 'API key', saveSettings: 'Save settings', testConnection: 'Test connection',
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
  toast.timer = setTimeout(() => { node.className = 'toast'; }, error ? 7000 : 2800);
}

function deviceDeepSeekKey() {
  try { return localStorage.getItem(DEEPSEEK_FLASH_KEY) || ''; }
  catch { return ''; }
}

function setKeyStatus(message, kind = '') {
  const node = $('keyStatus');
  node.textContent = message;
  node.className = kind;
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function responseJson(response, serviceName) {
  const raw = await response.text();
  let value = {};
  try { value = raw ? JSON.parse(raw) : {}; }
  catch { throw new Error(`${serviceName} 返回了无法解析的响应（HTTP ${response.status}）`); }
  if (!response.ok) {
    const remote = value?.error?.message || value?.error || value?.message;
    const known = {
      400: '请求格式或模型 ID 不正确',
      401: 'API 密钥无效或已失效',
      402: '账户余额不足',
      403: 'API 密钥没有访问该模型的权限',
      429: '请求过于频繁，请稍后重试',
      500: 'DeepSeek 服务内部错误',
      503: 'DeepSeek 服务繁忙，请稍后重试',
    }[response.status];
    throw new Error(remote || known || `${serviceName} HTTP ${response.status}`);
  }
  return value;
}

async function deepSeekFetch(url, options, timeoutMs) {
  try {
    return await fetchWithTimeout(url, options, timeoutMs);
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw new Error(`DeepSeek 请求超过 ${Math.round(timeoutMs / 1000)} 秒，已停止等待`);
    }
    throw new Error(`当前浏览器无法连接 DeepSeek：${error?.message || '网络请求失败'}`);
  }
}

function reportClientPlannerEvent(stage, status, details = {}) {
  // Deliberately omit the API key, prompt, headers and model output. This is
  // fire-and-forget diagnostics so a logging failure cannot block planning.
  fetch('/api/planning/client-event', {
    method: 'POST',
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      stage, status,
      model: String(details.model || '').slice(0, 100),
      duration_ms: Number(details.duration_ms || 0),
      message: String(details.message || '').slice(0, 500),
    }),
  }).catch(() => {});
}

function deepSeekOutputText(value) {
  const content = value?.choices?.[0]?.message?.content;
  if (typeof content === 'string') return content.trim();
  if (Array.isArray(content)) {
    const joined = content.map(part => {
      if (typeof part === 'string') return part;
      return part?.text || part?.content || '';
    }).join('').trim();
    if (joined) return joined;
  }
  if (typeof value?.choices?.[0]?.text === 'string') return value.choices[0].text.trim();
  if (typeof value?.output_text === 'string') return value.output_text.trim();
  if (Array.isArray(value?.output)) {
    return value.output.flatMap(item => Array.isArray(item?.content) ? item.content : [])
      .map(part => part?.text || '').join('').trim();
  }
  return '';
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
    node.onclick = () => { showWorkspace('agent'); openConversation(node.dataset.id); };
  });
  if (!state.conversationId && result.conversations.length) {
    await openConversation(result.conversations[0].id);
  }
}

async function newConversation() {
  state.autoRunTaskId = null;
  saveCurrentDraft();
  $('chatInput').value = '';
  const result = await api('/api/conversations', { method: 'POST', body: JSON.stringify({ title: '新任务' }) });
  state.conversationId = result.conversation.id;
  state.task = null;
  await loadConversations();
  await openConversation(state.conversationId);
  if (state.activeTab === 'agent') $('chatInput').focus();
}

async function openConversation(id) {
  saveCurrentDraft();
  if (state.conversationId && state.conversationId !== id) state.autoRunTaskId = null;
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
  state.autoRunTaskId = null;
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
      const key = deviceDeepSeekKey();
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
    const message = error?.message || '未知错误';
    $('messages').insertAdjacentHTML('beforeend', `<div class="message assistant"><div class="bubble planning-error"><b>规划失败</b><br>${esc(message)}</div></div>`);
    scrollMessages(true);
    toast(message, true);
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
  const startedAt = Date.now();
  const model = context.model || 'deepseek-flash';
  reportClientPlannerEvent('chat', 'started', { model });
  const elapsedTimer = setInterval(() => {
    $('phaseText').textContent = `正在等待 DeepSeek 返回… ${Math.floor((Date.now() - startedAt) / 1000)}秒`;
  }, 1000);
  $('phaseText').textContent = '正在等待 DeepSeek 返回… 0秒';
  try {
    let output = '';
    const failures = [];
    const attempts = [
      {
        name: 'responses',
        url: DEEPSEEK_RESPONSES_URL,
        body: {
          model,
          input: context.prompt,
          reasoning: { effort: 'none' },
          max_output_tokens: 1536,
          stream: false,
        },
      },
      {
        name: 'chat',
        url: DEEPSEEK_CHAT_URL,
        body: {
          model,
          messages: [{ role: 'user', content: context.prompt }],
          thinking: { type: 'disabled' },
          reasoning_effort: 'none',
          max_tokens: 1536,
          stream: false,
        },
      },
    ];
    for (const attempt of attempts) {
      const requestOptions = {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(attempt.body),
      };
      try {
        $('phaseText').textContent = `正在调用 DeepSeek ${attempt.name} 接口…`;
        const response = await deepSeekFetch(attempt.url, requestOptions, DEEPSEEK_CHAT_TIMEOUT_MS);
        reportClientPlannerEvent('chat', 'http_response', {
          model, duration_ms: Date.now() - startedAt, message: `${attempt.name} HTTP ${response.status}`,
        });
        const value = await responseJson(response, `DeepSeek ${attempt.name}`);
        output = deepSeekOutputText(value);
        if (output) break;
        const choice = value?.choices?.[0] || {};
        const usage = value?.usage || {};
        const detail = `${attempt.name}: status=${value?.status || 'unknown'}, finish_reason=${choice.finish_reason || value?.incomplete_details?.reason || 'unknown'}, output_tokens=${usage.output_tokens ?? usage.completion_tokens ?? 'unknown'}, reasoning_tokens=${usage?.output_tokens_details?.reasoning_tokens ?? usage?.completion_tokens_details?.reasoning_tokens ?? 'unknown'}`;
        failures.push(detail);
        reportClientPlannerEvent('chat', 'empty_response', {
          model, duration_ms: Date.now() - startedAt, message: detail,
        });
      } catch (attemptError) {
        const detail = `${attempt.name}: ${attemptError?.message || '调用失败'}`;
        failures.push(detail);
        reportClientPlannerEvent('chat', 'attempt_failed', {
          model, duration_ms: Date.now() - startedAt, message: detail,
        });
      }
      $('phaseText').textContent = '首选接口未生成计划，正在切换兼容接口…';
    }
    if (!output) throw new Error(`DeepSeek 两种接口均未生成计划：${failures.join('；')}`);
    $('phaseText').textContent = '正在校验 DeepSeek 计划…';
    const result = await api('/api/chat/client-plan', {
      method: 'POST',
      body: JSON.stringify({ conversation_id: state.conversationId, message: text, model_output: output }),
    });
    reportClientPlannerEvent('chat', 'completed', { model, duration_ms: Date.now() - startedAt });
    return result;
  } catch (error) {
    reportClientPlannerEvent('chat', 'failed', {
      model, duration_ms: Date.now() - startedAt, message: error?.message || '未知错误',
    });
    throw error;
  } finally {
    clearInterval(elapsedTimer);
  }
}

function stepState(id) { return state.task?.execution?.steps?.[id]?.status || 'pending'; }
function stepDescription(step) {
  if (step.action === 'drive_relative' && step.target?.direction === 'backward') {
    return `单次限幅后退 ${Number(step.target.distance_m || 0).toFixed(2)}m；后向雷达会看到车身，请先人工确认车后无障碍`;
  }
  if (step.action === 'drive_relative' && ['left', 'right'].includes(step.target?.direction)) {
    const direction = step.target.direction === 'left' ? '左转' : '右转';
    return `雷达保护下分段前进画弧${direction} ${Number(step.target.angle_deg || 0).toFixed(0)}°；净空不足时最多短回退一次，再复测是否继续`;
  }
  return step.description || '';
}
const statusNames = {
  pending: '等待中', completed: '已完成', running: '执行中', awaiting_verification: '待确认',
  failed: '失败', blocked: '已阻塞', paused: '已暂停', confirmed: '已确认', draft: '待确认',
};
const actionNames = {
  inspect_semantic_map: '读取语义地图', patrol_search: 'AI确认观察点并巡航', locate_object: '视觉寻找目标',
  navigate_to: '导航到目标附近', align_for_grasp: '视觉追踪并对准', grasp: '机械臂抓取',
  prepare_stationary_grasp: '准备原地抓取',
  model_grasp: '示教模型抓取', reset_arm: '机械臂复位',
  verify_grasp: '确认抓取', locate_destination: '定位放置点', place_on_floor: '地面放置', release: '放下物体',
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
    return `<div class="plan-step ${className}" ${index === currentIndex ? 'aria-current="step"' : ''}><div class="plan-step-head"><span class="step-index">${status === 'completed' ? '✓' : index + 1}</span><b>${esc(actionNames[step.action] || step.action)}</b><span class="step-status ${status}">${esc(statusNames[status] || status)}</span></div><p>${esc(stepDescription(step))}</p></div>`;
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
  if (['draft', 'paused'].includes(task.status)) return setTaskPrimary({ label: '确认并自动执行计划', mode: 'confirm' });
  const step = steps[currentIndex];
  const status = stepState(step.id);
  const action = actionNames[step.action] || step.action;
  const robotBusy = Boolean(state.robot?.managed?.running && state.robot?.managed?.kind !== 'camera_preview');
  if (status === 'running' || robotBusy && status === 'awaiting_verification')
    return setTaskPrimary({ label: `正在执行：${action}`, disabled: true, mode: 'busy', stepId: step.id });
  if (status === 'awaiting_verification')
    return setTaskPrimary({ label: step.action === 'verify_grasp' ? '我已确认实物被夹起' : `人工确认：${action}`, mode: 'complete', stepId: step.id });
  if (state.autoRunTaskId === task.id && !['failed', 'blocked'].includes(status))
    return setTaskPrimary({ label: `自动执行：${action}`, disabled: true, mode: 'busy', stepId: step.id });
  return setTaskPrimary({ label: `${['failed', 'blocked'].includes(status) ? '重试并继续' : '继续自动执行'}：${action}`, mode: 'execute', stepId: step.id });
}

async function confirmPlan() {
  if (!state.task) return false;
  const taskId = state.task.id;
  setTaskPrimary({ label: '正在确认计划…', disabled: true, mode: 'busy' });
  try {
    const result = await api(`/api/tasks/${taskId}/confirm`, { method: 'POST', body: '{}' });
    if (state.task?.id !== taskId) return false;
    state.task = result.task; renderTask();
    toast('计划已确认，开始自动执行');
    return true;
  } catch (error) { renderTask(); toast(error.message, true); return false; }
}

async function refreshTask() {
  if (!state.task) return;
  const taskId = state.task.id;
  $('refreshTask').disabled = true;
  try {
    const result = await api(`/api/tasks/${taskId}`);
    if (state.task?.id !== taskId) return;
    state.task = result.task; renderTask(); toast('任务状态已刷新');
  } catch (error) { toast(error.message, true); }
  finally { $('refreshTask').disabled = false; }
}

async function executeStep(id) {
  const taskId = state.task?.id;
  if (!taskId) return 'failed';
  setTaskPrimary({ label: '正在发送指令…', disabled: true, mode: 'busy', stepId: id });
  try {
    const result = await api(`/api/tasks/${taskId}/steps/${id}/execute`, { method: 'POST', body: '{}' });
    if (state.task?.id !== taskId) return result.result.status;
    state.task = result.task; renderTask();
    closeResponsivePanels();
    const blockedMessage = result.result.detail?.message || result.result.detail?.decision?.clarification_question || '当前地图信息不足，任务已暂停';
    if (['blocked', 'failed'].includes(result.result.status)) toast(blockedMessage, true);
    return result.result.status;
  } catch (error) {
    if (state.task?.id === taskId) { await refreshTask(); toast(error.message, true); }
    return 'failed';
  }
}

async function completeStep(id) {
  const taskId = state.task?.id;
  if (!taskId) return;
  const step = state.task?.plan?.steps?.find(item => item.id === id);
  if (step?.action === 'verify_grasp' && !confirm('请确认目标实物确实已经被夹起。仅机械臂动作完成不代表夹取成功。')) return;
  setTaskPrimary({ label: '正在确认…', disabled: true, mode: 'busy', stepId: id });
  try {
    const result = await api(`/api/tasks/${taskId}/steps/${id}/complete`, { method: 'POST', body: '{}' });
    if (state.task?.id !== taskId) return;
    state.task = result.task; renderTask(); toast('人工核验已记录');
    if (state.autoRunTaskId === state.task.id) await autoAdvanceTask();
  } catch (error) { toast(error.message, true); }
}

async function autoAdvanceTask({ retryFailed = false } = {}) {
  if (state.autoAdvancing || !state.task || state.autoRunTaskId !== state.task.id) return;
  state.autoAdvancing = true;
  try {
    while (state.task && state.autoRunTaskId === state.task.id) {
      const task = state.task;
      if (['completed', 'draft', 'paused'].includes(task.status)) break;
      const step = (task.plan?.steps || []).find(item => stepState(item.id) !== 'completed');
      if (!step) { state.autoRunTaskId = null; break; }
      const status = stepState(step.id);
      if (status === 'running' || status === 'awaiting_verification') break;
      if (['failed', 'blocked'].includes(status) && !retryFailed) { state.autoRunTaskId = null; break; }
      retryFailed = false;
      const result = await executeStep(step.id);
      if (result !== 'completed') {
        if (['failed', 'blocked'].includes(result)) state.autoRunTaskId = null;
        break;
      }
    }
  } finally {
    state.autoAdvancing = false;
    renderTask();
  }
}

async function runTaskPrimary() {
  const button = $('taskPrimaryAction');
  if (button.disabled || state.actionBusy) return;
  const mode = button.dataset.mode;
  const stepId = button.dataset.stepId;
  state.actionBusy = true;
  renderTask();
  try {
    if (mode === 'confirm') {
      if (await confirmPlan()) { state.autoRunTaskId = state.task.id; await autoAdvanceTask(); }
    }
    else if (mode === 'execute') {
      state.autoRunTaskId = state.task.id;
      await autoAdvanceTask({ retryFailed: true });
    }
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
  if (state.pollBusy) return;
  state.pollBusy = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const robot = await api('/api/robot/state', { signal: controller.signal });
    state.robot = robot;
    const activeStep = state.task?.execution?.active_step;
    const now = Date.now();
    if (activeStep && !state.taskSyncBusy && now - state.lastTaskSyncAt >= 2500) {
      state.taskSyncBusy = true;
      state.lastTaskSyncAt = now;
      try {
        const taskId = state.task.id;
        const result = await api(`/api/tasks/${taskId}`);
        if (state.task?.id === taskId) state.task = result.task;
      } finally {
        state.taskSyncBusy = false;
      }
    }
    if (robot.static_map) state.mapMeta = robot.static_map;
    $('piDot').className = 'ok';
    $('piConnection').textContent = '树莓派桥接在线';
    $('hardwareMode').textContent = robot.hardware_enabled ? '实车模式' : '安全模式';
    const managed = robot.managed || {};
    $('trackingState').textContent = processNames[managed.kind] || managed.kind || '待机';
    const grasp = robot.grasp || {};
    $('graspState').textContent = !grasp.connected ? '抓取服务离线'
      : grasp.data?.ble === false ? '机械臂 BLE 离线'
        : (phaseNames[grasp.data?.grasp?.phase] || grasp.data?.grasp?.phase || '待机');
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
    $('mapToolState').textContent = state.pose
      ? (state.poseFresh ? '定位实时更新' : '查看上次定位') : '底图可查看';
    $('visionToolState').textContent = managed.running && managed.kind === 'visual_pursuit'
      ? '追踪画面可看' : grasp.connected ? '双目预览在线' : '点击查看';
    if (isObserving('vision')) syncVisionSource(robot);
    $('observationAdapter').textContent = robot.interfaces?.observation_viewpoint_selection === 'connected_car1_search_route'
      ? 'car1.0 A* 多观察点已接入' : '观测点适配器离线';
    $('navigationAdapter').textContent = robot.interfaces?.text_map_navigation === 'connected_car1_object_search'
      ? '自然语言 → 语义区域 → 雷达导航' : '导航适配器离线';
    renderRobotLiveStatus(robot);
    renderTask();
    if (state.autoRunTaskId === state.task?.id) void autoAdvanceTask();
  } catch {
    state.robot = null;
    $('piDot').className = '';
    $('piConnection').textContent = '树莓派桥接离线';
    state.poseFresh = false;
    $('mapToolState').textContent = '连接中断';
    $('visionToolState').textContent = '连接中断';
    $('poseFreshness').textContent = '连接中断 · 位置暂未更新';
    $('poseFreshness').classList.add('stale');
    updateRobotMarker();
    renderRobotOffline();
    if (isObserving('vision')) syncVisionSource(null);
  } finally {
    clearTimeout(timeout);
    state.pollBusy = false;
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
  let completed = Number(taskPlan?.completed_steps);
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
  const mapAgeMs = Number(robot.time_ms || Date.now()) - Number(mapState._updated_at || 0);
  const freshMapTask = mapAgeMs >= 0 && mapAgeMs < 5000;
  const companionOwnsStatus = Boolean(robotTask.active || (freshMapTask && mapState.running));
  // A managed pursuit/motion process owns the current hardware state.  The
  // companion service may still contain a failure from an older map mission;
  // never show that stale message over the active process' own status.
  const managedOwnsStatus = Boolean(managed.running
    && !['idle', 'camera_preview'].includes(managed.kind));
  const command = (managedOwnsStatus || !companionOwnsStatus)
    ? {}
    : (companion.robot_command || mapCommand || {});
  const previewing = Boolean(managed.running && managed.kind === 'camera_preview');
  const running = Boolean(managedOwnsStatus || (freshMapTask && (robotTask.active || mapState.running)));
  const completed = state.task ? state.task.status === 'completed'
    : (!running && Boolean(robotTask.completed || taskPlan.status === 'completed'));
  const webProgress = webTaskProgress();
  const processLabel = processNames[managed.kind] || managed.kind || '待机';
  const phaseKey = String(robotTask.phase || command.state || '').toLowerCase();
  const phaseLabel = managedOwnsStatus
    ? processLabel
    : (phaseNames[phaseKey]
      || (companionOwnsStatus ? robotTask.phase_label : '')
      || command.state || processLabel);
  const plannedInstruction = mapState.task?.instruction
    || mapState.navigation?.task?.instruction
    || taskPlan.instruction
    || state.task?.instruction;
  const instruction = running
    ? ((managedOwnsStatus ? state.task?.instruction : null) || plannedInstruction || '正在执行任务')
    : (state.task?.instruction || (previewing ? '查看双目实时画面' : '等待新的执行指令'));
  const detail = managed.status_message
    || (managedOwnsStatus ? `${processLabel}正在运行` : '')
    || command.reason
    || (companionOwnsStatus ? robotTask.phase_label : '')
    || (companionOwnsStatus ? currentPlanProgress(robotTask, taskPlan).active?.detail : '')
    || (running ? `${processLabel}正在运行` : previewing ? '双目预览在线 · 当前没有底盘动作' : '小车已停止，等待下一条指令');
  const robotProgress = currentPlanProgress(robotTask, taskPlan);
  const progress = companionOwnsStatus && !managedOwnsStatus && robotProgress.total ? robotProgress : webProgress;
  const ratio = progress.total ? Math.min(100, progress.completed / progress.total * 100) : (completed ? 100 : 0);
  const failedDetail = webProgress.failed?.[1]?.detail || {};
  const processFailed = !running && managed.returncode != null && managed.returncode !== 0;
  const failureMessage = failedDetail.error || failedDetail.message || managed.error
    || failedDetail.decision?.clarification_question || '步骤未能启动，请查看运行日志';
  const failed = !running && Boolean(webProgress.failed || processFailed);
  const awaitingConfirmation = !running && Boolean(webProgress.active);
  const finishedDetail = managed.returncode === 0
    ? '底层任务已正常结束，等待确认当前步骤'
    : `底层任务已结束（退出码 ${managed.returncode ?? '未知'}），请检查日志`;

  $('robotLiveDot').className = failed ? 'error' : (running ? 'running' : 'online');
  $('robotLiveState').textContent = failed ? '步骤未启动'
    : (awaitingConfirmation ? '等待步骤确认' : (running ? phaseLabel : completed ? '任务完成' : previewing ? '双目预览' : '小车待机'));
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
  const values = [pose.x_m, pose.y_m, yaw, status.rmse_m ?? status.navigation?.map_rmse_m ?? status.localization_quality];
  const fields = $('poseFacts').querySelectorAll('dd');
  fields[0].textContent = values[0] == null ? '—' : `${Number(values[0]).toFixed(2)} m`;
  fields[1].textContent = values[1] == null ? '—' : `${Number(values[1]).toFixed(2)} m`;
  fields[2].textContent = values[2] == null ? '—' : `${Number(values[2]).toFixed(1)}°`;
  fields[3].textContent = values[3] == null ? '—' : String(values[3]);
  state.pose = values[0] == null || values[1] == null ? null : { x: Number(values[0]), y: Number(values[1]), yaw: Number(yaw || 0) };
  const updatedAt = Number(status._updated_at || 0);
  const ageMs = updatedAt > 0 ? Date.now() - updatedAt : NaN;
  state.poseFresh = Boolean(state.pose && Number.isFinite(ageMs) && ageMs >= 0 && ageMs < 10000);
  const freshness = $('poseFreshness');
  freshness.classList.toggle('stale', !state.poseFresh);
  freshness.textContent = !state.pose ? '暂无小车定位坐标'
    : state.poseFresh ? `实时定位 · ${Math.max(0, Math.round(ageMs / 1000))} 秒前更新`
      : Number.isFinite(ageMs) && ageMs >= 0 ? `上次定位 · ${Math.round(ageMs / 1000)} 秒前（非实时）`
        : '上次定位 · 更新时间未知';
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
  // Semantic report images contain margins and legends: they are not raw grid coordinates.
  const isBase = state.mapLoadedLayer === 'base';
  $('mapCoordinateNote').textContent = isBase
    ? '绿色为实时位置，琥珀色为上次定位。'
    : '当前为示意图。切换到雷达底图可查看小车位置标记。';
  marker.classList.toggle('stale', !state.poseFresh);
  if (!isBase || !state.pose || !image.naturalWidth) { marker.hidden = true; return; }
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

function scannedMapBounds(image) {
  if (state.mapLoadedLayer !== 'base') return null;
  try {
    const canvas = document.createElement('canvas');
    const ratio = Math.min(1, 512 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.round(image.naturalWidth * ratio);
    canvas.height = Math.round(image.naturalHeight * ratio);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let x0 = canvas.width, y0 = canvas.height, x1 = 0, y1 = 0, count = 0;
    for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
      const i = (y * canvas.width + x) * 4;
      const value = pixels[i];
      // Standard occupancy raster: black obstacles, white free cells, gray unknown.
      if (pixels[i + 3] && Math.abs(value - pixels[i + 1]) < 8 && Math.abs(value - pixels[i + 2]) < 8
          && (value < 90 || value > 240)) {
        x0 = Math.min(x0, x); y0 = Math.min(y0, y);
        x1 = Math.max(x1, x); y1 = Math.max(y1, y); count++;
      }
    }
    if (count < 20) return null;
    return { x0: x0 / canvas.width, y0: y0 / canvas.height,
      x1: x1 / canvas.width, y1: y1 / canvas.height };
  } catch { return null; }
}

function resetMapView() {
  const view = state.mapView;
  view.scale = 1; view.panX = 0; view.panY = 0;
  const marker = $('robotMarker');
  const frame = $('mapFrame');
  const bounds = scannedMapBounds($('mapImage'));
  let centerX = bounds ? (bounds.x0 + bounds.x1) / 2 : .5;
  let centerY = bounds ? (bounds.y0 + bounds.y1) / 2 : .5;
  if (!marker.hidden) {
    centerX = parseFloat(marker.style.left) / 100;
    centerY = parseFloat(marker.style.top) / 100;
  }
  if (bounds && frame.clientWidth) {
    const spanX = Math.max(.08, Math.abs(centerX - bounds.x0), Math.abs(bounds.x1 - centerX)) * 2;
    const spanY = Math.max(.08, Math.abs(centerY - bounds.y0), Math.abs(bounds.y1 - centerY)) * 2;
    view.scale = Math.max(1, Math.min(7, .84 * Math.min(
      $('mapViewport').clientWidth / (spanX * frame.clientWidth),
      $('mapViewport').clientHeight / (spanY * frame.clientHeight))));
  }
  view.panX = -view.scale * (centerX - .5) * frame.clientWidth;
  view.panY = -view.scale * (centerY - .5) * frame.clientHeight;
  updateMapTransform();
}

function setMapScale(next) {
  state.mapView.scale = Math.max(.5, Math.min(8, next));
  updateMapTransform();
}

function reloadMap({ preserveView = false } = {}) {
  if (preserveView && state.mapLoading) return;
  state.mapLoading = true;
  const image = $('mapImage');
  const layer = state.activeMapLayer || 'relationship_release';
  if (!preserveView || !image.naturalWidth) {
    $('mapEmpty').style.display = 'grid';
    $('mapEmpty').textContent = '正在载入地图…';
  }
  image.onload = () => {
    state.mapLoading = false;
    state.mapLoadedLayer = layer;
    $('mapEmpty').style.display = 'none';
    layoutMapFrame();
    if (preserveView) updateMapTransform(); else resetMapView();
  };
  image.onerror = () => {
    state.mapLoading = false;
    $('mapEmpty').style.display = 'grid';
    $('mapEmpty').textContent = '地图暂时无法加载，请点击上方“刷新”重试，或切换地图图层。';
  };
  image.src = `/api/robot/map/layer/${encodeURIComponent(layer)}.png?t=${Date.now()}`;
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
    const previousLayer = state.activeMapLayer;
    if (!state.activeMapLayer || !available.has(state.activeMapLayer)) {
      state.activeMapLayer = state.mapLayers.find(item => item.id === 'relationship_release')?.id
        || state.mapLayers.find(item => item.is_default)?.id
        || state.mapLayers.find(item => item.id === 'base')?.id
        || state.mapLayers[0]?.id
        || null;
    }
    $('mapLayerSelect').innerHTML = state.mapLayers.map(item =>
      `<option value="${esc(item.id)}">${esc(item.label)}</option>`
    ).join('') || '<option value="">地图不可用</option>';
    $('mapLayerSelect').value = state.activeMapLayer || '';
    if (state.activeMapLayer && previousLayer !== state.activeMapLayer && isObserving('map')) reloadMap();
    const semantic = result.semantic || {};
    if (semantic.warning) $('semanticState').title = semantic.warning;
  } catch (error) {
    state.mapLayers = [];
    $('mapLayerSelect').innerHTML = `<option value="${esc(state.activeMapLayer || 'relationship_release')}">地图图层列表暂不可用</option>`;
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

const videoRetries = new Map();
function cameraPlaceholder(side, title, detail) {
  const node = $(`${side}VideoEmpty`);
  node.hidden = false;
  node.querySelector('b').textContent = title;
  node.querySelector('small').textContent = detail;
}

function connectVideoStreams() {
  const stamp = Date.now();
  ['left', 'right'].forEach(side => {
    const image = $(`${side}Video`);
    const label = $(`${side}CameraState`);
    label.textContent = '连接中';
    cameraPlaceholder(side, '正在连接画面', '连接后会自动显示');
    image.onload = () => {
      if (state.visionMode !== 'preview') return;
      label.textContent = '实时';
      $(`${side}VideoEmpty`).hidden = true;
    };
    image.onerror = () => {
      label.textContent = '等待连接';
      cameraPlaceholder(side, '暂时没有画面', '可重新连接，或在空闲时启动预览');
      clearTimeout(videoRetries.get(side));
      videoRetries.set(side, setTimeout(() => {
        if (state.visionMode === 'preview' && isObserving('vision'))
          image.src = `/api/robot/video/${side}?t=${Date.now()}`;
      }, 5000));
    };
    image.src = `/api/robot/video/${side}?t=${stamp}`;
  });
}

function disconnectVideoStreams() {
  for (const side of ['left', 'right']) {
    clearTimeout(videoRetries.get(side));
    const image = $(`${side}Video`);
    image.onload = image.onerror = null;
    image.removeAttribute('src');
  }
  videoRetries.clear();
}

let pursuitStreamRetry = null;
function disconnectPursuitStream() {
  clearTimeout(pursuitStreamRetry);
  pursuitStreamRetry = null;
  const image = $('pursuitVideo');
  image.onload = image.onerror = null;
  image.removeAttribute('src');
}

function connectPursuitStream() {
  const image = $('pursuitVideo');
  $('pursuitCameraState').textContent = '连接中';
  $('pursuitVideoEmpty').hidden = false;
  image.onload = () => {
    if (state.visionMode !== 'pursuit') return;
    $('pursuitCameraState').textContent = '实时';
    $('pursuitVideoEmpty').hidden = true;
  };
  image.onerror = () => {
    $('pursuitCameraState').textContent = '重新连接中';
    $('pursuitVideoEmpty').hidden = false;
    clearTimeout(pursuitStreamRetry);
    if (state.visionMode === 'pursuit' && isObserving('vision'))
      pursuitStreamRetry = setTimeout(connectPursuitStream, 5000);
  };
  image.src = `/api/robot/pursuit/stream.mjpg?t=${Date.now()}`;
}

function visionModeFor(robot) {
  const managed = robot?.managed || {};
  if (managed.running && managed.kind === 'visual_pursuit') return 'pursuit';
  if (managed.running && managed.kind === 'camera_preview') return 'preview';
  // Do not hide healthy streams just because one status poll or one camera is delayed.
  if (robot?.grasp?.connected) return 'preview';
  if (managed.running) return 'unavailable';
  return 'preview';
}

function syncVisionSource(robot) {
  const managed = robot?.managed || {};
  const busy = Boolean(managed.running && managed.kind !== 'camera_preview');
  $('calibrateCameras').disabled = busy || Boolean(state.cameraCalibrating);
  $('swapCameras').disabled = busy;
  $('startPreview').disabled = busy;
  $('stopPreview').disabled = busy || managed.kind !== 'camera_preview';
  const mode = visionModeFor(robot);
  $('visionSourceNote').textContent = mode === 'pursuit'
    ? '视觉追踪运行中 · 已接入追踪画面。查看和切换页面不会中断任务。'
    : mode === 'preview' && robot?.grasp?.connected ? '双目画面在线 · 左右顺序沿用已保存的校准。'
      : busy ? '当前任务占用摄像头；画面暂不可用，任务切换后会自动恢复。'
        : !robot ? '正在连接小车。画面会自动重连，地图仍可独立查看。'
          : '正在请求双目画面。如摄像头尚未启动，可点击“启动预览”。';
  if (mode === state.visionMode) return;
  disconnectVideoStreams();
  disconnectPursuitStream();
  state.visionMode = mode;
  $('previewCameraGrid').hidden = mode === 'pursuit';
  $('pursuitCameraCard').hidden = mode !== 'pursuit';
  if (mode === 'preview') connectVideoStreams();
  if (mode === 'pursuit') connectPursuitStream();
  if (mode === 'unavailable') for (const side of ['left', 'right']) {
    $(`${side}CameraState`).textContent = '任务占用中';
    cameraPlaceholder(side, '等待摄像头可用', '当前任务结束后自动连接');
  }
}

function reconnectVision() {
  state.visionMode = null;
  disconnectVideoStreams();
  disconnectPursuitStream();
  syncVisionSource(state.robot);
  void pollRobot();
}

async function startPreview() {
  try {
    await api('/api/robot/preview/start', { method: 'POST', body: '{}' });
    toast('双目预览正在启动');
    setTimeout(pollRobot, 1300);
  } catch (error) { toast(error.message, true); }
}

async function stopPreview() {
  try {
    await api('/api/robot/preview/stop', { method: 'POST', body: '{}' });
    disconnectVideoStreams();
    state.visionMode = null;
    $('leftCameraState').textContent = '已停止';
    $('rightCameraState').textContent = '已停止';
    toast('双目预览已停止');
    void pollRobot();
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
    renderForwardReference(result.forward_reference);
  } catch (error) {
    $('cameraOrderState').textContent = '状态不可用';
  }
}

function renderForwardReference(reference) {
  for (const side of ['left', 'right']) {
    const image = $(side + 'Video');
    const svg = $(side + 'ForwardLine');
    const points = reference?.enabled && reference.lines?.[side];
    if (!points || points.length !== 2 || !points.every(point => Array.isArray(point) && point.length === 2)
        || !points.flat().every(value => Number.isFinite(value) && value >= 0 && value <= 1)) {
      svg.toggleAttribute('hidden', true);
      continue;
    }
    const line = svg.querySelector('line');
    for (const [name, value] of Object.entries({ x1: points[0][0], y1: points[0][1], x2: points[1][0], y2: points[1][1] })) {
      line.setAttribute(name, String(value * 1000));
    }
    const position = () => {
      if (!image.naturalWidth || !image.naturalHeight) { svg.toggleAttribute('hidden', true); return; }
      if (reference.frame_size?.[0] !== image.naturalWidth || reference.frame_size?.[1] !== image.naturalHeight) {
        svg.toggleAttribute('hidden', true);
        return;
      }
      const scale = Math.min(image.clientWidth / image.naturalWidth, image.clientHeight / image.naturalHeight);
      const width = image.naturalWidth * scale, height = image.naturalHeight * scale;
      Object.assign(svg.style, { width: width + 'px', height: height + 'px',
        left: (image.clientWidth - width) / 2 + 'px', top: (image.clientHeight - height) / 2 + 'px' });
      svg.toggleAttribute('hidden', false);
    };
    if (image._forwardLineLoad) image.removeEventListener('load', image._forwardLineLoad);
    image._forwardLineLoad = position;
    image.addEventListener('load', position);
    image._forwardLineResize?.disconnect();
    image._forwardLineResize = new ResizeObserver(position);
    image._forwardLineResize.observe(image);
    position();
  }
}

async function calibrateCameras() {
  if (state.robot?.managed?.running && state.robot.managed.kind !== 'camera_preview') return;
  state.cameraCalibrating = true;
  $('calibrateCameras').disabled = true;
  $('cameraOrderState').textContent = 'Qwen 正在比较双目视差…';
  try {
    const result = await api('/api/robot/camera-order/calibrate', { method: 'POST', body: '{}' });
    renderCameraOrder(result.camera_order);
    await loadCameraOrder();
    if (state.visionMode === 'preview') {
      disconnectVideoStreams();
      connectVideoStreams();
    }
    const inference = result.inference || {};
    toast(result.applied
      ? `左右校准完成：${inference.order === 'swapped' ? '已自动交换' : '顺序正常'}`
      : 'AI 证据不足，未自动更改；可使用手动交换', !result.applied);
  } catch (error) {
    toast(error.message, true);
    await loadCameraOrder();
  } finally {
    state.cameraCalibrating = false;
    $('calibrateCameras').disabled = Boolean(state.robot?.managed?.running && state.robot.managed.kind !== 'camera_preview');
  }
}

async function swapCameras() {
  try {
    const result = await api('/api/robot/camera-order/swap', { method: 'POST', body: '{}' });
    renderCameraOrder(result.camera_order);
    await loadCameraOrder();
    if (state.visionMode === 'preview') {
      disconnectVideoStreams();
      connectVideoStreams();
    }
    toast(result.camera_order.swapped ? '左右画面已交换' : '已恢复原始顺序');
  } catch (error) { toast(error.message, true); }
}

async function robotControl(name) {
  if (name === 'emergency-stop') state.autoRunTaskId = null;
  try { await api(`/api/robot/${name}`, { method: 'POST', body: '{}' }); toast('指令已发送'); }
  catch (error) { toast(error.message, true); }
}

async function loadLogs() {
  try {
    const result = await api('/api/robot/logs/list');
    const select = $('logSelect');
    const selected = select.value;
    select.innerHTML = '<option value="">当前动作</option>' + (result.logs || []).map(item => {
      const stamp = new Date(item.modified_at).toLocaleString([], { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
      return `<option value="${esc(item.name)}">${esc(stamp)} · ${esc(item.name)}</option>`;
    }).join('');
    if ([...select.options].some(option => option.value === selected)) select.value = selected;
  } catch (error) { $('logOutput').textContent = error.message; }
  await logs();
}

async function logs() {
  try {
    const name = $('logSelect').value;
    const result = await api(`/api/robot/logs${name ? `?name=${encodeURIComponent(name)}` : ''}`);
    $('logOutput').textContent = result.text || '当前进程尚无日志。';
  } catch (error) { $('logOutput').textContent = error.message; }
}

function isObserving(kind) {
  return state.activeTab === 'observe' && (state.observeView === 'split' || state.observeView === kind);
}

function showWorkspace(name, view = state.observeView) {
  state.activeTab = name;
  state.observeView = view;
  document.querySelectorAll('.tab').forEach(button => {
    const active = button.dataset.tab === name;
    button.classList.toggle('active', active);
    button.setAttribute('aria-current', active ? 'page' : 'false');
  });
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === `tab-${name}`));
  $('observeGrid').dataset.view = view;
  document.querySelectorAll('.view-switch button').forEach(button => {
    const active = button.dataset.observeView === view;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  closeResponsivePanels();
  if (isObserving('map')) {
    // Load the annotated release immediately; metadata must not block the image.
    if (!$('mapImage').getAttribute('src')) reloadMap();
    requestAnimationFrame(() => { layoutMapFrame(); updateMapTransform(); });
    if (!state.mapLayers.length) void loadMapLayers();
  }
  if (isObserving('vision')) { syncVisionSource(state.robot); void loadCameraOrder(); }
  else {
    state.visionMode = null;
    disconnectVideoStreams();
    disconnectPursuitStream();
  }
  if (name === 'logs') void loadLogs();
  if (name === 'agent') requestAnimationFrame(() => scrollMessages());
}

function openModal(name) {
  if (name === 'map' || name === 'vision') { showWorkspace('observe', name); return; }
  const modal = $(`${name}Modal`);
  modal.hidden = false;
  document.body.classList.add('modal-open');
  modal.querySelector('.modal-close')?.focus();
  if (name === 'settings') loadSettings().catch(error => toast(error.message, true));
}

function closeModal(modal) {
  if (!modal || modal.hidden) return;
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
  $('testDeepSeek').style.display = provider === 'deepseek_flash_api' ? 'block' : 'none';
  const details = providerDetails(provider);
  $('modelInput').value = details?.model || '';
  $('apiKeyInput').value = '';
  if (details?.client_managed_key) {
    const configured = Boolean(deviceDeepSeekKey());
    setKeyStatus(configured
      ? '已保存在当前浏览器，留空保持不变'
      : '尚未在当前浏览器配置');
  } else {
    setKeyStatus(details?.configured ? '已配置，留空保持不变' : '尚未配置');
  }
}

async function testDeepSeekConnection() {
  const button = $('testDeepSeek');
  const key = $('apiKeyInput').value.trim() || deviceDeepSeekKey();
  const model = $('modelInput').value.trim() || 'deepseek-flash';
  if (!key) return setKeyStatus('请先填写 API 密钥', 'error');
  button.disabled = true;
  button.textContent = '测试中…';
  setKeyStatus('正在从当前浏览器连接 DeepSeek…');
  const startedAt = Date.now();
  reportClientPlannerEvent('connection_test', 'started', { model });
  try {
    const response = await deepSeekFetch(DEEPSEEK_MODELS_URL, {
      method: 'GET',
      headers: { 'Authorization': `Bearer ${key}`, 'Accept': 'application/json' },
    }, DEEPSEEK_TEST_TIMEOUT_MS);
    const value = await responseJson(response, 'DeepSeek');
    const models = Array.isArray(value?.data) ? value.data.map(item => item?.id).filter(Boolean) : [];
    if (models.length && !models.includes(model)) {
      throw new Error(`连接正常，但账号模型列表中没有 ${model}；可用：${models.join('、')}`);
    }
    setKeyStatus(`连接正常，模型 ${model} 可用`, 'ok');
    reportClientPlannerEvent('connection_test', 'completed', { model, duration_ms: Date.now() - startedAt });
    toast('DeepSeek 连接测试成功');
  } catch (error) {
    reportClientPlannerEvent('connection_test', 'failed', {
      model, duration_ms: Date.now() - startedAt, message: error?.message || '连接测试失败',
    });
    setKeyStatus(error?.message || '连接测试失败', 'error');
    toast(error?.message || 'DeepSeek 连接测试失败', true);
  } finally {
    button.disabled = false;
    button.textContent = translations[state.settings?.language || 'zh-CN']?.testConnection || '测试连接';
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
  button.onclick = () => showWorkspace(button.dataset.tab);
});
document.querySelectorAll('[data-observe-view]').forEach(button => {
  button.onclick = () => showWorkspace('observe', button.dataset.observeView);
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
new ResizeObserver(() => {
  if (isObserving('map')) { layoutMapFrame(); updateMapTransform(); }
}).observe(viewport);

$('newChat').onclick = () => { showWorkspace('agent'); newConversation(); };
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
$('toggleMission').onclick = () => {
  if (state.activeTab !== 'agent') showWorkspace('agent');
  if (window.matchMedia('(max-width:1050px)').matches) toggleResponsivePanel('mission');
  else $('missionPanel').scrollIntoView({ block: 'nearest' });
};
$('closeMission').onclick = closeResponsivePanels;
$('uiBackdrop').onclick = closeResponsivePanels;
$('reloadMap').onclick = () => reloadMap();
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
$('reconnectVision').onclick = reconnectVision;
$('stopPreview').onclick = stopPreview;
$('calibrateCameras').onclick = calibrateCameras;
$('swapCameras').onclick = swapCameras;
$('refreshLogs').onclick = loadLogs;
$('logSelect').onchange = logs;
$('openMap').onclick = () => openModal('map');
$('openVision').onclick = () => openModal('vision');
$('openSettings').onclick = () => openModal('settings');
$('providerSelect').onchange = updateSettingsFields;
$('languageSelect').onchange = () => applyLanguage($('languageSelect').value);
$('settingsForm').onsubmit = saveSettings;
$('clearApiKey').onclick = clearApiKey;
$('testDeepSeek').onclick = testDeepSeekConnection;
$('emergencyStop').onclick = () => {
  if (confirm('立即停止底盘、追踪任务和机械臂？')) robotControl('emergency-stop');
};
$('startMapping').onclick = () => robotControl('mapping/start');
$('stopMapping').onclick = () => robotControl('mapping/stop');

loadSettings().catch(error => toast(error.message, true));
loadConversations().then(() => { if (!state.conversationId) newConversation(); }).catch(error => toast(error.message, true));
showWorkspace('observe', 'split');
void loadSemanticMap();
pollRobot();
setInterval(pollRobot, 1500);
setInterval(() => {
  if (isObserving('map')) reloadMap({ preserveView: true });
}, 8000);
