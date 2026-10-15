// Explicit physical confirmation, independent of automatic pick/place plans.
const actions = document.querySelector('.header-actions');
if (actions && !document.getElementById('confirmEmptyGripper')) {
  const button = document.createElement('button');
  button.id = 'confirmEmptyGripper';
  button.type = 'button';
  button.className = 'plan-toggle';
  button.textContent = '确认空夹爪';
  button.title = '夹爪已无物品时清除旧持物状态，不驱动机械臂';
  actions.prepend(button);
  button.addEventListener('click', async () => {
    if (!window.confirm('请确认夹爪已经没有物品。这里只清除持物记录，不移动机械臂或小车。')) return;
    button.disabled = true;
    try {
      const response = await fetch('/api/robot/state', {cache: 'no-store'});
      const state = await response.json();
      if (!response.ok || state.ok === false) throw new Error(state.error || '无法读取小车状态');
      const reset = await fetch('/api/robot/confirm-empty', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({confirmed_empty: true, expected_carry_id: state.robot_skills?.carry_id ?? null})
      });
      const result = await reset.json();
      if (!reset.ok || result.ok !== true) throw new Error(result.detail || result.error || '状态复位失败');
      window.alert(result.message || '空夹爪状态已复位，可以开始新的夹取任务');
    } catch (error) {
      window.alert(error.message || String(error));
    } finally {
      button.disabled = false;
    }
  });
}
