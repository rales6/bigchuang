/* Standalone extension: do not replace the deployed robot-control script. */
(() => {
  const list = document.getElementById('conversations');
  const menu = document.createElement('div');
  menu.className = 'history-menu'; menu.hidden = true; menu.setAttribute('role', 'menu');
  const remove = document.createElement('button');
  remove.type = 'button'; remove.setAttribute('role', 'menuitem'); menu.append(remove);
  document.body.append(menu);
  const dialog = document.createElement('dialog');
  dialog.className = 'history-delete-dialog';
  dialog.innerHTML = '<h2 id="historyDeleteTitle"></h2><p class="history-delete-name"></p><p class="history-delete-hint"></p><div class="history-delete-error" role="alert"></div><div class="history-delete-actions"><button type="button" data-cancel></button><button type="button" data-confirm class="history-delete-confirm"></button></div>';
  dialog.setAttribute('aria-labelledby', 'historyDeleteTitle'); document.body.append(dialog);
  let selected = null, origin = null, busy = false;
  const english = () => state.settings?.language === 'en-US';
  const tr = (zh, en) => english() ? en : zh;
  function closeMenu(restore = false) {
    menu.hidden = true;
    if (restore && origin?.isConnected) origin.focus();
  }
  function decorate() {
    list.querySelectorAll('.conversation').forEach(row => {
      row.tabIndex = 0;
      if (row.querySelector('.history-more')) return;
      const button = document.createElement('button'); button.type = 'button';
      button.className = 'history-more'; button.textContent = '⋯';
      button.setAttribute('aria-label', tr('聊天记录选项', 'Conversation options'));
      button.setAttribute('aria-haspopup', 'menu'); row.append(button);
    });
  }
  new MutationObserver(decorate).observe(list, {childList:true}); decorate();
  function showMenu(row, x, y, focusOrigin) {
    const copy = row.cloneNode(true); copy.querySelectorAll('small,button').forEach(node => node.remove());
    selected = {id:row.dataset.id, title:copy.textContent.trim()}; origin = focusOrigin || row;
    remove.textContent = tr('删除聊天记录', 'Delete conversation');
    menu.hidden = false;
    menu.style.left = `${Math.max(8, Math.min(x, innerWidth - menu.offsetWidth - 8))}px`;
    menu.style.top = `${Math.max(8, Math.min(y, innerHeight - menu.offsetHeight - 8))}px`;
    remove.focus();
  }
  list.addEventListener('contextmenu', event => {
    const row = event.target.closest('.conversation'); if (!row) return;
    event.preventDefault(); showMenu(row, event.clientX, event.clientY);
  });
  list.addEventListener('click', event => {
    const button = event.target.closest('.history-more'); if (!button) return;
    event.preventDefault(); event.stopImmediatePropagation();
    const rect = button.getBoundingClientRect(); showMenu(button.closest('.conversation'), rect.left, rect.bottom, button);
  }, true);
  list.addEventListener('keydown', event => {
    const row = event.target.closest('.conversation'); if (!row) return;
    if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) {
      event.preventDefault(); const rect = row.getBoundingClientRect(); showMenu(row, rect.left + 16, rect.bottom);
    } else if (event.target === row && ['Enter', ' '].includes(event.key)) { event.preventDefault(); row.click(); }
  });
  document.addEventListener('pointerdown', event => { if (!menu.contains(event.target)) closeMenu(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !menu.hidden) { event.preventDefault(); closeMenu(true); } });
  list.addEventListener('scroll', () => closeMenu()); window.addEventListener('resize', () => closeMenu());
  remove.onclick = () => {
    closeMenu();
    dialog.querySelector('h2').textContent = tr('删除这条聊天记录？', 'Delete this conversation?');
    dialog.querySelector('.history-delete-name').textContent = selected.title;
    dialog.querySelector('.history-delete-hint').textContent = tr('确认后将从聊天历史移除。底层任务与运行日志保留；不会停止或启动小车。正在执行的任务不能删除。', 'Remove it from chat history. Diagnostic logs are retained. This does not stop or start the robot. Active tasks cannot be deleted.');
    dialog.querySelector('[data-cancel]').textContent = tr('取消', 'Cancel');
    dialog.querySelector('[data-confirm]').textContent = tr('删除', 'Delete');
    dialog.querySelector('.history-delete-error').textContent = '';
    dialog.showModal(); dialog.querySelector('[data-cancel]').focus();
  };
  dialog.querySelector('[data-cancel]').onclick = () => { if (!busy) dialog.close(); };
  dialog.addEventListener('cancel', event => { if (busy) event.preventDefault(); });
  dialog.addEventListener('close', () => { if (origin?.isConnected) origin.focus(); });
  dialog.querySelector('[data-confirm]').onclick = async () => {
    if (busy) return;
    if (state.actionBusy || state.autoAdvancing || state.autoRunTaskId || $('sendButton').disabled) {
      dialog.querySelector('.history-delete-error').textContent = tr('请等待当前操作结束或先停止任务。', 'Wait for the current operation or stop the task first.'); return;
    }
    busy = true; const target = {...selected};
    dialog.querySelectorAll('button').forEach(button => button.disabled = true);
    try {
      await api(`/api/conversations/${encodeURIComponent(target.id)}`, {method:'DELETE'});
      try { localStorage.removeItem(`${DRAFT_PREFIX}${target.id}`); } catch { /* Storage can be disabled. */ }
      if (state.conversationId === target.id) {
        state.conversationId = null; state.task = null; state.autoRunTaskId = null;
        $('chatInput').value = ''; $('conversationTitle').textContent = tr('小车工作台', 'Robot workspace');
        $('messages').innerHTML = '<div class="empty-state"><h2></h2></div>';
        $('messages').querySelector('h2').textContent = tr('聊天已删除，可新建任务开始对话', 'Conversation deleted. Create a new task to start a chat.');
        renderTask();
      }
      dialog.close(); await loadConversations();
      toast(tr('聊天记录已删除，运行日志保留', 'Conversation deleted. Diagnostic logs retained.'));
    } catch (error) {
      dialog.querySelector('.history-delete-error').textContent = error.message;
    } finally {
      busy = false; dialog.querySelectorAll('button').forEach(button => button.disabled = false);
    }
  };
})();
