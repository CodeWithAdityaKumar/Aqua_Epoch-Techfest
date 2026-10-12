import { ApiError, RequestGate, ValidationError, availability, featuresFor, inferencePath, modelFor, requestJSON, serializeFields } from './core.js';
import { element, renderResult } from './render.js';

/** Exported for DOM tests; production starts once against the local FastAPI origin. */
export function createApp(document, { fetchImpl = globalThis.fetch, confirm = message => globalThis.confirm(message), timeoutMs = 20000 } = {}) {
  const $ = id => document.getElementById(id);
  const api = (path, options = {}) => requestJSON(path, { fetchImpl, timeoutMs, ...options });
  const inferenceGate = new RequestGate();
  const metadataGate = new RequestGate();
  const historyGate = new RequestGate();
  const recordGate = new RequestGate();
  let metadata = null;
  let controls = new Map();
  let refreshing = false;
  let pending = null;
  let source = 'empty';
  let unknownSavedOutcome = false;
  let historyBusy = false;
  const task = () => $('task').value;
  const mode = () => $('mode').value;

  function setStatus(id, text = '', state = 'neutral') {
    const node = $(id);
    node.textContent = text;
    node.classList.remove('status-loading', 'status-ready', 'status-success', 'status-warning', 'status-error', 'status-neutral');
    node.classList.add('status', `status-${state}`);
  }

  function saveNotice(text = '') {
    const message = [text, unknownSavedOutcome ? 'A previous save request was interrupted; its outcome is unknown. Check local history before retrying. No request is retried automatically.' : ''].filter(Boolean).join(' ');
    setStatus('save-status', message, unknownSavedOutcome || /not saved|unknown/i.test(text) ? 'warning' : 'neutral');
  }

  function clearErrors() {
    $('form-errors').hidden = true;
    $('form-errors').replaceChildren();
    for (const { input, error } of controls.values()) { input.removeAttribute('aria-invalid'); error.textContent = ''; }
  }

  function showErrors(errors, message = 'Check the measurements below.') {
    clearErrors();
    const box = $('form-errors');
    box.append(element(document, 'strong', message));
    const list = element(document, 'ul');
    for (const [name, text] of Object.entries(errors)) {
      const control = controls.get(name);
      const item = element(document, 'li');
      if (control) {
        control.input.setAttribute('aria-invalid', 'true');
        control.error.textContent = text;
        const link = element(document, 'a', `${name}: ${text}`);
        link.href = `#${control.input.id}`;
        item.append(link);
      } else item.textContent = name === '_form' ? text : `${name}: ${text}`;
      list.append(item);
    }
    box.append(list);
    box.hidden = false;
    box.focus();
  }

  function readEntries() {
    return Object.fromEntries([...controls].map(([name, control]) => [name, { value: control.input.value, missing: control.missing.checked }]));
  }

  function sourceNotice() {
    const message = {
      empty: '',
      entered: '',
      synthetic: 'Example values — not a real sample',
      edited: 'Edited example values — not a real sample',
    }[source];
    setStatus('input-source', message, ['synthetic', 'edited'].includes(source) ? 'warning' : 'neutral');
  }

  function updateActions() {
    const state = availability(metadata, task(), mode());
    const unavailable = `Unavailable for this task. ${state.message}`;
    setStatus('availability', refreshing ? 'Connecting…' : state.ready ? 'Ready' : unavailable, refreshing ? 'loading' : state.ready ? 'ready' : 'error');
    $('submit-prediction').disabled = refreshing || !state.ready || !!pending;
    $('submit-prediction').textContent = pending ? 'Checking…' : 'Get result';
    const refs = modelFor(metadata, task(), mode())?.reference_ranges;
    let syntheticReady = false;
    try { syntheticReady = featuresFor(metadata, task(), mode()).every(field => Number.isFinite(refs?.[field.name]?.median)); } catch { /* schema unavailable */ }
    $('synthetic-example').disabled = refreshing || !state.ready || !syntheticReady;
  }

  function invalidate({ editing = false } = {}) {
    if (pending?.saveHistory) { unknownSavedOutcome = true; saveNotice(); }
    pending = null;
    inferenceGate.invalidate();
    $('prediction-form').setAttribute('aria-busy', 'false');
    setStatus('request-status', '', 'neutral');
    $('live-results').hidden = true;
    $('result-content').replaceChildren();
    clearErrors();
    if (editing) { source = ['synthetic', 'edited'].includes(source) ? 'edited' : 'entered'; sourceNotice(); }
    updateActions();
  }

  function buildFields(previous = {}) {
    controls = new Map();
    $('fields').replaceChildren();
    $('fields').setAttribute('aria-busy', 'true');
    $('mode-help').textContent = task() === 'regression'
      ? 'Estimated pH has limited reliability; it is not a lab measurement.'
      : task() === 'combined' && mode() === 'full'
        ? 'Measured pH is optional. Leave it blank or select Not measured to estimate pH.'
      : mode() === 'quick' ? 'Quick check cannot confirm drinking safety.'
        : 'More readings still cannot confirm drinking safety.';
    let features;
    try { features = featuresFor(metadata, task(), mode()); }
    catch { $('fields').setAttribute('aria-busy', 'false'); updateActions(); return; }
    features.forEach((field, index) => {
      const id = `measurement-${index}`;
      const group = element(document, 'div', null, 'field');
      const combinedPHOptional = task() === 'combined' && mode() === 'full' && field.name === 'pH';
      const names = { TDS: 'TDS · total dissolved solids', Turbidity: 'Turbidity · cloudiness', Organic_carbon: 'Organic carbon', Trihalomethanes: 'Trihalomethanes', pH: combinedPHOptional ? 'Measured pH · optional' : 'Measured pH' };
      const label = element(document, 'label', names[field.name] || field.name.replaceAll('_', ' '));
      label.htmlFor = id;
      const helpDetails = element(document, 'details', null, 'field-help');
      helpDetails.append(element(document, 'summary', 'Help'));
      const help = element(document, 'p', `Engineering range: ${field.engineering_bounds[0]}–${field.engineering_bounds[1]}; not safe drinking-water limits. API key: ${field.name}.`, 'help');
      help.id = `${id}-help`;
      helpDetails.append(help);
      const input = element(document, 'input');
      input.type = 'number';
      input.inputMode = 'decimal';
      input.step = 'any';
      input.min = String(field.engineering_bounds[0]);
      input.max = String(field.engineering_bounds[1]);
      input.id = id;
      input.name = field.name;
      input.required = field.required === true && !combinedPHOptional;
      input.autocomplete = 'off';
      input.setAttribute('aria-describedby', `${help.id} ${id}-error`);
      const missingLabel = element(document, 'label', null, 'check-label');
      const missing = element(document, 'input');
      missing.type = 'checkbox';
      missing.id = `${id}-missing`;
      missing.disabled = field.nullable !== true;
      missingLabel.htmlFor = missing.id;
      missingLabel.append(missing, document.createTextNode('Not measured'));
      missing.setAttribute('aria-label', `${field.name}: Not measured`);
      const old = Object.hasOwn(previous, field.name) ? previous[field.name] : null;
      input.value = old?.value ?? '';
      missing.checked = !!old?.missing && field.nullable === true;
      input.disabled = missing.checked;
        input.required = field.required === true && !missing.checked && !combinedPHOptional;
      const error = element(document, 'p', null, 'field-error');
      error.id = `${id}-error`;
      input.addEventListener('input', () => invalidate({ editing: true }));
      missing.addEventListener('change', () => {
        input.disabled = missing.checked;
        input.required = field.required === true && !missing.checked;
        invalidate({ editing: true });
      });
      controls.set(field.name, { input, missing, error });
      group.append(label, input, missingLabel, error, helpDetails);
      $('fields').append(group);
    });
    $('fields').setAttribute('aria-busy', 'false');
    updateActions();
  }

  async function refreshMetadata() {
    invalidate();
    refreshing = true;
    updateActions();
    setStatus('service-status', 'Connecting…', 'loading');
    const ticket = metadataGate.begin();
    const results = await Promise.allSettled([
      api('/models', { signal: ticket.signal }), api('/health', { signal: ticket.signal }),
    ]);
    if (!metadataGate.current(ticket)) return;
    refreshing = false;
    $('model-status').replaceChildren();
    const [models, health] = results;
    if (models.status === 'fulfilled' && models.value?.models) {
      metadata = models.value;
      for (const [id, entry] of Object.entries(metadata.models)) {
        $('model-status').append(element(document, 'li', `${id}: ${entry.available ? `available · release ${entry.release_id}` : `unavailable (${entry.reason || 'unknown reason'})`}`, entry.available ? 'model-available' : 'model-unavailable'));
      }
      const readyCount = Object.values(metadata.models).filter(entry => entry.available).length;
      const modelCount = Object.keys(metadata.models).length;
      const healthMessage = `${readyCount}/${modelCount} demo models available. ${health.status === 'fulfilled'
        ? health.value.ready ? 'Service health: ready.' : 'Service health: degraded; available independent tasks may still work.'
        : `Health check unavailable: ${health.reason.message}`}`;
      setStatus('service-status', healthMessage, readyCount === modelCount && health.status === 'fulfilled' && health.value.ready ? 'ready' : 'warning');
    } else {
      metadata = null;
      setStatus('service-status', `API metadata unavailable. ${models.status === 'rejected' ? models.reason.message : 'Invalid model metadata response.'} Check the local server, then refresh metadata.`, 'error');
    }
    // Read at completion: edits and task changes made during the fetch win.
    if (metadata) buildFields(readEntries());
    else updateActions();
  }

  async function submit(event) {
    event?.preventDefault();
    invalidate();
    const state = availability(metadata, task(), mode());
    if (refreshing || !state.ready) { showErrors({ _form: state.message }, 'Inference is unavailable.'); return; }
    let inputs;
    try {
      const entries = readEntries();
      // In the combined full flow, an empty pH is an intentional omission:
      // the regression model estimates it and the classifier handles null via
      // its trained imputation pipeline. Other blank fields remain errors.
      if (task() === 'combined' && mode() === 'full' && entries.pH && !entries.pH.missing && entries.pH.value.trim() === '') {
        entries.pH = { ...entries.pH, missing: true };
      }
      inputs = serializeFields(featuresFor(metadata, task(), mode()), entries, task() === 'combined' ? featuresFor(metadata, 'regression', mode()) : null);
    } catch (error) {
      showErrors(error instanceof ValidationError ? error.errors : { _form: error.message });
      return;
    }
    const snapshot = {
      inputs, task: task(), mode: mode(), synthetic: source === 'synthetic', editedSynthetic: source === 'edited',
      references: modelFor(metadata, task(), mode())?.reference_ranges,
      saveHistory: $('save-history').checked,
    };
    const ticket = inferenceGate.begin();
    pending = snapshot;
    updateActions();
    $('prediction-form').setAttribute('aria-busy', 'true');
    setStatus('request-status', 'Checking…', 'loading');
    try {
      const response = await api(inferencePath(snapshot.task, snapshot.mode, snapshot.saveHistory), {
        method: 'POST', signal: ticket.signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(inputs),
      });
      if (!inferenceGate.current(ticket)) return;
      renderResult(document, $('result-content'), response, snapshot);
      $('live-results').hidden = false;
      $('result-heading').focus();
      setStatus('request-status', 'Result ready', 'success');
      if (snapshot.saveHistory) {
        if (response.history?.saved) { saveNotice(`Saved to local history · record ${response.history.id}.`); void refreshHistory(); }
        else saveNotice(response.history ? `Result received, but history was not saved: ${response.history.error || 'no reason supplied'}.`
          : 'Result received, but the server did not confirm whether history was saved. Check history before retrying.');
      } else saveNotice();
    } catch (error) {
      if (!inferenceGate.current(ticket)) return;
      const errors = {};
      if (error instanceof ApiError) {
        for (const detail of error.details) {
          const field = Array.isArray(detail.loc) ? detail.loc[detail.loc.length - 1] : '_form';
          errors[field] = [errors[field], detail.msg].filter(Boolean).join(' ');
        }
      }
      if (!Object.keys(errors).length) errors._form = error.message;
      showErrors(errors, error instanceof ApiError ? error.message : 'Unable to complete inference.');
      setStatus('request-status', 'Request failed. Please review the error.', 'error');
      if (snapshot.saveHistory && (!(error instanceof ApiError) || error.status >= 500)) {
        unknownSavedOutcome = true;
        saveNotice('Save outcome unknown after request failure.');
      }
    } finally {
      if (inferenceGate.current(ticket)) {
        pending = null;
        $('prediction-form').setAttribute('aria-busy', 'false');
        updateActions();
      }
    }
  }

  function historySummary(item) {
    const timestamp = formatTimestamp(item.created_at);
    const parts = [`${timestamp} · ${item.task} · ${item.mode} mode · release ${item.release_id}`];
    if (item.assessment) parts.push(`Assessment: ${item.assessment}`);
    if (Number.isFinite(item.probability_safe)) parts.push(`Uncalibrated P(class 1): ${item.probability_safe}`);
    if (Number.isFinite(item.ph_estimate)) parts.push(`pH estimate: ${item.ph_estimate} (limited reliability)`);
    return parts.join(' · ');
  }

  function setHistoryBusy(value) {
    historyBusy = value;
    $('clear-history').disabled = value;
    for (const button of $('history-list').querySelectorAll('button[data-delete]')) button.disabled = value;
  }

  async function refreshHistory() {
    const ticket = historyGate.begin();
    setStatus('history-status', 'Loading local history…', 'loading');
    $('history-list').classList.add('is-loading');
    $('history-list').replaceChildren();
    try {
      const response = await api('/history?limit=20', { signal: ticket.signal });
      if (!historyGate.current(ticket)) return;
      if (!Array.isArray(response.items)) throw new Error('The server returned an invalid history list.');
      const message = response.items.length
        ? `Showing ${response.items.length} recent records. Server retention limit: ${response.retention_limit}.`
        : 'No saved records. Saving is off by default.';
      setStatus('history-status', message, response.items.length ? 'success' : 'neutral');
      $('history-list').classList.remove('is-loading');
      for (const item of response.items) {
        const row = element(document, 'li');
        row.append(element(document, 'p', historySummary(item)));
        const actions = element(document, 'div', null, 'actions');
        const view = element(document, 'button', 'View record');
        view.type = 'button';
        view.setAttribute('aria-label', `View ${item.task} ${item.mode} record from ${formatTimestamp(item.created_at)}`);
        view.addEventListener('click', () => void viewRecord(item.id));
        const remove = element(document, 'button', 'Delete record');
        remove.type = 'button';
        remove.dataset.delete = item.id;
        remove.disabled = historyBusy;
        remove.setAttribute('aria-label', `Delete ${item.task} ${item.mode} record from ${formatTimestamp(item.created_at)}`);
        remove.addEventListener('click', () => void deleteHistory(item.id));
        actions.append(view, remove);
        row.append(actions);
        $('history-list').append(row);
      }
    } catch (error) {
      if (!historyGate.current(ticket)) return;
      $('history-list').classList.remove('is-loading');
      setStatus('history-status', `History unavailable: ${error.message} You can refresh history to check again.`, 'error');
    }
  }

  async function viewRecord(id) {
    const ticket = recordGate.begin();
    $('history-details').open = true;
    $('recorded-result').hidden = false;
    $('record-content').replaceChildren();
    setStatus('record-status', 'Loading recorded result…', 'loading');
    try {
      const record = await api(`/history/${encodeURIComponent(id)}`, { signal: ticket.signal });
      if (!recordGate.current(ticket)) return;
      setStatus('record-status', `Recorded ${formatTimestamp(record.created_at)} · release ${record.release_id}. Historical output, not a current-model assessment. Saved inputs may include synthetic examples; measurement provenance is not recorded.`, 'warning');
      renderResult(document, $('record-content'), record.result, { task: record.task, mode: record.mode, inputs: record.inputs });
      $('recorded-heading').focus();
    } catch (error) {
      if (recordGate.current(ticket)) setStatus('record-status', `Unable to read record: ${error.message}`, 'error');
    }
  }

  async function deleteHistory(id = null) {
    if (historyBusy) return;
    const accepted = confirm(id === null
      ? 'Permanently delete ALL saved measurements and results from this local server? This cannot be undone.'
      : 'Permanently delete this saved measurement and result from the local server?');
    if (!accepted) return;
    setHistoryBusy(true);
    recordGate.invalidate();
    $('recorded-result').hidden = true;
    $('record-content').replaceChildren();
    setStatus('history-action-status', 'Deleting saved history…', 'loading');
    try {
      const result = await api(id === null ? '/history' : `/history/${encodeURIComponent(id)}`, { method: 'DELETE' });
      setStatus('history-action-status', id === null ? `Deleted ${result.deleted_count} saved records.`
        : result.deleted ? 'Record deleted.' : 'Record was already absent.', 'success');
      await refreshHistory();
    } catch (error) {
      setStatus('history-action-status', `Deletion could not be confirmed: ${error.message} Refresh history to check its state before trying again. No automatic retry was made.`, 'error');
    } finally { setHistoryBusy(false); }
  }

  $('prediction-form').addEventListener('submit', event => void submit(event));
  for (const id of ['task', 'mode']) $(id).addEventListener('change', () => {
    const previous = readEntries();
    invalidate();
    if (source === 'synthetic') source = 'edited';
    sourceNotice();
    buildFields(previous);
  });
  $('save-history').setAttribute('aria-describedby', 'save-help');
  $('save-history').addEventListener('change', () => invalidate());
  $('refresh-metadata').addEventListener('click', () => void refreshMetadata());
  $('reset-form').addEventListener('click', () => {
    invalidate(); source = 'empty'; sourceNotice(); $('save-history').checked = false; saveNotice(); buildFields();
  });
  $('synthetic-example').addEventListener('click', () => {
    invalidate();
    const references = modelFor(metadata, task(), mode())?.reference_ranges;
    for (const [name, control] of controls) {
      control.input.value = String(references[name].median);
      control.missing.checked = false;
      control.input.disabled = false;
      control.input.required = true;
    }
    source = 'synthetic'; sourceNotice();
  });
  $('refresh-history').addEventListener('click', () => void refreshHistory());
  $('clear-history').addEventListener('click', () => void deleteHistory());
  $('close-record').addEventListener('click', () => { recordGate.invalidate(); $('recorded-result').hidden = true; $('record-content').replaceChildren(); $('history-heading').scrollIntoView?.({ block: 'nearest' }); });

  return { start: () => Promise.all([refreshMetadata(), refreshHistory()]), refreshMetadata, refreshHistory, submit, viewRecord, deleteHistory,
    destroy: () => { inferenceGate.invalidate(); metadataGate.invalidate(); historyGate.invalidate(); recordGate.invalidate(); } };
}

function formatTimestamp(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value || 'Unknown time') : date.toISOString();
}

if (typeof document !== 'undefined') {
  const app = createApp(document);
  void app.start();
}
