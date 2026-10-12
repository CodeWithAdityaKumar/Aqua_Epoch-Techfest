/** Pure contract helpers. No feature names or model input lists are duplicated here. */
export function modelFor(metadata, task, mode) {
  return metadata?.models?.[`${task === 'combined' ? 'classification' : task}_${mode}`];
}

export function featuresFor(metadata, task, mode) {
  const fields = modelFor(metadata, task, mode)?.features;
  if (!Array.isArray(fields) || !fields.length) throw new Error('Model feature schema is unavailable. Refresh metadata.');
  const names = new Set();
  for (const field of fields) {
    if (typeof field.name !== 'string' || !field.name || names.has(field.name)
      || !Array.isArray(field.engineering_bounds) || field.engineering_bounds.length !== 2
      || !field.engineering_bounds.every(Number.isFinite)
      || field.engineering_bounds[0] > field.engineering_bounds[1]) {
      throw new Error('Model feature schema is invalid. Refresh metadata.');
    }
    names.add(field.name);
  }
  return fields;
}

export function availability(metadata, task, mode) {
  const tasks = task === 'combined' ? ['classification', 'regression'] : [task];
  const entries = tasks.map(name => metadata?.models?.[`${name}_${mode}`]);
  const unavailable = entries.flatMap((entry, index) => entry?.available && entry.release_id ? []
    : [`${tasks[index]} ${mode}: ${entry?.reason || 'model unavailable'}`]);
  if (unavailable.length) return { ready: false, message: unavailable.join('; ') };
  if (task === 'combined' && entries[0].release_id !== entries[1].release_id) {
    return { ready: false, message: 'Combined inference is unavailable: model release IDs do not match.' };
  }
  try {
    featuresFor(metadata, task, mode);
    if (task === 'combined') featuresFor(metadata, 'regression', mode);
  } catch (error) { return { ready: false, message: error.message }; }
  return { ready: true, message: `Demo model${tasks.length > 1 ? 's' : ''} available · release ${entries[0].release_id}` };
}

export class ValidationError extends Error {
  constructor(errors) {
    super('Check the measurements below.');
    this.name = 'ValidationError';
    this.errors = errors;
  }
}

/** entries: exact field-name map of {value: string, missing: boolean}. */
export function serializeFields(features, entries, regressionFeatures = null) {
  const errors = {};
  const pairs = [];
  const allowed = new Set(features.map(field => field.name));
  for (const name of Object.keys(entries)) {
    if (!allowed.has(name)) errors[name] = 'Unknown field: not part of this model schema.';
  }
  for (const field of features) {
    const entry = Object.hasOwn(entries, field.name) ? entries[field.name] : null;
    if (!entry) { errors[field.name] = 'Required key is missing.'; continue; }
    if (entry.missing === true) {
      if (field.nullable !== true) errors[field.name] = 'This field does not allow Not measured.';
      else pairs.push([field.name, null]);
      continue;
    }
    const raw = entry.value;
    if (typeof raw !== 'string' || raw.trim() === '') {
      errors[field.name] = 'Enter a number or select Not measured.';
      continue;
    }
    // Decimal/scientific numeric input only; do not coerce whitespace, hex or booleans.
    const value = Number(raw);
    if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(raw.trim()) || !Number.isFinite(value)) {
      errors[field.name] = 'Enter a finite numeric measurement.';
    } else if (value < field.engineering_bounds[0] || value > field.engineering_bounds[1]) {
      errors[field.name] = `Use the engineering range ${field.engineering_bounds[0]}–${field.engineering_bounds[1]} (not safety limits).`;
    } else pairs.push([field.name, value]);
  }
  const values = Object.fromEntries(pairs);
  if (!Object.keys(errors).length) {
    if (pairs.every(([, value]) => value === null)) errors._form = 'At least one measured value is required; all-null inputs are not accepted.';
    else if (regressionFeatures && !regressionFeatures.some(field => Object.hasOwn(values, field.name) && values[field.name] !== null)) {
      errors._form = 'Combined inference needs at least one measured regression input; measured pH alone cannot estimate pH.';
    }
  }
  if (Object.keys(errors).length) throw new ValidationError(errors);
  return values;
}

export function inferencePath(task, mode, saveHistory = false) {
  const route = { combined: 'predict', classification: 'classify', regression: 'ph' }[task];
  if (!route || !['full', 'quick'].includes(mode)) throw new Error('Unknown task or mode.');
  return `/${route}/${mode}${saveHistory ? '?save_history=true' : ''}`;
}

export function assessmentPresentation(result, mode) {
  if (mode === 'quick' || result.assessment === 'insufficient_evidence') {
    return { text: 'More readings needed', style: 'caution' };
  }
  if (result.assessment === 'predicted_safe') return { text: 'Model predicts safe', style: 'caution' };
  if (result.assessment === 'predicted_unsafe') return { text: 'Model flags a concern', style: 'concern' };
  return { text: 'Assessment unavailable', style: 'caution' };
}

/** Compact, non-certifying summary for the top of a result. */
export function resultSummary(response, snapshot) {
  const classification = snapshot.task === 'combined' ? response?.classification
    : snapshot.task === 'classification' ? response : null;
  const regression = snapshot.task === 'combined' ? response?.regression
    : snapshot.task === 'regression' ? response : null;
  const assessment = classification ? assessmentPresentation(classification, snapshot.mode) : null;
  const allWarnings = [...(response?.warnings || []),
    ...(snapshot.task === 'combined' ? classification?.warnings || [] : []),
    ...(snapshot.task === 'combined' ? regression?.warnings || [] : [])];
  // Keep short, actionable input risks visible; original messages remain in details.
  const risks = [
    { codes: ['missing', 'missing_measurements'], text: 'Missing readings — measure the unrecorded inputs', style: 'caution' },
    { codes: ['outside_training_range'], text: 'Outside the training range — check these readings', style: 'concern' },
    { codes: ['unknown_range', 'unknown_training_range'], text: 'Training range unavailable — verify these readings independently', style: 'caution' },
    { codes: ['near_threshold'], text: 'Model result is borderline — confirm with laboratory testing', style: 'caution' },
    { codes: ['model_runtime_warning'], text: 'Model runtime warning — have the model setup checked', style: 'caution' },
    { codes: ['unusual_measurements'], text: 'Unusual readings for this model — check the measurements', style: 'caution' },
  ].flatMap(risk => {
    const matches = allWarnings.filter(warning => risk.codes.includes(warning.code));
    if (!matches.length) return [];
    const fields = [...new Set(matches.flatMap(warning => Array.isArray(warning.features) ? warning.features : []))];
    return [{ text: `${risk.text}${fields.length ? ` (${fields.join(', ')})` : ''}.`, style: risk.style }];
  });
  const details = [];
  if (assessment?.text === 'More readings needed') {
    if (snapshot.mode === 'quick') details.push('Quick mode uses too few readings; add a full set of measurements.');
    else details.push(risks.shift()?.text || 'The evidence is incomplete; collect a full set of readings and confirm with laboratory testing.');
  }

  return {
    title: assessment?.text || (snapshot.task === 'regression' ? 'Estimated pH' : 'Assessment unavailable'),
    style: assessment?.style || 'caution',
    details,
    risks,
  };
}

export function warningSeverity(warning = {}) {
  const value = `${warning.severity || ''} ${warning.code || ''}`.toLowerCase();
  if (/(critical|error|unsafe|out.of.range|outside_training_range)/.test(value)) return 'concern';
  if (/(caution|abstain|limited|quick|threshold|uncalibrated|missing|insufficient|unknown.*range|model_runtime_warning|unusual_measurements)/.test(value)) return 'caution';
  return 'info';
}

export class ApiError extends Error {
  constructor(status, payload) {
    const detail = payload?.detail;
    const object = detail && !Array.isArray(detail) && typeof detail === 'object' ? detail : payload;
    const message = object?.message || (typeof detail === 'string' ? detail : '')
      || (status === 503 ? 'The requested model or service is unavailable.' : status === 422 ? 'The server rejected these inputs.' : 'The server could not complete the request.');
    super(`HTTP ${status}: ${message}${object?.reason ? ` (${object.reason})` : ''}`);
    this.name = 'ApiError';
    this.status = status;
    this.details = Array.isArray(payload?.details) ? payload.details : Array.isArray(detail) ? detail : [];
  }
}

export async function requestJSON(path, { fetchImpl = globalThis.fetch, timeoutMs = 20000, signal, ...options } = {}) {
  const controller = new AbortController();
  let timer;
  let cancel;
  const interrupted = new Promise((_, reject) => {
    cancel = () => { controller.abort(); reject(new DOMException('Request interrupted', 'AbortError')); };
    signal?.addEventListener('abort', cancel, { once: true });
    if (signal?.aborted) cancel();
    timer = setTimeout(() => {
      controller.abort();
      reject(new Error('Request timed out. The server may still have processed it.'));
    }, timeoutMs);
  });
  const perform = async () => {
    const response = await fetchImpl(path, { ...options, signal: controller.signal, headers: { Accept: 'application/json', ...options.headers } });
    let data;
    try { data = await response.json(); } catch {
      if (!response.ok) throw new ApiError(response.status, null);
      throw new Error('The server returned an unreadable response.');
    }
    if (!response.ok) throw new ApiError(response.status, data);
    return data;
  };
  try { return await Promise.race([perform(), interrupted]); }
  finally { clearTimeout(timer); signal?.removeEventListener('abort', cancel); }
}

/** Abort and revision checks both matter: some transports finish after cancellation. */
export class RequestGate {
  constructor() { this.revision = 0; this.controller = null; }
  invalidate() { this.revision += 1; this.controller?.abort(); this.controller = null; }
  begin() { this.invalidate(); this.controller = new AbortController(); return { revision: this.revision, signal: this.controller.signal }; }
  current(ticket) { return ticket.revision === this.revision && !ticket.signal.aborted; }
}
