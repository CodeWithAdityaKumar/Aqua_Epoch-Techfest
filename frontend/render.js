import { resultSummary, warningSeverity } from './core.js';

export function element(document, tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function number(value) { return Number.isFinite(value) ? String(Number(value.toPrecision(6))) : 'unavailable'; }

function meter(document, { value, max = 1, threshold, label, position = false }) {
  const figure = element(document, 'figure');
  figure.style.margin = '1rem 0';
  figure.append(element(document, 'figcaption', label));
  const track = element(document, 'div', null, 'meter');
  track.setAttribute('aria-hidden', 'true');
  const percentage = Math.max(0, Math.min(100, value / max * 100));
  if (Number.isFinite(value)) {
    const mark = element(document, 'div', null, position ? 'meter-marker value' : 'meter-fill');
    mark.style[position ? 'left' : 'width'] = `${percentage}%`;
    track.append(mark);
  }
  if (Number.isFinite(threshold)) {
    const marker = element(document, 'div', null, 'meter-marker');
    marker.style.left = `${Math.max(0, Math.min(100, threshold / max * 100))}%`;
    track.append(marker);
  }
  const axis = element(document, 'div', null, 'meter-axis');
  axis.setAttribute('aria-hidden', 'true');
  axis.append(element(document, 'span', '0'), element(document, 'span', String(max)));
  figure.append(track, axis);
  return figure;
}

export function renderWarnings(document, parent, warnings, label = 'Original warnings') {
  const details = element(document, 'details');
  details.append(element(document, 'summary', label));
  const list = element(document, 'ul', null, 'warnings');
  for (const warning of warnings || []) {
    const severity = warningSeverity(warning);
    const item = element(document, 'li', null, `warning-${severity}`);
    item.append(element(document, 'strong', `${warning.code || 'warning'} · ${severity}`), document.createTextNode(` — ${warning.message || 'No message supplied.'}`));
    item.append(element(document, 'p', `Features: ${Array.isArray(warning.features) && warning.features.length ? warning.features.join(', ') : 'none specified'}`, 'meta'));
    list.append(item);
  }
  details.append(list.childNodes.length ? list : element(document, 'p', 'No server warnings supplied.'));
  parent.append(details);
}

function renderClassification(document, parent, classification) {
  const block = element(document, 'section', null, 'result-block');
  block.append(element(document, 'h3', 'Dataset-label classification'));
  block.append(element(document, 'p', `Server assessment: ${classification.assessment || 'unavailable'}`, 'meta'));
  block.append(element(document, 'p', `Raw model label: ${classification.model_label ?? 'unavailable'} · class ${classification.class_id ?? 'unavailable'}`));
  block.append(meter(document, {
    value: classification.probability_safe,
    threshold: classification.decision_threshold,
    label: `Uncalibrated probability of dataset class 1 (Potability=1): ${classification.probability_safe ?? 'unavailable'}. Decision threshold: ${classification.decision_threshold ?? 'unavailable'} (marker). Not confidence or a measured probability of water safety.`,
  }));
  block.append(element(document, 'p', `Model: ${classification.model_id ?? 'unavailable'}`, 'meta'));
  renderWarnings(document, block, classification.warnings, 'Classification warnings');
  parent.append(block);
}

function renderRegression(document, parent, regression, standalone) {
  const block = element(document, 'section', null, 'result-block');
  const estimate = Number.isFinite(regression?.estimate) ? regression.estimate.toFixed(2) : 'unavailable';
  block.append(element(document, 'p', `${standalone ? '' : 'Estimated pH: '}${estimate}`, 'assessment caution'));
  block.append(element(document, 'p', 'Estimate only · limited reliability', 'meta'));
  parent.append(block);
}

function renderInputs(document, parent, snapshot) {
  const details = element(document, 'details');
  details.append(element(document, 'summary', 'Submitted measurement snapshot'));
  const list = element(document, 'dl');
  for (const [name, value] of Object.entries(snapshot.inputs || {})) {
    list.append(element(document, 'dt', name), element(document, 'dd', value === null ? 'Not measured (null)' : `${value} · units unknown`));
  }
  details.append(list);
  parent.append(details);
  if (!snapshot.references) return;
  const references = element(document, 'details');
  references.append(element(document, 'summary', 'Inputs relative to training q01–q99'));
  references.append(element(document, 'p', 'Each row is normalized to its own training q01–q99 range. These are training distributions, not safety limits. Values outside a range are shown at its edge with their original value; units remain undocumented.'));
  let count = 0;
  for (const [name, value] of Object.entries(snapshot.inputs || {})) {
    const range = snapshot.references[name];
    if (value === null || !Number.isFinite(value) || !Number.isFinite(range?.q01) || !Number.isFinite(range?.q99)) continue;
    const row = element(document, 'div', null, 'reference-row');
    const note = value < range.q01 ? 'below q01' : value > range.q99 ? 'above q99' : 'within q01–q99';
    row.append(element(document, 'p', `${name}: ${number(value)}; training q01 ${number(range.q01)}, q99 ${number(range.q99)} · ${note} · units unknown`));
    if (range.q99 > range.q01) {
      const chart = meter(document, { value: (value - range.q01) / (range.q99 - range.q01), position: true,
        label: `${name}: normalized position within its own training range (0 = q01, 1 = q99).` });
      row.append(chart);
    } else row.append(element(document, 'p', 'Training quantiles coincide; no normalized marker is shown.'));
    references.append(row);
    count += 1;
  }
  if (!count) references.append(element(document, 'p', 'No training reference chart is available for these readings.'));
  parent.append(references);
}

function renderSummary(document, parent, summary) {
  const box = element(document, 'section', null, `result-summary ${summary.style}`);
  box.setAttribute('aria-label', 'Result summary');
  box.append(element(document, 'h3', summary.title));
  if (summary.details.length) {
    for (const detail of summary.details) box.append(element(document, 'p', detail));
  }
  parent.append(box);
}

export function renderResult(document, parent, response, snapshot) {
  parent.replaceChildren();
  const summary = resultSummary(response, snapshot);
  renderSummary(document, parent, summary);
  if (snapshot.synthetic) parent.append(element(document, 'p', 'Synthetic example · not a real sample.', 'notice'));
  else if (snapshot.editedSynthetic) parent.append(element(document, 'p', 'Edited example · measurements unverified.', 'notice'));
  if (snapshot.task === 'combined' || snapshot.task === 'regression') {
    renderRegression(document, parent, snapshot.task === 'combined' ? response.regression : response, snapshot.task === 'regression');
  }
  if (summary.risks.length) {
    const risks = element(document, 'ul', null, 'warnings input-risks');
    for (const risk of summary.risks) risks.append(element(document, 'li', risk.text, `warning-${risk.style}`));
    parent.append(risks);
  }
  parent.append(element(document, 'p', 'Not drinking-water certification · confirm with laboratory testing.', 'notice result-limitations'));

  const technical = element(document, 'details', null, 'technical-details');
  technical.append(element(document, 'summary', 'Technical details'));
  technical.append(element(document, 'p', `${snapshot.task} · ${snapshot.mode} mode · recorded response release ${response.release_id ?? 'unavailable'}`, 'meta'));
  if (snapshot.task === 'combined' || snapshot.task === 'classification') {
    const classification = snapshot.task === 'combined' ? response.classification : response;
    if (classification) renderClassification(document, technical, classification);
    else technical.append(element(document, 'p', 'Classification result unavailable.'));
  }
  if (snapshot.task === 'combined' || snapshot.task === 'regression') {
    const regression = snapshot.task === 'combined' ? response.regression : response;
    if (regression) {
      const block = element(document, 'section', null, 'result-block');
      block.append(element(document, 'h3', 'pH model details'));
      block.append(element(document, 'p', `Model: ${regression.model_id ?? 'unavailable'} · raw estimate: ${regression.estimate ?? 'unavailable'}`, 'meta'));
      block.append(element(document, 'p', 'Weak predictive evidence; not a laboratory measurement. No calibrated uncertainty interval is available.'));
      renderWarnings(document, block, regression.warnings, 'pH warnings');
      technical.append(block);
    }
  }
  if (response.warnings && snapshot.task === 'combined') renderWarnings(document, technical, response.warnings, 'Overall warnings');
  renderInputs(document, technical, snapshot);
  parent.append(technical);
}
