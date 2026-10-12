import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { renderResult, renderWarnings } from '../render.js';

const classification = {
  assessment: 'predicted_safe', model_label: 'safe', class_id: 1,
  probability_safe: 0.81723456789, decision_threshold: 0.63123456789,
  model_id: 'classification-private-id', release_id: 'recorded-release-id',
  warnings: [{ code: 'not_certification', message: 'Original certification warning.', features: [] }],
};
const regression = {
  estimate: 7.12654321, model_id: 'regression-private-id', reliability: 'limited',
  warnings: [{ code: 'weak_regression', message: 'Original regression warning.', features: [] }],
};
const snapshot = {
  task: 'classification', mode: 'full', inputs: { Hardness: 3 },
  references: { Hardness: { q01: 1, q99: 5 } },
};

function render(response = classification, context = snapshot) {
  const dom = new JSDOM('<main></main>');
  const parent = dom.window.document.querySelector('main');
  renderResult(dom.window.document, parent, response, context);
  return parent;
}

// JSDOM textContent includes collapsed content. Model native details visibility,
// so assertions about the beginner view never mistake hidden text for visible UI.
function visibleText(node) {
  if (node.nodeType === 3) return node.textContent;
  if (node.hidden) return '';
  if (node.tagName === 'DETAILS' && !node.open) return visibleText(node.querySelector(':scope > summary'));
  return [...node.childNodes].map(visibleText).join(' ');
}

test('default view has a short assessment and limitation; raw evidence is in closed technical details', () => {
  const parent = render();
  const visible = visibleText(parent);
  assert.equal(parent.querySelector('.result-summary h3').textContent, 'Model predicts safe');
  assert.match(visible, /Not drinking-water certification · confirm with laboratory testing/);
  assert.ok(visible.split(/\s+/).length < 35);
  assert.doesNotMatch(visible, /private-id|recorded-release-id|probability|threshold|Original|Hardness/);
  assert.equal(parent.querySelector(':scope > details > summary').textContent, 'Technical details');
  for (const details of parent.querySelectorAll('details')) assert.equal(details.open, false);
  const technical = parent.querySelector('.technical-details');
  assert.match(technical.textContent, /classification-private-id/);
  assert.match(technical.textContent, /recorded-release-id/);
  assert.match(technical.textContent, /Raw model label: safe · class 1/);
  assert.match(technical.textContent, /Uncalibrated probability.*0.81723456789/);
  assert.match(technical.textContent, /Decision threshold: 0.63123456789/);
  assert.match(technical.textContent, /Not confidence/);
  assert.ok(technical.querySelector('.meter'));
  assert.ok(technical.querySelector('.reference-row'));
  assert.match(technical.textContent, /training distributions, not safety limits/);
  technical.open = true;
  assert.match(visibleText(parent), /classification-private-id/);
  assert.doesNotMatch(visibleText(parent), /Original certification warning/);
});

test('quick classification and class-one abstention never show a safe headline', () => {
  for (const [response, mode] of [
    [classification, 'quick'],
    [{ ...classification, assessment: 'insufficient_evidence' }, 'full'],
  ]) {
    const parent = render(response, { ...snapshot, mode });
    assert.equal(parent.querySelector('.result-summary h3').textContent, 'More readings needed');
    assert.ok(parent.querySelector('.result-summary.caution'));
    assert.doesNotMatch(visibleText(parent), /predicts safe|Raw model label/);
    assert.equal(parent.querySelectorAll('.result-summary p').length, 1);
  }
  const concern = render({ ...classification, assessment: 'predicted_unsafe', class_id: 0 });
  assert.equal(concern.querySelector('.result-summary h3').textContent, 'Model flags a concern');
  assert.ok(concern.querySelector('.result-summary.concern'));
});

test('pH-only and combined results display the rounded estimate and caveat once', () => {
  for (const mode of ['full', 'quick']) {
    for (const task of ['regression', 'combined']) {
      const parent = render(task === 'combined' ? { classification, regression } : regression, { ...snapshot, mode, task });
      const visible = visibleText(parent);
      assert.equal((visible.match(/7\.13/g) || []).length, 1);
      assert.equal((visible.match(/Estimate only · limited reliability/g) || []).length, 1);
      assert.doesNotMatch(visible, /7\.12654321|regression-private-id/);
      if (task === 'regression') {
        assert.equal(parent.querySelector('.result-summary h3').textContent, 'Estimated pH');
        assert.doesNotMatch(visible, /More readings needed|safety assessment/);
      }
    }
  }
});

test('essential risks and affected fields remain visible while every original warning is retained', () => {
  const warnings = [
    { code: 'missing_measurements', message: 'original missing', features: ['Hardness'] },
    { code: 'outside_training_range', message: 'original outside', features: ['TDS'] },
    { code: 'unknown_training_range', message: 'original unknown', features: ['Turbidity'] },
    { code: 'near_threshold', message: 'original threshold', features: [] },
    { code: 'model_runtime_warning', message: 'original runtime', features: [] },
  ];
  const parent = render({ ...classification, assessment: 'insufficient_evidence', warnings });
  const visible = visibleText(parent);
  assert.match(visible, /Missing readings.*Hardness/);
  assert.match(visible, /Outside the training range.*TDS/);
  assert.match(visible, /Training range unavailable.*Turbidity/);
  assert.match(visible, /borderline/);
  assert.match(visible, /Model runtime warning/);
  assert.equal((visible.match(/Missing readings/g) || []).length, 1);
  assert.doesNotMatch(visible, /original/);
  for (const warning of warnings) assert.ok(parent.querySelector('.technical-details').textContent.includes(warning.message));
});

test('combined risk summaries deduplicate fields but retain all source warnings', () => {
  const warning = { code: 'missing_measurements', message: 'Repeated source warning.', features: ['TDS'] };
  const parent = render({
    classification: { ...classification, warnings: [warning] },
    regression: { ...regression, warnings: [warning] },
    warnings: [{ code: 'unknown_range', message: 'Top-level warning.', features: ['Hardness'] }],
  }, { ...snapshot, task: 'combined' });
  assert.equal((visibleText(parent).match(/Missing readings/g) || []).length, 1);
  assert.equal((parent.textContent.match(/Repeated source warning/g) || []).length, 2);
  assert.match(visibleText(parent), /Training range unavailable.*Hardness/);
  assert.match(parent.querySelector('.technical-details').textContent, /Top-level warning/);
});

test('provenance is concise and records without current metadata use the same presentation', () => {
  assert.match(visibleText(render(classification, { ...snapshot, synthetic: true })), /Synthetic example · not a real sample/);
  assert.match(visibleText(render(classification, { ...snapshot, editedSynthetic: true })), /Edited example · measurements unverified/);
  const historical = render(classification, { task: 'classification', mode: 'full', inputs: snapshot.inputs });
  assert.equal(historical.querySelector('.result-summary h3').textContent, 'Model predicts safe');
  assert.doesNotMatch(visibleText(historical), /recorded-release-id/);
  assert.match(historical.querySelector('.technical-details').textContent, /recorded-release-id/);
});

test('all untrusted labels, messages, fields and IDs are text-safe even after details expand', () => {
  const attack = '<img src=x onerror=bad()><script>bad()</script>';
  const parent = render({ ...classification, model_label: attack, model_id: attack, release_id: attack,
    warnings: [{ code: 'missing_measurements', message: attack, features: [attack] }],
  }, { ...snapshot, inputs: { [attack]: attack } });
  for (const details of parent.querySelectorAll('details')) details.open = true;
  assert.equal(parent.querySelector('img, script'), null);
  assert.ok(visibleText(parent).includes(attack));
  assert.ok(parent.querySelector('.warnings').textContent.includes(attack));
});

test('standalone warning rendering is optional by default and preserves original text', () => {
  const parent = new JSDOM('<main></main>').window.document.querySelector('main');
  renderWarnings(parent.ownerDocument, parent, classification.warnings);
  assert.equal(parent.querySelector('details').open, false);
  assert.doesNotMatch(visibleText(parent), /Original certification warning/);
  parent.querySelector('details').open = true;
  assert.match(visibleText(parent), /Original certification warning/);
});
