import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { createApp } from '../app.js';

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const field = (name, max = 1000) => ({ name, required: true, nullable: true, engineering_bounds: [0, max] });
const quick = [field('Turbidity'), field('Hardness'), field('TDS', 100000)];
const full = [field('pH', 14), ...quick, field('Conductivity')];
const model = features => ({
  available: true, release_id: 'test-release', features,
  reference_ranges: Object.fromEntries(features.map(item => [item.name, { median: 5, q01: 1, q99: 10 }])),
});
const metadata = { models: {
  classification_quick: model(quick), regression_quick: model(quick),
  classification_full: model(full), regression_full: model(full.filter(item => item.name !== 'pH')),
} };
const result = {
  release_id: 'test-release',
  classification: { assessment: 'insufficient_evidence', model_label: 'safe', class_id: 1, probability_safe: .7, decision_threshold: .5, warnings: [] },
  regression: { estimate: 7.1, warnings: [] }, warnings: [],
};
const response = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };

function fixture(t, { intercept = () => undefined, models = metadata, confirm = () => false } = {}) {
  const dom = new JSDOM(html, { url: 'http://localhost/' });
  const document = dom.window.document;
  const calls = [];
  const fetchImpl = async (path, options) => {
    const call = { path, ...options };
    calls.push(call);
    const custom = intercept(call);
    if (custom !== undefined) return custom;
    if (path === '/models') return response(models);
    if (path === '/health') return response({ ready: true });
    if (path === '/history?limit=20') return response({ items: [], retention_limit: 20 });
    if (options.method === 'POST') return response(result);
    throw new Error(`Unexpected request: ${path}`);
  };
  const app = createApp(document, { fetchImpl, confirm });
  t.after(() => { app.destroy(); dom.window.close(); });
  const $ = id => document.getElementById(id);
  const input = name => document.querySelector(`input[name="${name}"]`);
  const change = (id, value) => {
    $(id).value = value;
    $(id).dispatchEvent(new dom.window.Event('change', { bubbles: true }));
  };
  const enter = (name, value) => {
    input(name).value = value;
    input(name).dispatchEvent(new dom.window.Event('input', { bubbles: true }));
  };
  const missing = name => {
    const checkbox = document.getElementById(`${input(name).id}-missing`);
    checkbox.checked = true;
    checkbox.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
  };
  return { app, document, $, input, change, enter, missing, calls, dom };
}

test('starts with three blank schema fields, concise labels and closed optional details', async t => {
  const f = fixture(t);
  await f.app.start();
  assert.equal(f.$('task').value, 'combined');
  assert.equal(f.$('mode').value, 'quick');
  assert.deepEqual([...f.document.querySelectorAll('#fields input[type="number"]')].map(node => node.name), quick.map(item => item.name));
  for (const name of quick.map(item => item.name)) assert.equal(f.input(name).value, '');
  assert.equal(f.input('pH'), null);
  assert.equal(f.$('save-history').checked, false);
  assert.equal(f.$('history-details').open, false);
  assert.equal(f.$('service-details').open, false);
  assert.equal(f.$('availability').textContent, 'Ready');
  assert.equal(f.$('submit-prediction').textContent, 'Get result');
  assert.match(f.$('mode-help').textContent, /Quick check cannot confirm drinking safety/);
  assert.match(f.document.querySelector('label[for="measurement-0"]').textContent, /Turbidity · cloudiness/);
  assert.match(f.document.querySelector('label[for="measurement-2"]').textContent, /total dissolved solids/);
  assert.ok([...f.document.querySelectorAll('.field-help')].every(node => !node.open));
  assert.match(f.document.querySelector('.field-help').textContent, /Engineering range.*not safe drinking-water limits.*API key/);
});

test('shared readings and Not measured survive task/mode switches; new fields stay blank', async t => {
  const f = fixture(t);
  await f.app.start();
  f.enter('TDS', '245');
  f.enter('Turbidity', '0');
  f.missing('Hardness');
  f.change('mode', 'full');
  assert.equal(f.input('TDS').value, '245');
  assert.equal(f.input('Turbidity').value, '0');
  assert.equal(f.input('Hardness').disabled, true);
  assert.equal(f.input('pH').value, '');
  assert.equal(f.input('pH').required, false);
  assert.equal(f.input('Conductivity').value, '');
  assert.equal(f.document.querySelectorAll('#fields input[type="number"]').length, full.length);
  f.enter('pH', '7');
  f.change('task', 'regression');
  assert.equal(f.input('pH'), null);
  assert.equal(f.input('TDS').value, '245');
  assert.match(f.$('mode-help').textContent, /pH.*limited reliability/);
  assert.doesNotMatch(f.$('mode-help').textContent, /classification|water check|safety/i);
  f.change('mode', 'quick');
  assert.equal(f.input('TDS').value, '245');
  assert.equal(f.input('Hardness').disabled, true);
  assert.equal(f.input('pH'), null);
  f.change('task', 'classification');
  assert.equal(f.input('TDS').value, '245');
});

test('blank and range errors stay local; explicit null and zero serialize without imputation', async t => {
  const f = fixture(t);
  await f.app.start();
  await f.app.submit();
  assert.equal(f.$('form-errors').hidden, false);
  assert.match(f.$('form-errors').textContent, /Enter a number or select Not measured/);
  assert.equal(f.document.querySelectorAll('[aria-invalid="true"]').length, 3);
  assert.equal(f.calls.filter(call => call.method === 'POST').length, 0);
  f.enter('TDS', '-1');
  f.enter('Turbidity', '0');
  f.missing('Hardness');
  await f.app.submit();
  assert.match(f.$('form-errors').textContent, /engineering range/);
  assert.equal(f.calls.filter(call => call.method === 'POST').length, 0);
  f.enter('TDS', '123');
  await f.app.submit();
  const posted = f.calls.find(call => call.method === 'POST');
  assert.equal(posted.path, '/predict/quick');
  assert.deepEqual(JSON.parse(posted.body), { Turbidity: 0, Hardness: null, TDS: 123 });
  assert.equal(f.$('form-errors').hidden, true);
  assert.equal(f.$('live-results').hidden, false);
});

test('all-null and pH-only combined inputs cannot be submitted', async t => {
  const f = fixture(t);
  await f.app.start();
  quick.forEach(item => f.missing(item.name));
  await f.app.submit();
  assert.match(f.$('form-errors').textContent, /all-null/);
  f.change('mode', 'full');
  f.enter('pH', '7');
  f.missing('Conductivity');
  await f.app.submit();
  assert.match(f.$('form-errors').textContent, /measured pH alone/);
  assert.equal(f.calls.filter(call => call.method === 'POST').length, 0);
});

test('combined full flow estimates pH when it is left blank', async t => {
  const f = fixture(t);
  await f.app.start();
  f.change('mode', 'full');
  f.enter('TDS', '123');
  f.missing('Turbidity');
  f.missing('Hardness');
  f.missing('Conductivity');
  await f.app.submit();
  const posted = f.calls.find(call => call.method === 'POST');
  assert.equal(posted.path, '/predict/full');
  assert.equal(JSON.parse(posted.body).pH, null);
  assert.equal(f.$('form-errors').hidden, true);
  assert.equal(f.$('live-results').hidden, false);
});

test('examples are labelled, edits retain provenance, and Clear resets readings and opt-in', async t => {
  const f = fixture(t);
  await f.app.start();
  f.$('synthetic-example').click();
  assert.equal(f.$('input-source').textContent, 'Example values — not a real sample');
  assert.equal(f.input('TDS').value, '5');
  f.enter('TDS', '10');
  assert.equal(f.$('input-source').textContent, 'Edited example values — not a real sample');
  f.$('save-history').click();
  assert.equal(f.$('save-history').checked, true);
  await f.app.submit();
  assert.equal(f.calls.find(call => call.method === 'POST').path, '/predict/quick?save_history=true');
  f.missing('Hardness');
  f.$('reset-form').click();
  assert.equal(f.$('save-history').checked, false);
  assert.equal(f.$('input-source').textContent, '');
  assert.equal(f.$('live-results').hidden, true);
  for (const name of quick.map(item => item.name)) {
    assert.equal(f.input(name).value, '');
    assert.equal(f.input(name).disabled, false);
  }
});

test('edits abort in-flight requests and late responses cannot replace the current form', async t => {
  const pending = deferred();
  const f = fixture(t, { intercept: call => call.method === 'POST' ? pending.promise : undefined });
  await f.app.start();
  f.$('synthetic-example').click();
  const submission = f.app.submit();
  const posted = f.calls.find(call => call.method === 'POST');
  assert.equal(f.$('submit-prediction').disabled, true);
  f.enter('TDS', '55');
  assert.equal(posted.signal.aborted, true);
  pending.resolve(response(result));
  await submission;
  assert.equal(f.$('live-results').hidden, true);
  assert.equal(f.$('result-content').textContent, '');
  assert.equal(f.$('form-errors').hidden, true);
  assert.equal(f.$('submit-prediction').disabled, false);
  assert.equal(f.input('TDS').value, '55');
});

test('mode changes cancel saving requests and keep the unknown-save warning visible', async t => {
  const pending = deferred();
  const f = fixture(t, { intercept: call => call.method === 'POST' ? pending.promise : undefined });
  await f.app.start();
  f.$('synthetic-example').click();
  f.$('save-history').click();
  const submission = f.app.submit();
  f.change('mode', 'full');
  pending.resolve(response(result));
  await submission;
  assert.equal(f.calls.find(call => call.method === 'POST').signal.aborted, true);
  assert.equal(f.$('live-results').hidden, true);
  assert.match(f.$('save-status').textContent, /outcome is unknown.*Check local history/);
});

test('metadata completion preserves edits made while refresh is pending', async t => {
  const pending = deferred();
  let refresh = false;
  const f = fixture(t, { intercept: call => refresh && call.path === '/models' ? pending.promise : undefined });
  await f.app.start();
  f.enter('TDS', '10');
  refresh = true;
  const updating = f.app.refreshMetadata();
  assert.equal(f.$('availability').textContent, 'Connecting…');
  f.enter('TDS', '999');
  f.change('mode', 'full');
  f.enter('pH', '6.8');
  f.missing('Hardness');
  pending.resolve(response(metadata));
  await updating;
  assert.equal(f.input('TDS').value, '999');
  assert.equal(f.input('pH').value, '6.8');
  assert.equal(f.input('Hardness').disabled, true);
  assert.equal(f.$('availability').textContent, 'Ready');
});

test('server validation is visible beside fields and service failure is visible outside details', async t => {
  const f = fixture(t, { intercept: call => call.method === 'POST' ? response({ detail: [{ loc: ['body', 'TDS'], msg: 'Invalid reading from server' }] }, 422) : undefined });
  await f.app.start();
  f.$('synthetic-example').click();
  await f.app.submit();
  assert.equal(f.$('form-errors').hidden, false);
  assert.equal(f.input('TDS').getAttribute('aria-invalid'), 'true');
  assert.match(f.$('form-errors').textContent, /Invalid reading from server/);
  assert.equal(f.$('live-results').hidden, true);
  assert.equal(f.$('submit-prediction').disabled, false);

  const unavailable = structuredClone(metadata);
  unavailable.models.regression_quick.available = false;
  unavailable.models.regression_quick.reason = 'Missing artifact';
  const g = fixture(t, { models: unavailable });
  await g.app.start();
  assert.match(g.$('availability').textContent, /Unavailable.*Missing artifact/);
  assert.equal(g.$('availability').closest('details'), null);
  assert.equal(g.$('submit-prediction').disabled, true);
  g.change('task', 'classification');
  assert.equal(g.$('availability').textContent, 'Ready');
  assert.equal(g.$('submit-prediction').disabled, false);
});

test('network failure shows an actionable error without retrying or erasing input', async t => {
  const f = fixture(t, { intercept: call => call.method === 'POST' ? Promise.reject(new Error('Network unavailable')) : undefined });
  await f.app.start();
  f.$('synthetic-example').click();
  await f.app.submit();
  assert.match(f.$('form-errors').textContent, /Network unavailable/);
  assert.equal(f.$('live-results').hidden, true);
  assert.equal(f.input('TDS').value, '5');
  assert.equal(f.calls.filter(call => call.method === 'POST').length, 1);
});

test('history loading, empty and error messages are distinct and opening a record reveals history', async t => {
  const pending = deferred();
  let state = 'loading';
  const f = fixture(t, { intercept: call => {
    if (call.path === '/history?limit=20' && state === 'loading') return pending.promise;
    if (call.path === '/history?limit=20' && state === 'error') return response({ detail: 'History offline' }, 503);
    if (call.path === '/history/record-1') return response({ created_at: '2026-10-13T12:00:00Z', task: 'combined', mode: 'quick', inputs: { TDS: 5 }, result });
    return undefined;
  } });
  const starting = f.app.start();
  assert.match(f.$('history-status').textContent, /Loading/);
  assert.doesNotMatch(f.$('history-status').textContent, /No saved/);
  pending.resolve(response({ items: [] }));
  await starting;
  assert.match(f.$('history-status').textContent, /No saved records/);
  state = 'error';
  await f.app.refreshHistory();
  assert.match(f.$('history-status').textContent, /History unavailable/);
  assert.doesNotMatch(f.$('history-status').textContent, /No saved/);
  await f.app.viewRecord('record-1');
  assert.equal(f.$('history-details').open, true);
  assert.equal(f.$('recorded-result').hidden, false);
  assert.equal(f.$('live-results').hidden, true);
  f.$('close-record').click();
  assert.equal(f.$('recorded-result').hidden, true);
});

test('history deletion still requires confirmation for one record and all records', async t => {
  const prompts = [];
  const f = fixture(t, { confirm: message => { prompts.push(message); return false; } });
  await f.app.start();
  await f.app.deleteHistory('record-1');
  await f.app.deleteHistory();
  assert.equal(prompts.length, 2);
  assert.match(prompts[0], /Permanently delete this/);
  assert.match(prompts[1], /Permanently delete ALL/);
  assert.equal(f.calls.filter(call => call.method === 'DELETE').length, 0);
});
