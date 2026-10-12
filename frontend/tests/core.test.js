import test from 'node:test';
import assert from 'node:assert/strict';
import { assessmentPresentation, featuresFor, inferencePath, resultSummary, serializeFields, ValidationError, warningSeverity } from '../core.js';

const metadata = {
  models: {
    classification_full: { available: true, release_id: 'r', features: [
      { name: 'pH', required: true, nullable: true, engineering_bounds: [0, 14] },
      { name: 'TDS', required: true, nullable: true, engineering_bounds: [0, 100000] },
    ] },
    regression_full: { available: true, release_id: 'r', features: [
      { name: 'TDS', required: true, nullable: true, engineering_bounds: [0, 100000] },
    ] },
  },
};

test('schema-driven field serialization preserves exact keys and null', () => {
  const fields = featuresFor(metadata, 'combined', 'full');
  assert.deepEqual(serializeFields(fields, {
    pH: { value: '', missing: true }, TDS: { value: '200', missing: false },
  }, featuresFor(metadata, 'regression', 'full')), { pH: null, TDS: 200 });
});

test('blank, malformed, out-of-range and pH-only values are rejected', () => {
  const fields = featuresFor(metadata, 'combined', 'full');
  for (const entries of [
    { pH: { value: '', missing: false }, TDS: { value: '2', missing: false } },
    { pH: { value: 'NaN', missing: false }, TDS: { value: '2', missing: false } },
    { pH: { value: '7', missing: false }, TDS: { value: '', missing: false } },
  ]) assert.throws(() => serializeFields(fields, entries, fields.slice(1)), ValidationError);
  assert.throws(() => serializeFields(fields, { pH: { value: '7', missing: false }, TDS: { value: '', missing: true } }, fields.slice(1)), ValidationError);
});

test('assessment prioritizes limited quick mode and uses API route contracts', () => {
  assert.equal(assessmentPresentation({ assessment: 'predicted_safe' }, 'quick').style, 'caution');
  assert.equal(assessmentPresentation({ assessment: 'insufficient_evidence' }, 'full').style, 'caution');
  assert.equal(inferencePath('combined', 'quick', true), '/predict/quick?save_history=true');
  assert.equal(inferencePath('classification', 'full'), '/classify/full');
  assert.equal(inferencePath('regression', 'full'), '/ph/full');
});

test('result summary keeps quick and unavailable evidence visibly cautious', () => {
  const quick = resultSummary({ classification: { assessment: 'predicted_safe', model_label: 'safe' } }, { task: 'combined', mode: 'quick' });
  assert.equal(quick.style, 'caution');
  assert.equal(quick.title, 'Quick model signal: predicts safe');
  assert.equal(quick.details.length, 1);
  assert.match(quick.details[0], /limited screening/);

  const full = resultSummary({ classification: { assessment: 'insufficient_evidence', model_label: 'safe' } }, { task: 'combined', mode: 'full' });
  assert.equal(full.title, 'More readings needed');

  const regression = resultSummary({ estimate: 7.2 }, { task: 'regression', mode: 'full' });
  assert.equal(regression.style, 'caution');
  assert.equal(regression.title, 'Estimated pH');
  assert.deepEqual(regression.details, []);
  assert.equal(resultSummary({ estimate: 7.2 }, { task: 'regression', mode: 'quick' }).title, 'Estimated pH');
});

test('warning presentation maps explicit severity cues to non-green classes', () => {
  assert.equal(warningSeverity({ code: 'insufficient_evidence' }), 'caution');
  assert.equal(warningSeverity({ code: 'uncalibrated_probability' }), 'caution');
  assert.equal(warningSeverity({ code: 'training_range_note' }), 'info');
  assert.equal(warningSeverity({ code: 'outside_training_range' }), 'concern');
  assert.equal(warningSeverity({ code: 'unknown_training_range' }), 'caution');
  assert.equal(warningSeverity({ code: 'model_runtime_warning' }), 'caution');
});

test('summary translates server assessment rather than inferring safety from class or probability', () => {
  assert.equal(assessmentPresentation({ assessment: 'predicted_safe' }, 'full').text, 'Model predicts safe');
  assert.equal(assessmentPresentation({ assessment: 'predicted_unsafe' }, 'full').text, 'Model flags a concern');
  assert.equal(assessmentPresentation({ class_id: 1, probability_safe: 0.99 }, 'full').text, 'Assessment unavailable');
  const summary = resultSummary({ assessment: 'insufficient_evidence', class_id: 1, warnings: [
    { code: 'missing_measurements', features: ['Hardness'] },
  ] }, { task: 'classification', mode: 'full' });
  assert.equal(summary.title, 'More readings needed');
  assert.deepEqual(summary.details, ['Missing readings — measure the unrecorded inputs (Hardness).']);
  assert.deepEqual(summary.risks, []);
});
