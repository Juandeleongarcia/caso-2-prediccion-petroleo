import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parameters,number,modeLabel,resetEvaluation,freshnessLabel} from '../frontend/domain.mjs';
test('filters preserve all scientific dimensions and encode user values',()=>{const p=new URLSearchParams(parameters({model:'GRU',horizon:252,phase:'test',origin:'2024-01-02',barrels:0}));assert.equal(p.get('model'),'GRU');assert.equal(p.get('horizon'),'252');assert.equal(p.get('barrels'),'0')});
test('missing metrics cannot become invented zero values',()=>{for(const n of [null,undefined,NaN,Infinity])assert.equal(number(n),'No disponible');assert.equal(number(0),'0')});
test('archive mode cannot claim live inference',()=>{assert.match(modeLabel('archive'),/sin inferencia nueva/);assert.match(modeLabel('current'),/Inferencia real/)});
test('failed attempts are never labelled as successful data acquisition or training',()=>{const text=freshnessLabel({status:'failed',last_attempt_at:'2026-11-19',last_success_at:'2026-11-18'},'2026-11-11');assert.match(text,/2026-11-18/);assert.doesNotMatch(text,/2026-11-19/);assert.match(text,/No es la fecha de entrenamiento/);assert.match(freshnessLabel({},'2026-11-11'),/2026-11-11/)});
test('evaluation dimensions reset only dependent selectors',()=>{assert.deepEqual(resetEvaluation({model:'GRU',variant:'D',phase:'test',block:'test_2024H1',horizon:63},'model'),{model:'GRU',horizon:63});assert.equal(resetEvaluation({variant:'C',phase:'test',block:'x'},'phase').variant,'C')});
