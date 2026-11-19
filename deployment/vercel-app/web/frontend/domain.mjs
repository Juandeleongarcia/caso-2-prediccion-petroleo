export const labels={persistence:'Persistencia',RNN:'RNN-D',LSTM:'LSTM-B',GRU:'GRU-C'};
export const horizons={5:'Una semana · 5 días',63:'Tres meses · 63 días',252:'Un año · 252 días'};
export function parameters(state){return new URLSearchParams(Object.entries(state).filter(([,v])=>v!==undefined&&v!==null&&v!=='')).toString()}
export function number(v,digits=3){return v===null||v===undefined||!Number.isFinite(Number(v))?'No disponible':Number(v).toLocaleString('es-ES',{maximumFractionDigits:digits})}
export function modeLabel(source){return source==='archive'?'Backtesting archivado · sin inferencia nueva':'Inferencia real · pesos congelados'}
export function resetEvaluation(state,key){const next={...state};if(['experiment','model'].includes(key)){delete next.variant;delete next.phase;delete next.block}if(['variant','phase'].includes(key))delete next.block;return next}
