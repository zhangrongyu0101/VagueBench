const turns = [
  ['“Are you looking to invest, collect, or just research?”', '“Invest, to hedge inflation.”'],
  ['“What is your budget, risk level, and time horizon?”', '“$500K, medium risk, and 2–3 years.”'],
  ['“What do you hold now? Would you prefer ETFs or physical metal?”', '“Stocks and funds. I prefer ETFs.”']
];
document.querySelectorAll('[data-turn]').forEach(button => {
  button.addEventListener('click', () => {
    const [question, answer] = turns[Number(button.dataset.turn)];
    document.querySelector('#dialogue .question').textContent = question;
    document.querySelector('#dialogue .answer').textContent = answer;
    document.querySelectorAll('[data-turn]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  });
});

const splits = {
  intent: {
    rank: 'ask',
    columns: ['ask', 'cov', 'iar'],
    note: 'Intent-vague queries hide the user goal but keep some details. Ranked by Ask%, the share of queries where the model asks before answering. Coverage F1 and IAR are also higher-is-better. Numbers are from the paper.'
  },
  fully: {
    rank: 'ask',
    columns: ['ask', 'cov', 'iar'],
    note: 'Fully-vague queries hide both intent and task details. Ranked by Ask%. Coverage F1 compares the final answer with the gold information set. IAR is hidden information recovered per elicitation turn. Numbers are from the paper.'
  },
  complete: {
    rank: 'cov',
    columns: ['ask', 'cov'],
    note: 'Information-complete seed queries. Ranked by coverage F1. Ask% is lower-is-better here: asking means interrupting a request that was already specified. Numbers are from the paper.'
  }
};

const headers = {
  ask: 'Ask%',
  cov: 'Cov. F1',
  iar: 'IAR'
};

const models = [
  { name: 'GPT-5.4', group: 'closed', kind: 'Closed-source', intent: { ask: 1.6, cov: 0.58, iar: 0.06 }, fully: { ask: 10.8, cov: 0.53, iar: 0.65 } },
  { name: 'Claude Sonnet 4.6', group: 'closed', kind: 'Closed-source', intent: { ask: 8.8, cov: 0.60, iar: 0.08 }, fully: { ask: 8.4, cov: 0.52, iar: 0.56 } },
  { name: 'Gemini 3.1 Pro', group: 'closed', kind: 'Closed-source', intent: { ask: 2.4, cov: 0.57, iar: 0.00 }, fully: { ask: 3.6, cov: 0.54, iar: 0.80 } },
  { name: 'Qwen3-8B', group: 'open', kind: 'Open-source', intent: { ask: 1.2, cov: 0.47, iar: 0.00 }, fully: { ask: 1.6, cov: 0.36, iar: 0.21 } },
  { name: 'Llama-3.1-8B-Instruct', group: 'open', kind: 'Open-source', intent: { ask: 0.8, cov: 0.50, iar: 0.00 }, fully: { ask: 1.2, cov: 0.35, iar: 0.23 } },
  { name: 'Qwen3.5-27B', group: 'open', kind: 'Open-source', intent: { ask: 1.6, cov: 0.49, iar: 0.08 }, fully: { ask: 2.0, cov: 0.38, iar: 0.37 }, complete: { ask: 1.6, cov: 0.58 } },
  { name: 'GPT-oss-120B', group: 'open', kind: 'Open-source', intent: { ask: 2.0, cov: 0.46, iar: 0.02 }, fully: { ask: 3.6, cov: 0.36, iar: 0.35 } },
  { name: 'Qwen3.5-27B + Prompt', group: 'method', kind: 'Method', intent: { ask: 38.4, cov: 0.52, iar: 0.22 }, fully: { ask: 47.2, cov: 0.50, iar: 0.38 }, complete: { ask: 42.8, cov: 0.59 } },
  { name: 'Qwen3.5-27B + SFT', group: 'method', kind: 'Method', intent: { ask: 82.0, cov: 0.60, iar: 0.64 }, fully: { ask: 85.6, cov: 0.63, iar: 0.80 }, complete: { ask: 6.4, cov: 0.71 } },
  { name: 'Qwen3.5-27B + DialogGRPO', group: 'method', kind: 'Method', ours: true, intent: { ask: 88.4, cov: 0.64, iar: 0.76 }, fully: { ask: 91.2, cov: 0.67, iar: 0.87 }, complete: { ask: 5.2, cov: 0.74 } }
];

const boardHead = document.querySelector('#board-head');
const boardBody = document.querySelector('#board-body');
const boardNote = document.querySelector('#board-note');
let activeSplit = 'intent';
let activeGroup = 'all';

function formatMetric(key, value) {
  return key === 'ask' ? `${value.toFixed(1)}%` : value.toFixed(2);
}

function metricCell(key, value, warn) {
  const cell = document.createElement('td');
  cell.className = warn ? 'metric warn' : 'metric';
  const label = document.createElement('div');
  label.className = 'metric-top';
  label.textContent = formatMetric(key, value);
  const track = document.createElement('span');
  track.className = 'bar';
  const fill = document.createElement('span');
  const width = key === 'ask' ? value : value * 100;
  fill.style.width = `${Math.max(0, Math.min(100, width))}%`;
  track.append(fill);
  cell.append(label, track);
  return cell;
}

function renderBoard() {
  const split = splits[activeSplit];
  const ranked = models
    .filter(model => model[activeSplit] && (activeGroup === 'all' || model.group === activeGroup))
    .sort((a, b) => b[activeSplit][split.rank] - a[activeSplit][split.rank] || a.name.localeCompare(b.name));

  boardHead.replaceChildren();
  const headRow = document.createElement('tr');
  ['Rank', 'Model', 'Type', ...split.columns.map(key => {
    const arrow = activeSplit === 'complete' && key === 'ask' ? ' ↓' : ' ↑';
    return headers[key] + arrow;
  })].forEach(text => {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = text;
    headRow.append(cell);
  });
  boardHead.append(headRow);

  boardBody.replaceChildren();
  if (!ranked.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.className = 'empty';
    cell.colSpan = 3 + split.columns.length;
    cell.textContent = 'No evaluated models in this group for the selected split.';
    row.append(cell);
    boardBody.append(row);
  } else {
    ranked.forEach((model, index) => {
      const scores = model[activeSplit];
      const row = document.createElement('tr');
      if (model.ours) row.className = 'ours';
      const rankCell = document.createElement('td');
      const rank = document.createElement('span');
      rank.className = index === 0 ? 'rank top' : 'rank';
      rank.textContent = String(index + 1).padStart(2, '0');
      rankCell.append(rank);
      const nameCell = document.createElement('td');
      nameCell.className = 'model';
      nameCell.textContent = model.name;
      const kindCell = document.createElement('td');
      kindCell.className = 'kind';
      kindCell.textContent = model.kind;
      row.append(rankCell, nameCell, kindCell);
      split.columns.forEach(key => {
        row.append(metricCell(key, scores[key], activeSplit === 'complete' && key === 'ask'));
      });
      boardBody.append(row);
    });
  }
  boardNote.textContent = split.note;
}

document.querySelectorAll('[data-split]').forEach(button => {
  button.addEventListener('click', () => {
    activeSplit = button.dataset.split;
    document.querySelectorAll('[data-split]').forEach(item => item.setAttribute('aria-selected', String(item === button)));
    renderBoard();
  });
});

document.querySelectorAll('[data-group]').forEach(button => {
  button.addEventListener('click', () => {
    activeGroup = button.dataset.group;
    document.querySelectorAll('[data-group]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    renderBoard();
  });
});

renderBoard();
