/* ============================================================================
   Social Media Monitoring Agent — frontend logic (vanilla JS + ECharts)
   ========================================================================== */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const COLORS = {
  positive: '#34d399',
  negative: '#f87171',
  neutral:  '#94a3b8',
};
const CLUSTER_COLORS = ['#6ea8fe', '#a78bfa', '#f0abfc', '#fbbf24', '#5eead4'];
const AXIS = '#5e6a82';
const GRID = '#232c3d';
const FONT = "Inter, system-ui, sans-serif";

let STATE = {
  mode: 'hybrid',
  data: null,
  charts: {},          // id -> echarts instance
  colorBy: 'cluster',
  polFilter: new Set(['positive', 'negative', 'neutral']),
  emoFilter: new Set(),
};

/* ── theme helpers for ECharts ──────────────────────────────────────────── */
const baseGrid = (over = {}) => ({ left: 40, right: 20, top: 30, bottom: 40, containLabel: true, ...over });
const axisStyle = {
  axisLine:  { lineStyle: { color: GRID } },
  axisTick:  { show: false },
  axisLabel: { color: AXIS, fontFamily: FONT, fontSize: 11 },
  splitLine: { lineStyle: { color: GRID, type: 'dashed' } },
};
const tooltipStyle = {
  backgroundColor: '#141925',
  borderColor: '#2e3a50',
  textStyle: { color: '#e6ebf4', fontFamily: FONT, fontSize: 12 },
  extraCssText: 'border-radius:9px;box-shadow:0 8px 24px -8px rgba(0,0,0,.6);',
};

function makeChart(id) {
  const el = document.getElementById(id);
  if (STATE.charts[id]) STATE.charts[id].dispose();
  const c = echarts.init(el, null, { renderer: 'canvas' });
  STATE.charts[id] = c;
  return c;
}
function resizeAll() { Object.values(STATE.charts).forEach(c => c.resize()); }
window.addEventListener('resize', resizeAll);

/* ── sidebar controls ───────────────────────────────────────────────────── */
$$('#mode button').forEach(b => b.addEventListener('click', () => {
  $$('#mode button').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  STATE.mode = b.dataset.mode;
  $('#alpha-field').classList.toggle('disabled', STATE.mode !== 'hybrid');
}));

$('#alpha').addEventListener('input', e => $('#alpha-val').textContent = (+e.target.value).toFixed(2));
$('#topk').addEventListener('input', e => $('#topk-val').textContent = e.target.value);

$('#query').addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });
$('#run').addEventListener('click', runSearch);

/* ── tabs ───────────────────────────────────────────────────────────────── */
$$('#tabs button').forEach(b => b.addEventListener('click', () => {
  $$('#tabs button').forEach(x => x.classList.remove('active'));
  $$('.panel').forEach(p => p.classList.remove('active'));
  b.classList.add('active');
  $(`.panel[data-panel="${b.dataset.tab}"]`).classList.add('active');
  // charts need a resize when their panel becomes visible
  requestAnimationFrame(resizeAll);
}));

/* ── color-by toggle (clusters) ─────────────────────────────────────────── */
$$('#color-by button').forEach(b => b.addEventListener('click', () => {
  $$('#color-by button').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  STATE.colorBy = b.dataset.by;
  if (STATE.data) renderUmap(STATE.data);
}));

/* ── busy state helpers ─────────────────────────────────────────────────── */
function setBusy(btn, busy) {
  btn.disabled = busy;
  btn.querySelector('.run-label').hidden = busy;
  btn.querySelector('.spinner').hidden = !busy;
}

/* ══════════════════════════════════════════════════════════════════════════
   SEARCH
   ══════════════════════════════════════════════════════════════════════════ */
async function runSearch() {
  const query = $('#query').value.trim();
  if (!query) { $('#query').focus(); return; }

  setBusy($('#run'), true);
  $('#empty').hidden = true;
  $('#results').hidden = true;
  $('#loading').hidden = false;
  $('#loading-text').textContent = 'Retrieving & analysing tweets…';

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        mode: STATE.mode,
        alpha: +$('#alpha').value,
        top_k: +$('#topk').value,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    STATE.data = data;
    renderAll(data);
  } catch (err) {
    $('#loading-text').textContent = '⚠ ' + err.message;
    return;
  } finally {
    setBusy($('#run'), false);
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   RENDER
   ══════════════════════════════════════════════════════════════════════════ */
function renderAll(d) {
  $('#loading').hidden = true;
  $('#results').hidden = false;

  // topbar
  $('#result-title').textContent = `Results for “${d.query}”`;
  $('#result-sub').innerHTML = `<b>${d.count}</b> tweets retrieved · scored across 11 emotions`;
  $('#topbar-stats').hidden = false;
  $('#stat-total').textContent = d.count;
  $('#stat-pos').textContent = d.polarity_counts.positive;
  $('#stat-neg').textContent = d.polarity_counts.negative;
  $('#stat-neu').textContent = d.polarity_counts.neutral;

  renderSentiment(d);
  renderEmoFilter(d);
  renderTweets(d);
  renderClusters(d);
  renderAspects(d);
  renderReport(d);

  requestAnimationFrame(resizeAll);
}

/* ── TAB 1: sentiment ───────────────────────────────────────────────────── */
function renderSentiment(d) {
  const pc = d.polarity_counts;

  // pie
  makeChart('chart-pie').setOption({
    tooltip: { ...tooltipStyle, trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    series: [{
      type: 'pie', radius: ['52%', '78%'], center: ['50%', '52%'],
      avoidLabelOverlap: true,
      itemStyle: { borderColor: '#141925', borderWidth: 3 },
      label: { color: '#e6ebf4', fontFamily: FONT, fontSize: 12, formatter: '{b}\n{d}%' },
      data: ['positive', 'negative', 'neutral'].map(p => ({
        name: p, value: pc[p], itemStyle: { color: COLORS[p] },
      })),
    }],
  });

  // polarity bar
  makeChart('chart-pol-bar').setOption({
    tooltip: { ...tooltipStyle, trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: baseGrid(),
    xAxis: { type: 'category', data: ['positive', 'negative', 'neutral'], ...axisStyle, splitLine: { show: false } },
    yAxis: { type: 'value', ...axisStyle },
    series: [{
      type: 'bar', barWidth: '46%',
      itemStyle: { borderRadius: [6, 6, 0, 0] },
      label: { show: true, position: 'top', color: '#e6ebf4', fontFamily: FONT },
      data: ['positive', 'negative', 'neutral'].map(p => ({ value: pc[p], itemStyle: { color: COLORS[p] } })),
    }],
  });

  // heatmap polarity × emotion
  const pols = ['positive', 'negative', 'neutral'];
  const cells = [];
  pols.forEach((p, yi) => d.heatmap[p].forEach((v, xi) => cells.push([xi, yi, +v.toFixed(2)])));
  makeChart('chart-heat').setOption({
    tooltip: { ...tooltipStyle, formatter: o => `${pols[o.value[1]]} · ${d.emotions_order[o.value[0]]}<br/><b>${o.value[2]}</b>` },
    grid: baseGrid({ top: 10, bottom: 60, left: 70 }),
    xAxis: { type: 'category', data: d.emotions_order, ...axisStyle, splitLine: { show: false }, axisLabel: { ...axisStyle.axisLabel, rotate: 40 } },
    yAxis: { type: 'category', data: pols, ...axisStyle, splitLine: { show: false } },
    visualMap: { min: 0, max: 1, calculable: false, show: false, inRange: { color: ['#1b2231', '#6ea8fe', '#a78bfa'] } },
    series: [{
      type: 'heatmap', data: cells,
      label: { show: true, color: '#cdd6e6', fontSize: 10, fontFamily: 'JetBrains Mono' },
      itemStyle: { borderColor: '#141925', borderWidth: 2 },
    }],
  });

  // overall emotion bar
  const emo = Object.entries(d.emotion_means).sort((a, b) => b[1] - a[1]);
  makeChart('chart-emo').setOption({
    tooltip: { ...tooltipStyle, trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: o => `${o[0].name}<br/><b>${o[0].value.toFixed(3)}</b>` },
    grid: baseGrid({ bottom: 60 }),
    xAxis: { type: 'category', data: emo.map(e => e[0]), ...axisStyle, splitLine: { show: false }, axisLabel: { ...axisStyle.axisLabel, rotate: 35 } },
    yAxis: { type: 'value', ...axisStyle },
    series: [{
      type: 'bar', barWidth: '54%',
      itemStyle: {
        borderRadius: [5, 5, 0, 0],
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#a78bfa' }, { offset: 1, color: '#6ea8fe' },
        ]),
      },
      data: emo.map(e => +e[1].toFixed(3)),
    }],
  });
}

/* ── TAB 2: tweets ──────────────────────────────────────────────────────── */
function renderEmoFilter(d) {
  const wrap = $('#emo-filter');
  wrap.innerHTML = '';
  STATE.emoFilter.clear();
  d.emotions_order.forEach(e => {
    const b = document.createElement('button');
    b.className = 'chip'; b.textContent = e; b.dataset.emo = e;
    b.addEventListener('click', () => {
      b.classList.toggle('active');
      b.classList.contains('active') ? STATE.emoFilter.add(e) : STATE.emoFilter.delete(e);
      renderTweets(STATE.data);
    });
    wrap.appendChild(b);
  });
}

$$('#pol-filter .chip').forEach(b => b.addEventListener('click', () => {
  b.classList.toggle('active');
  const p = b.dataset.pol;
  b.classList.contains('active') ? STATE.polFilter.add(p) : STATE.polFilter.delete(p);
  if (STATE.data) renderTweets(STATE.data);
}));

function renderTweets(d) {
  const tbody = $('#tweet-body');
  const rows = d.tweets.filter(t =>
    STATE.polFilter.has(t.polarity) &&
    (STATE.emoFilter.size === 0 || t.emotions.some(e => STATE.emoFilter.has(e)))
  );
  $('#tweet-count').innerHTML = `<b>${rows.length}</b> / ${d.tweets.length}`;

  tbody.innerHTML = rows.map(t => `
    <tr>
      <td class="rank">${t.rank}</td>
      <td class="txt">${esc(t.tweet)}</td>
      <td><span class="pol-tag ${t.polarity}">${t.polarity}</span></td>
      <td class="score">${t.score.toFixed(4)}</td>
      <td><div class="emo-tags">${t.emotions.map(e => `<span>${e}</span>`).join('') || '<span style="opacity:.4">—</span>'}</div></td>
    </tr>`).join('');
}

/* ── TAB 3: clusters ────────────────────────────────────────────────────── */
function renderClusters(d) {
  $('#cluster-terms').innerHTML = Object.entries(d.cluster_terms)
    .map(([cl, terms]) => `<li><b>C${cl}</b>${esc(terms)}</li>`).join('');
  renderUmap(d);
}

function renderUmap(d) {
  const by = STATE.colorBy;
  let series;
  if (by === 'polarity') {
    series = ['positive', 'negative', 'neutral'].map(p => ({
      name: p, type: 'scatter', symbolSize: 9,
      itemStyle: { color: COLORS[p], opacity: .75 },
      data: d.points.filter(pt => pt.polarity === p).map(pt => ({ value: [pt.x, pt.y], t: pt.tweet, s: pt.score })),
    }));
  } else {
    const clusters = [...new Set(d.points.map(p => p.cluster))].sort();
    series = clusters.map((cl, i) => ({
      name: 'Cluster ' + cl, type: 'scatter', symbolSize: 9,
      itemStyle: { color: CLUSTER_COLORS[i % CLUSTER_COLORS.length], opacity: .75 },
      data: d.points.filter(pt => pt.cluster === cl).map(pt => ({ value: [pt.x, pt.y], t: pt.tweet, s: pt.score })),
    }));
  }
  makeChart('chart-umap').setOption({
    tooltip: { ...tooltipStyle, formatter: o => `${esc(o.data.t)}<br/><b>score ${o.data.s}</b>` },
    legend: { textStyle: { color: '#93a0b8', fontFamily: FONT }, top: 0, icon: 'circle' },
    grid: baseGrid({ top: 40 }),
    xAxis: { type: 'value', ...axisStyle, axisLabel: { show: false } },
    yAxis: { type: 'value', ...axisStyle, axisLabel: { show: false } },
    series,
  });
}

/* ── TAB 4: aspects ─────────────────────────────────────────────────────── */
function renderAspects(d) {
  const asp = d.aspects;
  const labels = asp.map(a => `C${a.cluster}`);

  // grouped polarity bars
  makeChart('chart-asp-bar').setOption({
    tooltip: { ...tooltipStyle, trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { textStyle: { color: '#93a0b8', fontFamily: FONT }, top: 0, icon: 'circle' },
    grid: baseGrid({ top: 36 }),
    xAxis: { type: 'category', data: labels, ...axisStyle, splitLine: { show: false } },
    yAxis: { type: 'value', ...axisStyle },
    series: ['positive', 'negative', 'neutral'].map(p => ({
      name: p, type: 'bar',
      itemStyle: { color: COLORS[p], borderRadius: [4, 4, 0, 0] },
      data: asp.map(a => a.polarity[p]),
    })),
  });

  // radar
  makeChart('chart-radar').setOption({
    tooltip: { ...tooltipStyle },
    legend: { textStyle: { color: '#93a0b8', fontFamily: FONT }, bottom: 0, icon: 'circle', type: 'scroll' },
    radar: {
      indicator: d.emotions_order.map(e => ({ name: e, max: 1 })),
      axisName: { color: AXIS, fontFamily: FONT, fontSize: 10 },
      splitLine: { lineStyle: { color: GRID } },
      splitArea: { areaStyle: { color: ['transparent', 'rgba(110,168,254,.03)'] } },
      axisLine: { lineStyle: { color: GRID } },
    },
    series: [{
      type: 'radar', symbol: 'none',
      data: asp.map((a, i) => ({
        name: `C${a.cluster}`, value: a.emotions,
        lineStyle: { color: CLUSTER_COLORS[i % CLUSTER_COLORS.length], width: 2 },
        areaStyle: { opacity: .12, color: CLUSTER_COLORS[i % CLUSTER_COLORS.length] },
        itemStyle: { color: CLUSTER_COLORS[i % CLUSTER_COLORS.length] },
      })),
    }],
  });

  // aspect heatmap
  const cells = [];
  asp.forEach((a, yi) => a.emotions.forEach((v, xi) => cells.push([xi, yi, +v.toFixed(2)])));
  makeChart('chart-asp-heat').setOption({
    tooltip: { ...tooltipStyle, formatter: o => `C${asp[o.value[1]].cluster} · ${d.emotions_order[o.value[0]]}<br/><b>${o.value[2]}</b>` },
    grid: baseGrid({ top: 10, bottom: 60, left: 50 }),
    xAxis: { type: 'category', data: d.emotions_order, ...axisStyle, splitLine: { show: false }, axisLabel: { ...axisStyle.axisLabel, rotate: 40 } },
    yAxis: { type: 'category', data: labels, ...axisStyle, splitLine: { show: false } },
    visualMap: { min: 0, max: 1, show: false, inRange: { color: ['#1b2231', '#6ea8fe', '#a78bfa'] } },
    series: [{
      type: 'heatmap', data: cells,
      label: { show: true, color: '#cdd6e6', fontSize: 10, fontFamily: 'JetBrains Mono' },
      itemStyle: { borderColor: '#141925', borderWidth: 2 },
    }],
  });

  // representative tweets
  $('#aspect-reps').innerHTML = asp.map(a => `
    <div class="rep">
      <div class="rep-head">C${a.cluster} — ${esc(a.terms)}
        <span class="pol-tag ${a.rep_polarity}">${a.rep_polarity}</span>
      </div>
      <p>${esc(a.rep_tweet) || '<span class="empty-note">no tweets</span>'}</p>
    </div>`).join('');
}

/* ── TAB 5: report ──────────────────────────────────────────────────────── */
function renderReport(d) {
  $('#report-title').textContent = `Summary report — “${d.query}”`;
  const ICON = { positive: '😊', negative: '😠', neutral: '😐' };

  $('#report-body').innerHTML = ['positive', 'negative', 'neutral'].map(p => {
    const sec = d.extractive[p];
    const ext = sec.items.length ? `
      <div class="ext-label">Extractive — 3 most representative tweets</div>
      ${sec.items.map(it => `
        <div class="ext-item">
          <p>${esc(it.tweet)}</p>
          <small>sim=${it.sim} &nbsp;|&nbsp; ${it.emotions.join(' · ') || '—'}</small>
        </div>`).join('')}` : '<p class="empty-note">No tweets in this group.</p>';

    return `
      <div class="report-section ${p}" data-pol="${p}">
        <h5>${ICON[p]} ${cap(p)} <span class="cnt">${sec.count} tweets</span></h5>
        <div class="abs-slot"></div>
        ${ext}
      </div>`;
  }).join('');
}

$('#gen-abs').addEventListener('click', async () => {
  if (!STATE.data) return;
  setBusy($('#gen-abs'), true);
  try {
    const res = await fetch('/api/summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result_id: STATE.data.result_id }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { abstractive } = await res.json();
    ['positive', 'negative', 'neutral'].forEach(p => {
      const slot = $(`.report-section[data-pol="${p}"] .abs-slot`);
      if (slot) slot.innerHTML = `<div class="abs-box"><span class="abs-label">Abstractive summary</span>${esc(abstractive[p])}</div>`;
    });
  } catch (err) {
    alert('Summary failed: ' + err.message);
  } finally {
    setBusy($('#gen-abs'), false);
  }
});

/* ── utils ──────────────────────────────────────────────────────────────── */
function esc(s) {
  return String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
