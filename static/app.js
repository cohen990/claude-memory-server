/* Memory Browser — vanilla JS SPA */

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------

const tabs = document.querySelectorAll('nav .tab');
const panels = document.querySelectorAll('.tab-content');
let activeTab = 'graph';

tabs.forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    if (tab === activeTab) return;
    activeTab = tab;

    tabs.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    panels.forEach(p => p.classList.toggle('active', p.id === 'tab-' + tab));

    stopPolling();
    if (tab === 'graph' && !graphLoaded) loadGraph();
    if (tab === 'recalls') { if (!recallsLoaded) loadRecalls(); startRecallsPoll(); }
    if (tab === 'stats') { loadStats(); startStatsPoll(); }
  });
});

// ---------------------------------------------------------------------------
// Graph module
// ---------------------------------------------------------------------------

let cy = null;
let graphLoaded = false;

async function loadGraph() {
  const status = document.getElementById('graph-status');
  status.textContent = 'Loading graph...';

  try {
    const resp = await fetch('/api/graph');
    const data = await resp.json();

    const elements = [];

    for (const node of data.nodes) {
      elements.push({
        data: {
          id: node.id,
          label: node.text.slice(0, 60),
          type: node.type,
          fullText: node.text,
        },
      });
    }

    // Build degree map for node sizing
    const degree = {};
    for (const edge of data.edges) {
      degree[edge.source_id] = (degree[edge.source_id] || 0) + 1;
      degree[edge.target_id] = (degree[edge.target_id] || 0) + 1;
    }

    for (const edge of data.edges) {
      elements.push({
        data: {
          id: edge.source_id + '-' + edge.target_id,
          source: edge.source_id,
          target: edge.target_id,
          weight: edge.weight,
        },
      });
    }

    cy = cytoscape({
      container: document.getElementById('cy'),
      elements: elements,
      style: [
        {
          selector: 'node',
          style: {
            'label': '',
            'width': function(ele) {
              const d = degree[ele.id()] || 1;
              return Math.min(8 + d * 3, 40);
            },
            'height': function(ele) {
              const d = degree[ele.id()] || 1;
              return Math.min(8 + d * 3, 40);
            },
            'background-color': function(ele) {
              return ele.data('type') === 'vibe' ? '#6366f1' : '#14b8a6';
            },
            'border-width': 0,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 2,
            'border-color': '#fff',
          },
        },
        {
          selector: 'edge',
          style: {
            'width': function(ele) {
              return 0.5 + ele.data('weight') * 3;
            },
            'line-color': '#2a2a35',
            'opacity': function(ele) {
              return 0.2 + ele.data('weight') * 0.6;
            },
            'curve-style': 'bezier',
          },
        },
        {
          selector: 'edge.edge-highlighted',
          style: {
            'line-color': '#818cf8',
            'opacity': 0.9,
            'width': function(ele) {
              return 1.5 + ele.data('weight') * 3;
            },
          },
        },
        {
          selector: 'node.neighbor-highlighted',
          style: {
            'border-width': 1.5,
            'border-color': '#818cf8',
          },
        },
      ],
      layout: { name: 'preset' },
      autoungrabifyNodes: true,
      minZoom: 0.1,
      maxZoom: 5,
    });

    cy.on('tap', 'node', function(evt) {
      // Clear previous selection highlights
      cy.edges().removeClass('edge-highlighted');
      cy.nodes().removeClass('neighbor-highlighted');

      const node = evt.target;
      const connectedEdges = node.connectedEdges();
      connectedEdges.addClass('edge-highlighted');
      connectedEdges.connectedNodes().not(node).addClass('neighbor-highlighted');

      showNodeSidebar(node.id());
    });

    cy.on('tap', function(evt) {
      if (evt.target === cy) {
        cy.edges().removeClass('edge-highlighted');
        cy.nodes().removeClass('neighbor-highlighted');
        closeSidebar();
      }
    });

    status.textContent = data.nodes.length + ' nodes, ' + data.edges.length + ' edges — laying out...';
    graphLoaded = true;

    // Run force-directed layout async so initial render isn't blocked
    requestAnimationFrame(() => {
      cy.layout({
        name: 'cose',
        animate: true,
        animationDuration: 500,
        fit: true,
        nodeOverlap: 20,
        componentSpacing: 40,
      }).run();
      cy.one('layoutstop', () => {
        status.textContent = data.nodes.length + ' nodes, ' + data.edges.length + ' edges';
      });
    });
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
  }
}

async function showNodeSidebar(nodeId) {
  try {
    const [nodeResp, neighborsResp] = await Promise.all([
      fetch('/api/nodes/' + encodeURIComponent(nodeId)),
      fetch('/api/nodes/' + encodeURIComponent(nodeId) + '/neighbors'),
    ]);
    const nodeData = await nodeResp.json();
    const neighborsData = await neighborsResp.json();

    const sidebar = document.getElementById('sidebar');
    const typeEl = document.getElementById('sidebar-type');
    const textEl = document.getElementById('sidebar-text');
    const metaEl = document.getElementById('sidebar-meta');
    const neighborsList = document.getElementById('sidebar-neighbors');

    typeEl.textContent = nodeData.node.type;
    typeEl.className = nodeData.node.type;
    textEl.textContent = nodeData.node.text;

    metaEl.innerHTML =
      '<div>ID: ' + nodeData.node.id.slice(0, 8) + '...</div>' +
      '<div>Created: ' + formatDate(nodeData.node.created_at) + '</div>' +
      '<div>Updated: ' + formatDate(nodeData.node.updated_at) + '</div>' +
      '<div>Sources: ' + (nodeData.node.source_ids || []).length + '</div>' +
      '<div>Edges: ' + nodeData.edges.length + '</div>';

    neighborsList.innerHTML = '';
    for (const n of neighborsData.neighbors) {
      const li = document.createElement('li');
      li.innerHTML =
        '<span class="neighbor-type ' + n.node.type + '">' + n.node.type + '</span>' +
        '<span class="neighbor-weight">w=' + n.edge.weight.toFixed(2) + '</span>' +
        '<span class="neighbor-text">' + escapeHtml(n.node.text.slice(0, 80)) + '</span>';
      li.addEventListener('click', () => {
        // Select and center the neighbor in the graph
        const cyNode = cy.getElementById(n.node.id);
        if (cyNode.length) {
          cy.animate({ center: { eles: cyNode }, zoom: 2 }, { duration: 300 });
          cyNode.select();
        }
        showNodeSidebar(n.node.id);
      });
      neighborsList.appendChild(li);
    }

    sidebar.classList.remove('hidden');
  } catch (err) {
    console.error('Failed to load node:', err);
  }
}

function closeSidebar() {
  document.getElementById('sidebar').classList.add('hidden');
}

// Toolbar buttons
document.getElementById('sidebar-close').addEventListener('click', closeSidebar);
document.getElementById('btn-fit').addEventListener('click', () => { if (cy) cy.fit(); });
document.getElementById('btn-relayout').addEventListener('click', () => {
  if (cy) cy.layout({ name: 'cose', animate: true, animationDuration: 500 }).run();
});

// Type filters
document.getElementById('filter-vibe').addEventListener('change', applyFilters);
document.getElementById('filter-detail').addEventListener('change', applyFilters);

function applyFilters() {
  if (!cy) return;
  const showVibe = document.getElementById('filter-vibe').checked;
  const showDetail = document.getElementById('filter-detail').checked;

  cy.batch(() => {
    cy.nodes().forEach(node => {
      const type = node.data('type');
      if ((type === 'vibe' && !showVibe) || (type === 'detail' && !showDetail)) {
        node.style('display', 'none');
      } else {
        node.style('display', 'element');
      }
    });
  });
}

// Graph search
let graphSearchTimer = null;
document.getElementById('graph-search').addEventListener('input', function() {
  clearTimeout(graphSearchTimer);
  graphSearchTimer = setTimeout(() => applyGraphSearch(this.value), 200);
});

function applyGraphSearch(query) {
  if (!cy) return;
  const term = query.trim().toLowerCase();

  cy.batch(() => {
    if (!term) {
      // Reset all nodes to default appearance
      cy.nodes().forEach(node => {
        node.style({
          'opacity': 1,
          'border-width': 0,
          'border-color': '#fff',
        });
      });
      cy.edges().style({ 'opacity': null });
      return;
    }

    cy.nodes().forEach(node => {
      const text = (node.data('fullText') || '').toLowerCase();
      if (text.includes(term)) {
        node.style({
          'opacity': 1,
          'border-width': 3,
          'border-color': '#fff',
        });
      } else {
        node.style({
          'opacity': 0.15,
          'border-width': 0,
        });
      }
    });
    cy.edges().style({ 'opacity': 0.05 });
  });
}

// ---------------------------------------------------------------------------
// Recalls module
// ---------------------------------------------------------------------------

let recallsLoaded = false;
let recallsOffset = 0;
const RECALLS_PAGE = 20;
let recallsPollTimer = null;

document.getElementById('recalls-load').addEventListener('click', () => {
  recallsOffset = 0;
  loadRecalls();
});

document.getElementById('recalls-more').addEventListener('click', () => {
  recallsOffset += RECALLS_PAGE;
  loadRecalls(true);
});

function buildRecallCard(recall) {
  const card = document.createElement('div');
  card.className = 'recall-card';
  card.dataset.recallId = recall.recall_id;

  const sessionLabel = recall.session_id
    ? recall.session_id.slice(0, 12) + '...'
    : 'no session';

  let header = '<div class="recall-header">' +
    '<span>' + formatDate(recall.created_at) + '</span>' +
    (recall.session_id
      ? '<a href="#" class="session-link" data-session="' + escapeHtml(recall.session_id) + '">' + sessionLabel + '</a>'
      : '<span>' + sessionLabel + '</span>') +
    '</div>';

  let querySection = '';
  if (recall.query_text) {
    querySection = '<details class="recall-query">' +
      '<summary>Query</summary>' +
      '<p>' + escapeHtml(recall.query_text) + '</p>' +
      '</details>';
  }

  let results = '';
  for (const r of recall.results) {
    const reflection = r.reflection
      ? '<span class="reflection-badge reflection-' + r.reflection + '">' + r.reflection + '</span>'
      : '';
    results += '<div class="recall-result">' +
      '<span class="recall-result-type ' + (r.type || '') + '">' + (r.type || '?') + '</span>' +
      '<span class="recall-result-text">' + escapeHtml(r.text || '(deleted node)') + '</span>' +
      '<span class="recall-result-meta">' +
        '<span class="recall-result-sim">' + (r.similarity || 0).toFixed(3) + '</span> ' +
        reflection +
      '</span>' +
      '</div>';
  }

  card.innerHTML = header + querySection + results;
  return card;
}

function wireSessionLinks(container) {
  container.querySelectorAll('.session-link:not([data-wired])').forEach(link => {
    link.dataset.wired = '1';
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const sid = link.dataset.session;
      document.getElementById('recalls-session').value = sid;
      recallsOffset = 0;
      loadRecalls();
    });
  });
}

async function loadRecalls(append) {
  const sessionId = document.getElementById('recalls-session').value.trim();
  const list = document.getElementById('recalls-list');
  const moreBtn = document.getElementById('recalls-more');

  // Full reset on manual Load (not append, not poll)
  if (!append) list.innerHTML = '';

  const params = new URLSearchParams({ limit: RECALLS_PAGE });
  if (sessionId) params.set('session_id', sessionId);

  try {
    const resp = await fetch('/api/recalls?' + params);
    const data = await resp.json();

    for (const recall of data.recalls) {
      const card = buildRecallCard(recall);
      list.appendChild(card);
    }

    wireSessionLinks(list);
    moreBtn.classList.toggle('hidden', data.recalls.length < RECALLS_PAGE);
    recallsLoaded = true;
  } catch (err) {
    list.innerHTML = '<div style="color: var(--reflection-M)">Error: ' + err.message + '</div>';
  }
}

async function pollRecalls() {
  const sessionId = document.getElementById('recalls-session').value.trim();
  const list = document.getElementById('recalls-list');

  const params = new URLSearchParams({ limit: RECALLS_PAGE });
  if (sessionId) params.set('session_id', sessionId);

  try {
    const resp = await fetch('/api/recalls?' + params);
    const data = await resp.json();

    // Collect IDs already in the DOM
    const existingIds = new Set(
      Array.from(list.querySelectorAll('.recall-card[data-recall-id]'))
        .map(el => el.dataset.recallId)
    );

    // Prepend new cards (response is newest-first, so insert in reverse to maintain order)
    const newRecalls = data.recalls.filter(r => !existingIds.has(r.recall_id));
    for (let i = newRecalls.length - 1; i >= 0; i--) {
      const card = buildRecallCard(newRecalls[i]);
      list.prepend(card);
    }

    wireSessionLinks(list);
  } catch (err) {
    // Silent fail on poll — don't nuke the DOM
    console.error('Recalls poll failed:', err);
  }
}

// ---------------------------------------------------------------------------
// Stats module
// ---------------------------------------------------------------------------

async function loadStats() {
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = '';

  try {
    const resp = await fetch('/api/stats');
    const data = await resp.json();

    const cards = [];

    // Graph stats
    cards.push({ label: 'Graph Nodes', value: data.graph.total_nodes });
    if (data.graph.nodes_by_type) {
      cards.push({ label: 'Vibes', value: data.graph.nodes_by_type.vibe || 0, cls: 'vibe' });
      cards.push({ label: 'Details', value: data.graph.nodes_by_type.detail || 0, cls: 'detail' });
    }
    cards.push({ label: 'Graph Edges', value: data.graph.total_edges });
    cards.push({ label: 'Activated Edges', value: data.graph.activated_edges });

    // ChromaDB stats
    if (data.chromadb) {
      cards.push({ label: 'Conversations', value: data.chromadb.total_documents });
      cards.push({ label: 'Subchunks', value: data.chromadb.total_subchunks });
      cards.push({ label: 'User Inputs', value: data.chromadb.total_user_inputs });
      cards.push({ label: 'Queue Pending', value: data.chromadb.queue_pending });
      if (data.chromadb.pending_dream != null) {
        cards.push({ label: 'Pending Dream', value: data.chromadb.pending_dream });
      }
    }

    // Reflection distribution
    if (data.reflections && Object.keys(data.reflections).length > 0) {
      for (const [code, count] of Object.entries(data.reflections)) {
        cards.push({ label: 'Reflection: ' + code, value: count });
      }
    }

    for (const card of cards) {
      const el = document.createElement('div');
      el.className = 'stat-card';
      el.innerHTML =
        '<div class="stat-label">' + card.label + '</div>' +
        '<div class="stat-value' + (card.cls ? ' ' + card.cls : '') + '">' +
        card.value.toLocaleString() + '</div>';
      grid.appendChild(el);
    }
  } catch (err) {
    grid.innerHTML = '<div style="color: var(--reflection-M)">Error: ' + err.message + '</div>';
  }

  // Load reflection timeline chart
  loadReflectionTimeline();
}

async function loadReflectionTimeline() {
  try {
    const resp = await fetch('/api/reflection-timeline');
    const buckets = await resp.json();
    drawReflectionChart(buckets);
  } catch (err) {
    console.error('Failed to load reflection timeline:', err);
  }
}

function drawReflectionChart(buckets) {
  const canvas = document.getElementById('reflection-chart');
  const container = document.getElementById('reflection-chart-container');
  if (!Array.isArray(buckets) || buckets.length === 0) {
    container.style.display = 'none';
    return;
  }
  container.style.display = '';

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const H = rect.height;

  ctx.clearRect(0, 0, W, H);

  const CODES = ['U', 'I', 'N', 'D', 'M'];
  const COLORS = {
    U: '#22c55e',  // green
    I: '#3b82f6',  // blue
    N: '#eab308',  // yellow
    D: '#f97316',  // orange
    M: '#ef4444',  // red
  };

  // Find max value across all series
  var maxVal = 0;
  for (const b of buckets) {
    for (const code of CODES) {
      if (b[code] > maxVal) maxVal = b[code];
    }
  }
  if (maxVal === 0) maxVal = 1;

  // Chart area with padding for labels
  const padL = 40;
  const padR = 16;
  const padT = 10;
  const padB = 50;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  // Grid lines
  ctx.strokeStyle = '#2a2a35';
  ctx.lineWidth = 1;
  const gridLines = 4;
  for (var i = 0; i <= gridLines; i++) {
    var y = padT + (chartH / gridLines) * i;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();

    // Y-axis label
    var val = Math.round(maxVal * (1 - i / gridLines));
    ctx.fillStyle = '#8888a0';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(val, padL - 6, y + 3);
  }

  // X-axis labels (show a subset)
  var labelEvery = Math.max(1, Math.floor(buckets.length / 8));
  ctx.textAlign = 'center';
  for (var i = 0; i < buckets.length; i += labelEvery) {
    var x = padL + (i / (buckets.length - 1 || 1)) * chartW;
    var d = new Date(buckets[i].bucket);
    var label = (d.getMonth() + 1) + '/' + d.getDate() + ' ' +
      d.getHours().toString().padStart(2, '0') + ':00';
    ctx.fillStyle = '#8888a0';
    ctx.font = '10px monospace';
    ctx.save();
    ctx.translate(x, H - padB + 14);
    ctx.rotate(-0.4);
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }

  // Draw lines
  for (const code of CODES) {
    ctx.strokeStyle = COLORS[code];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var i = 0; i < buckets.length; i++) {
      var x = padL + (i / (buckets.length - 1 || 1)) * chartW;
      var y = padT + chartH - (buckets[i][code] / maxVal) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Legend
  var legendX = padL + 8;
  var legendY = padT + 14;
  var LABELS = { U: 'Used', I: 'Interesting', N: 'Noise', D: 'Distracting', M: 'Misleading' };
  for (const code of CODES) {
    ctx.fillStyle = COLORS[code];
    ctx.fillRect(legendX, legendY - 8, 12, 3);
    ctx.fillStyle = '#d4d4dc';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(LABELS[code], legendX + 16, legendY - 4);
    legendX += ctx.measureText(LABELS[code]).width + 32;
  }
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

const POLL_INTERVAL = 7000; // 7 seconds
let pollTimer = null;

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

function startRecallsPoll() {
  stopPolling();
  pollTimer = setInterval(() => pollRecalls(), POLL_INTERVAL);
}

function startStatsPoll() {
  stopPolling();
  pollTimer = setInterval(() => loadStats(), POLL_INTERVAL);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Initial load
// ---------------------------------------------------------------------------

loadGraph();
