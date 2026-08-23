/* Grouped bar chart for the comparison page.
 *
 * Two things make this less trivial than a one-off Chart.js call:
 *
 *  1. The results block polls itself via HTMX while a comparison is running,
 *     so the <canvas> and its data script are replaced wholesale every few
 *     seconds. We therefore rebuild on `htmx:afterSettle` and destroy the
 *     previous chart instance first -- otherwise Chart.js leaks one live
 *     instance (and one animation loop) per poll.
 *
 *  2. Chart.js resolves colours once at construction, so a dark-mode toggle
 *     leaves the axes unreadable until we rebuild. `litRedrawCharts` is the
 *     hook base.html calls on toggle.
 */
(function () {
  'use strict';

  var chart = null;

  // One colour per metric series. Chosen to stay distinguishable in both
  // themes and to remain distinct for the most common colour-vision
  // deficiencies (no red/green-only pairings).
  var PALETTE = [
    '#f59e0b', // amber
    '#3b82f6', // blue
    '#10b981', // emerald
    '#a855f7', // purple
    '#ec4899', // pink
    '#14b8a6', // teal
    '#f97316', // orange
    '#64748b'  // slate
  ];

  function isDark() {
    return document.documentElement.classList.contains('dark');
  }

  function readData() {
    var node = document.getElementById('cmp-chart-data');
    if (!node) return null;
    try {
      return JSON.parse(node.textContent);
    } catch (err) {
      console.error('LitmusLLM: could not parse chart data', err);
      return null;
    }
  }

  function build() {
    var canvas = document.getElementById('cmp-chart');
    var data = readData();

    if (chart) {          // always tear down, even if the canvas is gone now
      chart.destroy();
      chart = null;
    }
    if (!canvas || !data || !data.labels.length) return;

    var dark = isDark();
    var grid = dark ? 'rgba(148,163,184,0.15)' : 'rgba(100,116,139,0.15)';
    var text = dark ? '#cbd5e1' : '#475569';

    chart = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: data.metrics.map(function (metric, i) {
          return {
            label: metric.label + (metric.higher_is_better ? '' : ' (lower is better)'),
            data: metric.scores,
            backgroundColor: PALETTE[i % PALETTE.length],
            borderRadius: 4,
            borderSkipped: false,
            maxBarThickness: 56
          };
        })
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
            ticks: { color: text, stepSize: 0.2 },
            grid: { color: grid },
            title: { display: true, text: 'Average score (0-1)', color: text }
          },
          x: {
            ticks: { color: text, autoSkip: false, maxRotation: 30, minRotation: 0 },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { labels: { color: text, boxWidth: 12, usePointStyle: true } },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                if (ctx.parsed.y === null) return ctx.dataset.label + ': not scored';
                return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(3);
              }
            }
          }
        }
      }
    });
  }

  // Export the visible chart as a PNG. The canvas is composited onto an
  // opaque background first -- Chart.js leaves it transparent, which turns
  // into an unreadable black-on-black image in most viewers.
  window.downloadChart = function () {
    var canvas = document.getElementById('cmp-chart');
    if (!canvas) return;

    var out = document.createElement('canvas');
    out.width = canvas.width;
    out.height = canvas.height;
    var ctx = out.getContext('2d');
    ctx.fillStyle = isDark() ? '#0f172a' : '#ffffff';
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(canvas, 0, 0);

    var link = document.createElement('a');
    link.download = 'litmusllm-comparison.png';
    link.href = out.toDataURL('image/png');
    link.click();
  };

  window.litRedrawCharts = build;

  document.addEventListener('DOMContentLoaded', build);
  document.body.addEventListener('htmx:afterSettle', build);
})();
