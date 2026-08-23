/* Horizontal bar chart for the published-benchmark reference page.
 *
 * Deliberately a different chart type, orientation and palette from the
 * comparison chart on /compare. That visual distance is the point: these are
 * third-party figures on a 0-100 index, and they must never be mistaken for
 * the 0-1 measured scores shown elsewhere in the app.
 */
(function () {
  'use strict';

  var chart = null;

  function isDark() {
    return document.documentElement.classList.contains('dark');
  }

  function build() {
    var canvas = document.getElementById('ref-chart');
    var node = document.getElementById('ref-chart-data');

    if (chart) { chart.destroy(); chart = null; }
    if (!canvas || !node) return;

    var data;
    try {
      data = JSON.parse(node.textContent);
    } catch (err) {
      console.error('LitmusLLM: could not parse reference chart data', err);
      return;
    }
    if (!data.labels || !data.labels.length) return;

    var dark = isDark();
    var grid = dark ? 'rgba(148,163,184,0.15)' : 'rgba(100,116,139,0.15)';
    var text = dark ? '#cbd5e1' : '#475569';

    // Frontier vs open-weight get distinct hues; a model from a family the
    // user runs locally is highlighted in amber so it's findable at a glance.
    var colours = data.scores.map(function (_, i) {
      if (data.highlighted[i]) return '#f59e0b';
      return data.kinds[i] === 'frontier' ? '#a855f7' : '#14b8a6';
    });

    chart = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: [{
          label: 'Intelligence Index (0-100)',
          data: data.scores,
          backgroundColor: colours,
          borderRadius: 3,
          borderSkipped: false
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            ticks: { color: text },
            grid: { color: grid },
            title: { display: true, text: 'Index score (0-100) - published, not measured here', color: text }
          },
          y: {
            ticks: { color: text, font: { size: 11 } },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var kind = data.kinds[ctx.dataIndex] === 'frontier' ? 'frontier' : 'open weight';
                return ctx.parsed.x.toFixed(1) + ' / 100  (' + kind + ')';
              }
            }
          }
        }
      }
    });
  }

  window.litRedrawCharts = build;   // theme toggle hook
  document.addEventListener('DOMContentLoaded', build);
})();
