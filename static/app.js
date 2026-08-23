/* Small enhancements shared across pages. */
(function () {
  'use strict';

  // The Models page links to /?model=local:llama3.1:latest -- preselect that
  // radio so "Evaluate" lands on a form that's already configured.
  document.addEventListener('DOMContentLoaded', function () {
    var wanted = new URLSearchParams(window.location.search).get('model');
    if (!wanted) return;
    var input = document.querySelector('input[name="model"][value="' + CSS.escape(wanted) + '"]');
    if (input && !input.disabled) {
      input.checked = true;
      input.closest('label').scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  });
})();
