/* ================================================================
   StudiPredict AI — main.js
   JavaScript utama untuk interaksi halaman
================================================================ */

// Aktifkan tooltip Bootstrap di seluruh halaman
document.addEventListener("DOMContentLoaded", function () {
  // Bootstrap Tooltips
  const tooltipEls = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltipEls.forEach(el => new bootstrap.Tooltip(el));

  // Animasi progress bar saat halaman dimuat
  document.querySelectorAll(".progress-bar").forEach(bar => {
    const target = bar.style.width;
    bar.style.width = "0%";
    setTimeout(() => {
      bar.style.transition = "width 1s ease";
      bar.style.width = target;
    }, 200);
  });
});
