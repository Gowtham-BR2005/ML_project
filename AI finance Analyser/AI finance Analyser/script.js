document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("datasetForm");
  const tableBody = document.querySelector("#datasetTable tbody");

  const totalEl = document.getElementById("total");
  const avgEl = document.getElementById("average");
  const maxEl = document.getElementById("max");
  const minEl = document.getElementById("min");
  const growthEl = document.getElementById("growth");

  let dataset = [];

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const year = parseInt(document.getElementById("year").value);
    const value = parseFloat(document.getElementById("value").value);

    dataset.push({ year, value });
    dataset.sort((a, b) => a.year - b.year); // Sort by year
    renderTable();
    calculateStats();
    form.reset();
  });

  function renderTable() {
    tableBody.innerHTML = "";
    dataset.forEach((entry) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${entry.year}</td>
        <td>$${entry.value.toLocaleString()}</td>
      `;
      tableBody.appendChild(row);
    });
  }

  function calculateStats() {
    if (dataset.length === 0) return;

    const values = dataset.map(d => d.value);
    const total = values.reduce((a, b) => a + b, 0);
    const avg = total / values.length;
    const max = Math.max(...values);
    const min = Math.min(...values);

    const first = values[0];
    const last = values[values.length - 1];
    const growthRate = ((last - first) / first) * 100;

    totalEl.textContent = `$${total.toLocaleString()}`;
    avgEl.textContent = `$${avg.toFixed(2)}`;
    maxEl.textContent = `$${max.toLocaleString()}`;
    minEl.textContent = `$${min.toLocaleString()}`;
    growthEl.textContent = `${growthRate.toFixed(2)}%`;
  }
});
